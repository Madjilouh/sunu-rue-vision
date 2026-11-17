#!/usr/bin/env python3
"""
Sunu Rue Vision - détection d'anomalies routières sur vidéo, avec localisation GPS exacte

Entrée  : - une vidéo filmée dans la rue (téléphone fixé face à la route)
          - la trace GPS enregistrée PENDANT le tournage (fichier .gpx ou .csv)
Sorties : <out>/annotated.mp4    vidéo annotée (cadres, gravité, coordonnées)
          <out>/detections.csv   une ligne par détection, avec lat/lon
          <out>/events.csv       une ligne par anomalie unique, avec lat/lon
          <out>/events.geojson   à charger directement dans la carte Leaflet
          <out>/snapshots/       une capture par anomalie unique
          <out>/summary.json     chiffres pour le dossier

Installation :
    pip install ultralytics opencv-python

Exemples :
    # trace GPX enregistrée en même temps que la vidéo
    python sunu_rue_vision.py --video rue.mp4 --model nids.pt --gps trace.gpx --out resultats

    # trace GPX + heure exacte de début de la vidéo (UTC)
    python sunu_rue_vision.py --video rue.mp4 --model nids.pt --gps trace.gpx \
        --video-start 2026-11-17T09:32:10 --out resultats

    # sans trace : position fixe pour toute la vidéo
    python sunu_rue_vision.py --video rue.mp4 --model nids.pt --lat 14.76 --lon -17.34

Format CSV de trace (temps en secondes depuis le début de la vidéo) :
    time_s,lat,lon
    0,14.76012,-17.34120
    5,14.76020,-17.34105
"""
import argparse
import bisect
import csv
import json
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import cv2
from ultralytics import YOLO


# ----------------------------------------------------------------------------
# Localisation
# ----------------------------------------------------------------------------
class GPSTrack:
    """Trace GPS (GPX ou CSV) avec interpolation linéaire entre deux points."""

    def __init__(self, path, video_start=None, offset=0.0):
        self.times, self.lats, self.lons = [], [], []
        self.is_gpx = path.lower().endswith(".gpx")
        if self.is_gpx:
            self._load_gpx(path)
        else:
            self._load_csv(path)
        if len(self.times) < 2:
            raise SystemExit("La trace GPS doit contenir au moins 2 points.")
        self.offset = offset
        # instant correspondant à t=0 de la vidéo
        if self.is_gpx:
            self.video_t0 = video_start.timestamp() if video_start else self.times[0]
        else:
            self.video_t0 = 0.0

    @staticmethod
    def _parse_time(s):
        dt = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt

    def _load_gpx(self, path):
        root = ET.parse(path).getroot()
        for el in root.iter():
            if el.tag.endswith("trkpt") or el.tag.endswith("rtept"):
                t_el = next((c for c in el if c.tag.endswith("time")), None)
                if t_el is None or not t_el.text:
                    continue
                self.times.append(self._parse_time(t_el.text).timestamp())
                self.lats.append(float(el.attrib["lat"]))
                self.lons.append(float(el.attrib["lon"]))

    def _load_csv(self, path):
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.times.append(float(row["time_s"]))
                self.lats.append(float(row["lat"]))
                self.lons.append(float(row["lon"]))

    def at(self, video_t, tolerance=3.0):
        """Position (lat, lon) à l'instant video_t (secondes). None si hors de la trace."""
        target = self.video_t0 + self.offset + video_t
        ts = self.times
        if target < ts[0] - tolerance or target > ts[-1] + tolerance:
            return None
        if target <= ts[0]:
            return self.lats[0], self.lons[0]
        if target >= ts[-1]:
            return self.lats[-1], self.lons[-1]
        i = bisect.bisect_right(ts, target)
        t0, t1 = ts[i - 1], ts[i]
        k = 0.0 if t1 == t0 else (target - t0) / (t1 - t0)
        return (self.lats[i - 1] + k * (self.lats[i] - self.lats[i - 1]),
                self.lons[i - 1] + k * (self.lons[i] - self.lons[i - 1]))


def severity_from_area(area_ratio: float) -> str:
    """Gravité estimée d'après la part de l'image occupée par l'anomalie.
    Heuristique simple, à présenter comme telle (ce n'est pas une mesure en cm)."""
    if area_ratio >= 0.08:
        return "elevee"
    if area_ratio >= 0.02:
        return "moyenne"
    return "faible"


def main():
    p = argparse.ArgumentParser(description="Sunu Rue Vision - anomalies routières géolocalisées")
    p.add_argument("--video", required=True)
    p.add_argument("--model", required=True, help="poids YOLO (.pt) pour les nids-de-poule")
    p.add_argument("--out", default="resultats")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--min-frames", type=int, default=3,
                   help="une anomalie est confirmée si elle est vue sur au moins N images")
    # localisation
    p.add_argument("--gps", default=None, help="trace GPS : fichier .gpx ou .csv (time_s,lat,lon)")
    p.add_argument("--video-start", default=None,
                   help="heure de début de la vidéo en UTC, ex. 2026-11-17T09:32:10 "
                        "(sinon : la vidéo est supposée démarrer au premier point du GPX)")
    p.add_argument("--gps-offset", type=float, default=0.0,
                   help="décalage en secondes si la vidéo et le GPS ne sont pas synchronisés")
    p.add_argument("--lat", type=float, default=None, help="position fixe (si pas de trace GPS)")
    p.add_argument("--lon", type=float, default=None)
    args = p.parse_args()

    os.makedirs(os.path.join(args.out, "snapshots"), exist_ok=True)

    track_gps = None
    if args.gps:
        vs = None
        if args.video_start:
            vs = datetime.fromisoformat(args.video_start)
            if vs.tzinfo is None:
                vs = vs.replace(tzinfo=timezone.utc)
        track_gps = GPSTrack(args.gps, video_start=vs, offset=args.gps_offset)
        gps_source = f"trace {os.path.basename(args.gps)}"
    elif args.lat is not None and args.lon is not None:
        gps_source = "position fixe"
    else:
        gps_source = "aucune"

    def locate(t_sec):
        if track_gps is not None:
            return track_gps.at(t_sec)
        if args.lat is not None and args.lon is not None:
            return args.lat, args.lon
        return None

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Impossible d'ouvrir la vidéo : {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    writer = cv2.VideoWriter(os.path.join(args.out, "annotated.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    tracks = defaultdict(lambda: {"frames": 0, "best_conf": 0.0, "best_frame": None,
                                  "best_t": None, "class": None,
                                  "severity": "faible", "best_area": 0.0})
    det_rows = []
    frame_idx = 0

    results = model.track(source=args.video, conf=args.conf, imgsz=args.imgsz,
                          tracker="bytetrack.yaml", persist=True, stream=True, verbose=False)

    for r in results:
        frame = r.orig_img.copy()
        t_sec = frame_idx / fps
        pos = locate(t_sec)
        lat, lon = (pos if pos else (None, None))

        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            ids = boxes.id.int().tolist() if boxes.id is not None else [None] * len(boxes)
            for box, tid in zip(boxes, ids):
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                conf = float(box.conf[0])
                cls_name = model.names[int(box.cls[0])]
                area_ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
                sev = severity_from_area(area_ratio)

                det_rows.append([frame_idx, round(t_sec, 2), tid, cls_name, round(conf, 3),
                                 x1, y1, x2, y2, round(area_ratio, 4), sev,
                                 None if lat is None else round(lat, 7),
                                 None if lon is None else round(lon, 7)])

                if tid is not None:
                    tr = tracks[tid]
                    tr["frames"] += 1
                    tr["class"] = cls_name
                    if conf > tr["best_conf"]:
                        # position retenue = celle de la meilleure image (anomalie la plus nette)
                        tr.update(best_conf=conf, best_frame=frame.copy(), best_t=t_sec)
                    if area_ratio > tr["best_area"]:
                        tr["best_area"] = area_ratio
                        tr["severity"] = sev

                color = {"faible": (0, 200, 0), "moyenne": (0, 165, 255), "elevee": (0, 0, 255)}[sev]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID{tid if tid is not None else '?'} {cls_name} {conf:.2f} [{sev}]"
                cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, color, 2, cv2.LINE_AA)

        n_confirmed = sum(1 for t in tracks.values() if t["frames"] >= args.min_frames)
        cv2.putText(frame, f"Anomalies confirmees : {n_confirmed}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        gps_txt = f"GPS {lat:.6f}, {lon:.6f}" if lat is not None else "GPS : indisponible"
        cv2.putText(frame, gps_txt, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(frame)
        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  image {frame_idx}/{total}")

    writer.release()

    # --- detections.csv
    with open(os.path.join(args.out, "detections.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["frame", "temps_s", "track_id", "classe", "confiance",
                     "x1", "y1", "x2", "y2", "part_image", "gravite", "lat", "lon"])
        wr.writerows(det_rows)

    # --- events.csv, events.geojson, snapshots
    events, features = [], []
    for tid, tr in sorted(tracks.items()):
        if tr["frames"] < args.min_frames:
            continue
        pos = locate(tr["best_t"]) if tr["best_t"] is not None else None
        lat, lon = (pos if pos else (None, None))
        snap = None
        if tr["best_frame"] is not None:
            snap = os.path.join("snapshots", f"anomalie_{tid}.jpg")
            cv2.imwrite(os.path.join(args.out, snap), tr["best_frame"])
        ev = {
            "id": tid,
            "classe": tr["class"],
            "gravite": tr["severity"],
            "confiance_max": round(tr["best_conf"], 3),
            "temps_video_s": round(tr["best_t"], 2),
            "images_vues": tr["frames"],
            "lat": None if lat is None else round(lat, 7),
            "lon": None if lon is None else round(lon, 7),
            "capture": snap,
        }
        events.append(ev)
        if lat is not None:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(lon, 7), round(lat, 7)]},
                "properties": {k: v for k, v in ev.items() if k not in ("lat", "lon")},
            })

    with open(os.path.join(args.out, "events.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["id", "classe", "gravite", "confiance_max", "temps_video_s",
                "images_vues", "lat", "lon", "capture"]
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(events)

    with open(os.path.join(args.out, "events.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False)

    # --- summary.json
    by_sev = defaultdict(int)
    for e in events:
        by_sev[e["gravite"]] += 1
    summary = {
        "video": os.path.basename(args.video),
        "modele": os.path.basename(args.model),
        "duree_s": round(frame_idx / fps, 1),
        "images_analysees": frame_idx,
        "seuil_confiance": args.conf,
        "source_localisation": gps_source,
        "anomalies_uniques_confirmees": len(events),
        "anomalies_localisees": len(features),
        "repartition_gravite": dict(by_sev),
        "confiance_moyenne": round(sum(e["confiance_max"] for e in events) / len(events), 3) if events else None,
        "note": "Gravité = heuristique (taille à l'écran). Précision GPS d'un téléphone : "
                "quelques mètres ; l'anomalie est devant le véhicule, donc légèrement en avant du point GPS.",
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nTerminé.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Résultats dans : {args.out}/")


if __name__ == "__main__":
    main()
