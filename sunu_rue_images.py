#!/usr/bin/env python3
"""
Sunu Rue Vision - mode images géolocalisées (Mapillary ou vos propres photos)

Entrée  : un dossier produit par mapillary_fetch.py (meta.csv + images/) et un modèle YOLO
Sorties : <out>/events.csv, <out>/events.geojson (pour la carte Leaflet), <out>/snapshots/,
          <out>/summary.json. Le champ « credit » conserve l'auteur de l'image (attribution).

    pip install ultralytics opencv-python
    python sunu_rue_images.py --data pikine --model nids.pt --out resultats
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import cv2
from ultralytics import YOLO


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def severity_from_area(ratio):
    return "elevee" if ratio >= 0.08 else "moyenne" if ratio >= 0.02 else "faible"


def cluster(dets, radius_m):
    """Fusionne les détections voisines (même anomalie vue sur plusieurs images) :
    on garde la plus confiante de chaque groupe."""
    kept = []
    for d in sorted(dets, key=lambda x: -x["conf"]):
        for k in kept:
            if k["classe"] == d["classe"] and haversine_m(k["lat"], k["lon"], d["lat"], d["lon"]) <= radius_m:
                k["vues"] += 1
                break
        else:
            d["vues"] = 1
            kept.append(d)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dossier contenant meta.csv et images/")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="resultats")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--keep", nargs="*", default=None,
                    help="classes à garder (ex. --keep pothole). Par défaut : toutes")
    ap.add_argument("--merge-radius-m", type=float, default=8.0,
                    help="détections plus proches que ça = même anomalie")
    a = ap.parse_args()

    os.makedirs(os.path.join(a.out, "snapshots"), exist_ok=True)
    model = YOLO(a.model)
    with open(os.path.join(a.data, "meta.csv"), newline="", encoding="utf-8") as f:
        metas = list(csv.DictReader(f))

    dets, n_img = [], 0
    for m in metas:
        path = os.path.join(a.data, m["fichier"].replace("\\", "/"))
        img = cv2.imread(path)
        if img is None:
            continue
        n_img += 1
        h, w = img.shape[:2]
        res = model.predict(img, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
        for box in res.boxes:
            if a.keep and model.names[int(box.cls[0])] not in a.keep:
                continue
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            ratio = ((x2 - x1) * (y2 - y1)) / float(w * h)
            dets.append({"classe": model.names[int(box.cls[0])], "conf": float(box.conf[0]),
                         "gravite": severity_from_area(ratio),
                         "lat": float(m["lat"]), "lon": float(m["lon"]),
                         "image_id": m["id"], "date": m.get("date", ""),
                         "credit": m.get("auteur", ""), "lien": m.get("lien", ""),
                         "box": (x1, y1, x2, y2), "path": path})

    events = cluster(dets, a.merge_radius_m)
    features, rows = [], []
    for i, e in enumerate(events, 1):
        img = cv2.imread(e["path"])
        x1, y1, x2, y2 = e["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        snap = os.path.join("snapshots", f"anomalie_{i}.jpg")
        cv2.imwrite(os.path.join(a.out, snap), img)
        props = {"id": i, "classe": e["classe"], "gravite": e["gravite"],
                 "confiance_max": round(e["conf"], 3), "images_vues": e["vues"],
                 "date_image": e["date"], "credit": e["credit"], "lien_source": e["lien"],
                 "capture": snap}
        rows.append({**props, "lat": round(e["lat"], 7), "lon": round(e["lon"], 7)})
        features.append({"type": "Feature",
                         "geometry": {"type": "Point", "coordinates": [round(e["lon"], 7), round(e["lat"], 7)]},
                         "properties": props})

    with open(os.path.join(a.out, "events.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["id", "classe", "gravite", "confiance_max", "images_vues", "date_image",
                "credit", "lien_source", "capture", "lat", "lon"]
        wr = csv.DictWriter(f, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(a.out, "events.geojson"), "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": features}, f, ensure_ascii=False)

    by = defaultdict(int)
    for r in rows:
        by[r["gravite"]] += 1
    summary = {"images_analysees": n_img, "detections_brutes": len(dets),
               "anomalies_uniques": len(rows), "repartition_gravite": dict(by),
               "seuil_confiance": a.conf, "rayon_fusion_m": a.merge_radius_m,
               "source_images": "Mapillary (CC BY-SA 4.0) - vérifier la licence de chaque image",
               "note": "Position = celle de la prise de vue ; l'anomalie est devant la caméra (quelques mètres). "
                       "Gravité = heuristique sur la taille à l'écran."}
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
