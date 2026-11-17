#!/usr/bin/env python3
"""
Crée meta.csv à partir de VOS photos de téléphone (position GPS lue dans les données EXIF).

Organisation :
    mes_photos/images/IMG_0001.jpg, IMG_0002.jpg, ...      (photos originales, localisation activée)

    python exif_to_meta.py --data mes_photos --auteur "Madiou Sarr"
    -> crée mes_photos/meta.csv, utilisable ensuite avec sunu_rue_images.py --data mes_photos

Pillow est déjà installé sur Google Colab (sinon : pip install pillow).
Attention : WhatsApp, Messenger, etc. SUPPRIMENT la position des photos. Copiez les originaux par câble ou Drive.
Les photos sans GPS sont ignorées, sauf si vous donnez --lat/--lon (position de repli).
"""
import argparse
import csv
import glob
import os

from PIL import Image


def dms_to_deg(dms, ref):
    d, m, s = (float(x) for x in dms)
    v = d + m / 60.0 + s / 3600.0
    return -v if ref in ("S", "W") else v


def read_gps_date(path):
    ex = Image.open(path).getexif()
    gps = ex.get_ifd(0x8825)
    lat = lon = None
    if gps and 2 in gps and 4 in gps:
        lat = dms_to_deg(gps[2], gps.get(1, "N"))
        lon = dms_to_deg(gps[4], gps.get(3, "E"))
    date = ex.get_ifd(0x8769).get(36867) or ex.get(306) or ""
    return lat, lon, str(date)[:10].replace(":", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dossier contenant images/")
    ap.add_argument("--auteur", default="")
    ap.add_argument("--lat", type=float, default=None, help="position de repli si une photo n'a pas de GPS")
    ap.add_argument("--lon", type=float, default=None)
    a = ap.parse_args()

    files = sorted(f for ext in ("jpg", "jpeg", "JPG", "JPEG")
                   for f in glob.glob(os.path.join(a.data, "images", "*." + ext)))
    ok = skipped = 0
    with open(os.path.join(a.data, "meta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "fichier", "lat", "lon", "date", "sequence", "auteur", "lien"])
        for p in files:
            try:
                lat, lon, date = read_gps_date(p)
            except Exception as e:
                print("  illisible :", os.path.basename(p), e)
                skipped += 1
                continue
            if lat is None and a.lat is not None:
                lat, lon = a.lat, a.lon
            if lat is None:
                print("  pas de GPS :", os.path.basename(p))
                skipped += 1
                continue
            name = os.path.basename(p)
            w.writerow([os.path.splitext(name)[0], "images/" + name, round(lat, 7), round(lon, 7),
                        date, "", a.auteur, ""])
            ok += 1
    print(f"{ok} photo(s) géolocalisée(s), {skipped} ignorée(s). -> {a.data}/meta.csv")


if __name__ == "__main__":
    main()
