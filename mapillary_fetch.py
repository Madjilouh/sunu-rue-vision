import argparse, csv, json, math, os, sys, time
import urllib.error, urllib.parse, urllib.request
from datetime import datetime, timezone

API = "https://graph.mapillary.com/images"
FIELDS = "id,captured_at,computed_geometry,geometry,thumb_2048_url,is_pano,sequence,creator"


def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "sunu-rue-vision/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def dist_m(la1, lo1, la2, lo2):
    p1, p2 = math.radians(la1), math.radians(la2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 12742000 * math.asin(math.sqrt(a))


ap = argparse.ArgumentParser()
ap.add_argument("--token", default=os.environ.get("MAPILLARY_TOKEN"))
ap.add_argument("--center", nargs=2, type=float, required=True, metavar=("LAT", "LON"))
ap.add_argument("--radius-m", type=float, default=700)
ap.add_argument("--max", type=int, default=150)
ap.add_argument("--min-spacing-m", type=float, default=6)
ap.add_argument("--since", type=int, default=None)
ap.add_argument("--out", default="mapillary_out")
a = ap.parse_args()
if not a.token:
    sys.exit("Jeton manquant : definissez MAPILLARY_TOKEN")

lat0, lon0 = a.center
dlat = a.radius_m / 111320.0
dlon = a.radius_m / (111320.0 * math.cos(math.radians(lat0)))
bbox = (lon0 - dlon, lat0 - dlat, lon0 + dlon, lat0 + dlat)
if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) >= 0.01:
    sys.exit("Zone trop grande : reduisez --radius-m")

params = {"access_token": a.token, "bbox": ",".join("%.6f" % v for v in bbox),
          "fields": FIELDS, "limit": 2000, "is_pano": "false"}
print("Requete Mapillary...")
code, body = http_get(API + "?" + urllib.parse.urlencode(params))
if code != 200:
    sys.exit("Erreur API %s : %s" % (code, body[:300].decode("utf-8", "replace")))
data = json.loads(body).get("data", [])
print("%d image(s) trouvee(s)." % len(data))

rows = []
for it in data:
    g = it.get("computed_geometry") or it.get("geometry")
    if not g or not it.get("thumb_2048_url") or it.get("is_pano"):
        continue
    ts = it.get("captured_at") or 0
    year = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).year if ts else None
    if a.since and (year is None or year < a.since):
        continue
    cr = it.get("creator") or {}
    rows.append({"id": it["id"], "lon": g["coordinates"][0], "lat": g["coordinates"][1], "ts": ts,
                 "seq": it.get("sequence", ""), "url": it["thumb_2048_url"],
                 "creator": cr.get("username", "") if isinstance(cr, dict) else ""})

rows.sort(key=lambda r: (r["seq"], r["ts"]))
kept, last = [], {}
for r in rows:
    p = last.get(r["seq"])
    if p is None or dist_m(p["lat"], p["lon"], r["lat"], r["lon"]) >= a.min_spacing_m:
        kept.append(r)
        last[r["seq"]] = r
kept = kept[:a.max]
print("%d image(s) retenue(s)." % len(kept))

os.makedirs(os.path.join(a.out, "images"), exist_ok=True)
with open(os.path.join(a.out, "meta.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "fichier", "lat", "lon", "date", "sequence", "auteur", "lien"])
    for i, r in enumerate(kept, 1):
        rel = os.path.join("images", "%s.jpg" % r["id"])
        try:
            c, content = http_get(r["url"])
            if c != 200:
                raise RuntimeError("HTTP %s" % c)
            with open(os.path.join(a.out, rel), "wb") as out:
                out.write(content)
        except Exception as e:
            print("  echec %s : %s" % (r["id"], e))
            continue
        d = datetime.fromtimestamp(r["ts"] / 1000, tz=timezone.utc).date().isoformat() if r["ts"] else ""
        w.writerow([r["id"], rel, r["lat"], r["lon"], d, r["seq"], r["creator"],
                    "https://www.mapillary.com/app/?pKey=%s" % r["id"]])
        if i % 20 == 0:
            print("  %d/%d" % (i, len(kept)))
        time.sleep(0.1)
print("Termine : %s/images et %s/meta.csv" % (a.out, a.out))
print("Attribution a citer : Images (c) contributeurs Mapillary, CC BY-SA 4.0")
