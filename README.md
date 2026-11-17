# Sunu Rue Vision

**Détecter, localiser et prioriser les dégradations des routes de la banlieue de Dakar grâce à la vision par ordinateur.**

Projet présenté à **Banlieue Smart Days** (2e édition, Thiaroye, 12-13 décembre 2026), catégorie *Intelligence & Machines*.

> **Statut : prototype.** La chaîne complète fonctionne (images géolocalisées → détection → fusion des doublons → gravité → carte). Le modèle de détection vient de données publiques internationales ; son adaptation aux routes de Dakar est une hypothèse en cours de vérification (voir [Résultats](#résultats-mesurés) et [Limites](#limites)).

![Carte des anomalies](docs/carte.png)

## Le problème

Dans la banlieue de Dakar, les nids-de-poule et les chaussées dégradées apparaissent vite et les communes manquent d'une vue d'ensemble : **où** sont les dégradations, **lesquelles sont les plus graves**, **lesquelles sont réparées**. Les travaux existants sur l'IA et la sécurité routière au Sénégal s'intéressent surtout à l'alerte du conducteur dans le véhicule. Ce projet vise le **gestionnaire de la route**.

## La solution

```
photos / vidéo + GPS ──► détection (YOLO) ──► fusion des doublons ──► gravité + priorité ──► carte web
```

1. **Acquisition** : images de rues Mapillary, photos de téléphone (GPS lu dans les données EXIF) ou vidéo avec trace GPX.
2. **Détection** : un modèle YOLO repère les anomalies sur chaque image.
3. **Fusion** : une même anomalie vue sur plusieurs images n'est comptée qu'une fois (rayon de 8 m par défaut).
4. **Gravité** : faible, moyenne ou élevée, d'après la taille de l'anomalie à l'écran (heuristique, pas une mesure en centimètres).
5. **Carte** : tableau de bord Leaflet avec filtres, captures, suivi de statut (signalé, en cours, résolu), liste de priorité et export CSV.

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `mapillary_fetch.py` | Télécharge des images de rues Mapillary autour d'un point (sans dépendance, bibliothèque standard) |
| `exif_to_meta.py` | Crée `meta.csv` à partir de vos propres photos géolocalisées |
| `sunu_rue_images.py` | Détection sur un dossier d'images : `events.csv`, `events.geojson`, captures, `summary.json` |
| `sunu_rue_vision.py` | Détection sur une vidéo, avec suivi ByteTrack et trace GPS (GPX ou CSV) |
| `sunu_rue_colab.ipynb` | Notebook Google Colab pour lancer la détection sans installation locale |
| `carte/index.html` | Carte web Leaflet qui lit `events.geojson` |
| `carte/events_demo.geojson` | Données **fictives** de démonstration (bandeau d'avertissement dans la carte) |
| `docs/` | Captures d'écran et exemples de résultats |

## Utilisation

### Option A : images de rues Mapillary

```bash
# 1. Un jeton Mapillary gratuit : mapillary.com/developer (Client Token)
#    Ne jamais l'écrire dans un fichier du dépôt.
export MAPILLARY_TOKEN="..."          # PowerShell : $env:MAPILLARY_TOKEN="..."

# 2. Télécharger des images autour d'un point (latitude, longitude)
python mapillary_fetch.py --center LAT LON --radius-m 700 --max 150 --out pikine

# 3. Détecter (voir le notebook Colab si l'installation locale est impossible)
pip install ultralytics opencv-python
python sunu_rue_images.py --data pikine --model poids.pt --out resultats --conf 0.25
```

### Option B : vos propres photos

Mettez vos photos **originales** (localisation activée) dans `mes_photos/images/`. Les messageries comme WhatsApp suppriment la position : copiez-les par câble ou Drive.

```bash
python exif_to_meta.py --data mes_photos --auteur "Votre nom"
python sunu_rue_images.py --data mes_photos --model poids.pt --out resultats
```

### Option C : vidéo avec trace GPS

```bash
python sunu_rue_vision.py --video rue.mp4 --model poids.pt --gps trace.gpx --out resultats
```

### Le modèle de détection

Le projet a été testé avec le modèle public [`vinothvikas1987/pothole-detection-yolov8`](https://huggingface.co/vinothvikas1987/pothole-detection-yolov8) (licence Apache 2.0). Ses classes incluent `Pothole` ; utilisez `--keep Pothole` dans `sunu_rue_images.py` pour ne garder que les nids-de-poule.

```python
from huggingface_hub import hf_hub_download
poids = hf_hub_download("vinothvikas1987/pothole-detection-yolov8", "best.pt")
```

### Afficher la carte

Copiez `carte/index.html` dans le dossier `resultats/` (à côté de `events.geojson` et `snapshots/`), puis :

```bash
cd resultats
python -m http.server 8000
# ouvrir http://localhost:8000
```

Le bouton **Démo** charge `carte/events_demo.geojson` (données fictives, signalées par un bandeau). Les statuts de réparation sont conservés dans le navigateur, sans serveur. La carte a besoin d'une connexion internet (fond OpenStreetMap, bibliothèque Leaflet).

## Résultats mesurés

Ces résultats sont **préliminaires** : peu d'images, aucune évaluation à grande échelle.

| Test | Données | Résultat |
|---|---|---|
| A. Photos locales de nids-de-poule | [N] photos à [quartier], [date] | [X] détectés sur [N] présents, [Y] fausses détections |
| B. Route en bon état (fausses alertes) | 150 images Mapillary (autoroute récente, Pikine/Thiaroye) | [nombre de détections à seuil [S]] |
| C. Jeu public annoté | [nom du jeu, licence], [N] images | précision [P], rappel [R], mAP50 [M] |

## Limites

- Le modèle a été entraîné sur des données publiques internationales. Les routes en latérite ou en sable, les trous remplis d'eau, la poussière et les ralentisseurs peuvent dégrader les résultats à Dakar.
- Les images Mapillary de la zone testée montrent surtout des autoroutes neuves, souvent de nuit. Elles servent de test de fausses alertes, pas de démonstration de détection.
- La gravité est estimée par la taille à l'écran, pas par une mesure de profondeur.
- La position est celle de la prise de vue (précision GPS d'un téléphone : quelques mètres) ; l'anomalie est quelques mètres devant la caméra.

## Suite prévue (pilote de 4 semaines)

1. Collecter 300 à 500 photos géolocalisées dans 2 ou 3 quartiers de Thiaroye et Pikine.
2. Les annoter et ajuster le modèle sur ces données locales.
3. Mesurer précision et rappel sur des images locales non vues pendant l'entraînement.
4. Présenter la carte à la commune et fixer les priorités de déploiement.

Autres pistes : application de signalement citoyen hors-ligne, détection des eaux stagnantes et des caniveaux bouchés, capteurs de niveau d'eau à bas coût.

## Vie privée

Le système traite des routes, pas des personnes. Aucune lecture de plaque d'immatriculation n'est incluse. Toute fonction de contrôle des infractions exigerait l'accord des autorités, un cadre juridique validé et un avis de la CDP.

## Licences et attributions

- Code : licence MIT (voir `LICENSE`).
- Images de rues : © contributeurs Mapillary, licence CC BY-SA 4.0. Les auteurs sont conservés dans le champ `credit` des résultats. Ces images ne sont pas redistribuées dans ce dépôt.
- Modèle : `vinothvikas1987/pothole-detection-yolov8` (Apache 2.0), construit avec [Ultralytics YOLO](https://github.com/ultralytics/ultralytics).
- Fond de carte : © contributeurs OpenStreetMap. Bibliothèque cartographique : [Leaflet](https://leafletjs.com).

## Auteur

**Madiou Sarr**, étudiant en L2 Ingénierie Informatique (Systèmes d'information, Data & Sécurité), Polytech Diamniadio, Université Amadou Mahtar Mbow, Dakar.
GitHub : [Madjilouh](https://github.com/Madjilouh) · Contact : [email]
