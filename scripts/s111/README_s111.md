# scripts/s111 — Pipeline S-111 → Trajectoires lagrangiennes → CZML

## Pipeline complet

```
donner3.h5  (S-111 HDF5)
    │
    ▼ 01_explore_s111.py        ← explorer la structure du fichier
    │
    ▼ 02_read_s111.py           ← lire et extraire U/V (bibliothèque)
    │
    ▼ 03_parcels_s111.py        ← simulation lagrangienne → .zarr
    │
    ▼ 04_s111_to_czml.py        ← conversion zarr → CZML statique + animé
    │
    ▼ 05_visualise_s111.py      ← carte matplotlib (debug/rapport)
    │
    ▼ index_s111.html           ← visualisation Cesium
```

## Ordre d'exécution

```bash
# 1. Explorer le fichier (une seule fois)
python scripts/s111/01_explore_s111.py

# 2. Vérifier la lecture
python scripts/s111/02_read_s111.py

# 3. Lancer la simulation Parcels
python scripts/s111/03_parcels_s111.py

# 4. Générer les CZML
python scripts/s111/04_s111_to_czml.py

# 5. Visualiser en matplotlib
python scripts/s111/05_visualise_s111.py

# 6. Copier les CZML + index_s111.html dans web/cesium-demo/
#    puis servir avec un serveur HTTP local :
python -m http.server 8080 --directory web/cesium-demo/
```

## Placer le fichier de données

Copiez `donner3.h5` (ou tout fichier S-111) dans :
```
data/s111/donner3.h5
```

## Structure S-111 attendue

Le format IHO S-111 organise les données ainsi :

```
/ (racine)
├── SurfaceCurrent/
│   └── SurfaceCurrent.01/
│       ├── @gridOriginLongitude
│       ├── @gridOriginLatitude
│       ├── @gridSpacing          (ou gridSpacingLongitudinal/Latitudinal)
│       ├── @numPointsLongitudinal
│       ├── @numPointsLatitudinal
│       ├── @dateTimeOfFirstRecord
│       ├── @timeRecordInterval   (secondes)
│       ├── Group_001/
│       │   └── values  [compound: surfaceCurrentSpeed, surfaceCurrentDirection]
│       ├── Group_002/
│       │   └── values
│       └── ...
```

## Dépendances Python

```
h5py
numpy
xarray
parcels
matplotlib
cartopy
```

## Différences avec le pipeline HYCOM

| Aspect | HYCOM (.nc) | S-111 (.h5) |
|--------|------------|-------------|
| Format | NetCDF | HDF5 |
| Variables | u, v (m/s) | speed (m/s) + direction (°) |
| Grille | curviligne 2D | régulière (attributs HDF5) |
| Temps | variable NetCDF | attributs + groupes |
| Lecture | xarray direct | h5py → numpy → xarray |
