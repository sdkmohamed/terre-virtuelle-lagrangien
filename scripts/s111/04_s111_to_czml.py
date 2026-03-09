# -*- coding: utf-8 -*-
"""
04_s111_to_czml.py
------------------
Conversion parcels_s111_*.zarr -> CZML statique + anime pour Cesium.
"""

from pathlib import Path
import numpy as np
import xarray as xr
import json

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "output"

zarr_files = sorted(OUT.glob("parcels_s111_*.zarr"))
if not zarr_files:
    raise FileNotFoundError("Aucun parcels_s111_*.zarr dans output/")
ZARR_FILE = zarr_files[-1]
print(f"Fichier zarr : {ZARR_FILE}")

CZML_STATIC   = OUT / (ZARR_FILE.stem + "_static.czml")
CZML_ANIMATED = OUT / (ZARR_FILE.stem + "_animated.czml")

# ========================================================== #
# Chargement                                                  #
# ========================================================== #
ds = xr.open_zarr(ZARR_FILE)
n_traj    = len(ds.trajectory)
lons_all  = ds.lon.values
lats_all  = ds.lat.values
times_all = ds.time.values

t0 = np.nanmin(times_all)
t1 = np.nanmax(times_all)
duration_h = float((t1 - t0) / np.timedelta64(1, "h"))
print(f"Trajectoires : {n_traj}  |  Duree : {duration_h:.1f} h")

def iso_z(t):
    s = np.datetime_as_string(t, unit="s")
    return s if s.endswith("Z") else s + "Z"

def secs(t, ref):
    return float((t - ref) / np.timedelta64(1, "s"))

interval_str = f"{iso_z(t0)}/{iso_z(t1)}"

# ========================================================== #
# CZML STATIQUE                                               #
# ========================================================== #
print("Generation CZML statique...")
packets_static = [{"id": "document", "name": "S-111 statique", "version": "1.0"}]
n_ok = 0
for i in range(n_traj):
    lon = lons_all[i]
    lat = lats_all[i]
    valid = np.isfinite(lon) & np.isfinite(lat)
    if valid.sum() < 2:
        continue
    coords = []
    for lo, la in zip(lon[valid], lat[valid]):
        coords += [float(lo), float(la), 10.0]
    packets_static.append({
        "id": f"traj_{i}",
        "polyline": {
            "width": 2,
            "material": {"solidColor": {"color": {"rgba": [0, 220, 255, 200]}}},
            "clampToGround": False,
            "positions": {"cartographicDegrees": coords}
        }
    })
    n_ok += 1

with open(CZML_STATIC, "w", encoding="utf-8") as f:
    json.dump(packets_static, f, indent=2)
print(f"  {n_ok} trajectoires -> {CZML_STATIC.name}")

# ========================================================== #
# CZML ANIME                                                  #
# ========================================================== #
print("Generation CZML anime...")
packets_anim = [{
    "id": "document",
    "name": "S-111 anime",
    "version": "1.0",
    "clock": {
        "interval": interval_str,
        "currentTime": iso_z(t0),
        "multiplier": 300,
        "range": "LOOP_STOP"
    }
}]
n_ok = 0
for i in range(n_traj):
    lon  = lons_all[i]
    lat  = lats_all[i]
    time = times_all[i]
    valid = np.isfinite(lon) & np.isfinite(lat)
    if valid.sum() < 2:
        continue
    flat_pos = []
    for lo, la, ti in zip(lon[valid], lat[valid], time[valid]):
        flat_pos += [secs(ti, t0), float(lo), float(la), 10.0]
    packets_anim.append({
        "id": f"traj_{i}",
        "availability": interval_str,
        "position": {
            "epoch": iso_z(t0),
            "cartographicDegrees": flat_pos,
            "interpolationAlgorithm": "LAGRANGE",
            "interpolationDegree": 1
        },
        "point": {
            "pixelSize": 7,
            "color": {"rgba": [0, 220, 255, 255]},
            "outlineColor": {"rgba": [0, 80, 120, 200]},
            "outlineWidth": 1
        },
        "path": {
            "resolution": 60,
            "width": 1.5,
            "leadTime": 0,
            "trailTime": 3600 * 3,
            "material": {"solidColor": {"color": {"rgba": [0, 220, 255, 150]}}}
        }
    })
    n_ok += 1

with open(CZML_ANIMATED, "w", encoding="utf-8") as f:
    json.dump(packets_anim, f, indent=2)
print(f"  {n_ok} trajectoires -> {CZML_ANIMATED.name}")

print("\nFichiers prets. Copiez-les dans web/cesium-demo/ et ouvrez index_s111.html")