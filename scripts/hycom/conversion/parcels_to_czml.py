# -*- coding: utf-8 -*-
"""
Conversion Parcels (.zarr) -> CZML (Cesium)
Version STATIQUE : polylines complètes (comme l'image du prof)
"""

from pathlib import Path
import xarray as xr
import json
import numpy as np

PROJECT = Path(__file__).resolve().parents[3]

OUT = PROJECT / "output"

ZARR_FILE = OUT / "parcels_hycom_bretagne_23h.zarr"
CZML_FILE = OUT / "parcels_hycom_bretagne_23h_static.czml"

print("📁 Lecture :", ZARR_FILE)
ds = xr.open_zarr(ZARR_FILE)

packets = [{
    "id": "document",
    "name": "Trajectoires Lagrangiennes (statique)",
    "version": "1.0"
}]

for i in range(len(ds.trajectory)):
    lon = ds.lon.values[i]
    lat = ds.lat.values[i]
    valid = np.isfinite(lon) & np.isfinite(lat)
    if valid.sum() < 2:
        continue

    line_pos = []
    for lo, la in zip(lon[valid], lat[valid]):
        line_pos += [float(lo), float(la), 0.0]

    packets.append({
        "id": f"traj_{i}",
        "polyline": {
            "width": 2,
            "material": {
                "solidColor": {
                    "color": {"rgba": [0, 255, 255, 200]}
                }
            },
            "positions": {
                "cartographicDegrees": line_pos
            }
        }
    })

with open(CZML_FILE, "w", encoding="utf-8") as f:
    json.dump(packets, f, indent=2)

print("✅ CZML statique généré :", CZML_FILE)
