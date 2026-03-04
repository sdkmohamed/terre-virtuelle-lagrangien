# -*- coding: utf-8 -*-
"""
Conversion Parcels (.zarr) -> CZML ANIMÉ (Cesium)
2 particules: point + path (pour voir les inversions)
"""

from pathlib import Path
import xarray as xr
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "output"

ZARR_FILE = OUT / "parcels_hycom_bretagne_23h.zarr"
CZML_FILE = OUT / "parcels_hycom_bretagne_23h_animated.czml"

print("📁 Lecture :", ZARR_FILE)
ds = xr.open_zarr(ZARR_FILE)

times = ds.time.values
t0 = np.nanmin(times)

def seconds_since_t0(t):
    return float((t - t0) / np.timedelta64(1, "s"))

def iso_z(t):
    s = np.datetime_as_string(t, unit="s")
    return s + "Z" if not s.endswith("Z") else s

t_end = np.nanmax(times)
interval = f"{iso_z(t0)}/{iso_z(t_end)}"

packets = [{
    "id": "document",
    "name": "Trajectoires Lagrangiennes (animé)",
    "version": "1.0",
    "clock": {
        "interval": interval,
        "currentTime": iso_z(t0),
        "multiplier": 600,
        "range": "LOOP_STOP"
    }
}]

for i in range(len(ds.trajectory)):
    lon = ds.lon.values[i]
    lat = ds.lat.values[i]
    t = ds.time.values[i]

    valid = np.isfinite(lon) & np.isfinite(lat)
    if valid.sum() < 2:
        continue

    flat_pos = []
    for lo, la, ti in zip(lon[valid], lat[valid], t[valid]):
        flat_pos += [seconds_since_t0(ti), float(lo), float(la), 0.0]

    packets.append({
        "id": f"traj_{i}",
        "availability": interval,
        "position": {
            "epoch": iso_z(t0),
            "cartographicDegrees": flat_pos
        },
        "point": {
            "pixelSize": 8,
            "color": {"rgba": [0, 255, 255, 255]}
        },
        "path": {
            "resolution": 60,
            "width": 2,
            "leadTime": 0,
            "trailTime": 3600 * 24,
            "material": {
                "solidColor": {"color": {"rgba": [0, 255, 255, 200]}}
            }
        }
    })

with open(CZML_FILE, "w", encoding="utf-8") as f:
    json.dump(packets, f, indent=2)

print("✅ CZML animé généré :", CZML_FILE)
