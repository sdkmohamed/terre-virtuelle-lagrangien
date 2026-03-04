# -*- coding: utf-8 -*-
"""
Parcels (.zarr) -> CZML statique (polylines) + flèches directionnelles (cone)
- Une polyline par trajectoire
- Une flèche 3D à l'extrémité (cone) orientée selon le dernier segment
"""

from pathlib import Path
import xarray as xr
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "output"

ZARR_FILE = OUT / "parcels_hycom_bretagne_23h.zarr"
CZML_FILE = OUT / "parcels_hycom_bretagne_23h_static_arrows.czml"

print(" Lecture :", ZARR_FILE)
ds = xr.open_zarr(ZARR_FILE)

# ---- utils: conversion cartographique -> cartésien (ECEF) pour orientation ----
def cartesian_from_lonlat(lon_deg, lat_deg, h=0.0):
    # WGS84 approx radius
    a = 6378137.0
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    x = (a + h) * np.cos(lat) * np.cos(lon)
    y = (a + h) * np.cos(lat) * np.sin(lon)
    z = (a + h) * np.sin(lat)
    return np.array([x, y, z], dtype=float)

def normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def quat_from_two_vectors(v0, v1):
    """
    Quaternion that rotates v0 -> v1.
    Output as [x,y,z,w] (Cesium quaternion order).
    """
    v0 = normalize(v0)
    v1 = normalize(v1)
    c = np.cross(v0, v1)
    d = float(np.dot(v0, v1))
    if d < -0.999999:
        # opposite vectors: pick an orthogonal axis
        axis = normalize(np.cross(v0, np.array([1.0, 0.0, 0.0])))
        if np.linalg.norm(axis) < 1e-6:
            axis = normalize(np.cross(v0, np.array([0.0, 1.0, 0.0])))
        # 180° rotation: w=0
        return [float(axis[0]), float(axis[1]), float(axis[2]), 0.0]
    s = np.sqrt((1.0 + d) * 2.0)
    invs = 1.0 / s
    qx = float(c[0] * invs)
    qy = float(c[1] * invs)
    qz = float(c[2] * invs)
    qw = float(s * 0.5)
    return [qx, qy, qz, qw]

packets = [{
    "id": "document",
    "name": "Trajectoires Lagrangiennes (statique + flèches)",
    "version": "1.0"
}]

LINE_RGBA = [0, 255, 255, 200]
ARROW_RGBA = [0, 255, 255, 255]

# On oriente le cône depuis l'axe Z local (approx) vers la direction du mouvement.
# Ici on prend une flèche orientée selon la direction ECEF du dernier segment.
BASE_AXIS = np.array([0.0, 0.0, 1.0], dtype=float)

for i in range(len(ds.trajectory)):
    lon = ds.lon.values[i]
    lat = ds.lat.values[i]
    valid = np.isfinite(lon) & np.isfinite(lat)
    if valid.sum() < 3:
        continue

    lonv = lon[valid]
    latv = lat[valid]

    # ---- polyline complète ----
    line_pos = []
    for lo, la in zip(lonv, latv):
        line_pos += [float(lo), float(la), 0.0]

    packets.append({
        "id": f"traj_{i}",
        "polyline": {
            "width": 2,
            "material": {"solidColor": {"color": {"rgba": LINE_RGBA}}},
            "positions": {"cartographicDegrees": line_pos}
        }
    })

    # ---- flèche à l'extrémité (cone) ----
    lon_end, lat_end = float(lonv[-1]), float(latv[-1])
    lon_prev, lat_prev = float(lonv[-2]), float(latv[-2])

    p_end = cartesian_from_lonlat(lon_end, lat_end, 0.0)
    p_prev = cartesian_from_lonlat(lon_prev, lat_prev, 0.0)

    direction = p_end - p_prev
    if np.linalg.norm(direction) < 1e-6:
        continue

    q = quat_from_two_vectors(BASE_AXIS, direction)

    packets.append({
        "id": f"traj_{i}_arrow",
        "position": {"cartographicDegrees": [lon_end, lat_end, 0.0]},
        "orientation": {"unitQuaternion": q},
        "cylinder": {
            # un cône "flèche": longueur + rayon
            # (cylinder dans CZML est en fait cylindre; on l'utilise comme flèche simple)
            "length": 2000.0,        # 2 km (ajuste si besoin)
            "topRadius": 0.0,        # pointe
            "bottomRadius": 300.0,   # base
            "material": {"solidColor": {"color": {"rgba": ARROW_RGBA}}}
        }
    })

with open(CZML_FILE, "w", encoding="utf-8") as f:
    json.dump(packets, f)

print(" CZML statique + flèches généré :", CZML_FILE)
