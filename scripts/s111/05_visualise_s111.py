# -*- coding: utf-8 -*-
"""
05_visualise_s111.py
--------------------

Visualisation scientifique des trajectoires S-111 avec matplotlib.

Affiche :
- trajectoires lagrangiennes
- point de départ (vert)
- point d'arrivée (rouge)
- flèche de direction finale
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==========================================================
# Chemins
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "output"

zarr_files = sorted(OUT.glob("parcels_s111_*.zarr"))

if not zarr_files:
    raise FileNotFoundError("Aucun parcels_s111_*.zarr dans output/")

ZARR_FILE = zarr_files[-1]

print("Lecture :", ZARR_FILE)


# ==========================================================
# Chargement dataset Parcels
# ==========================================================

ds = xr.open_zarr(ZARR_FILE)

lons = ds.lon.values
lats = ds.lat.values
times = ds.time.values

n_traj = len(ds.trajectory)

t0 = np.nanmin(times)
t1 = np.nanmax(times)

duration_h = float((t1 - t0) / np.timedelta64(1, "h"))

valid_counts = (~np.isnan(lons)).sum(axis=1)

print(f"{n_traj} trajectoires | {duration_h:.1f} h")
print(f"Observations valides : min={valid_counts.min()}  max={valid_counts.max()}")
print(f"Trajectoires >= 2 points : {(valid_counts >= 2).sum()}")


# ==========================================================
# Création figure
# ==========================================================

fig, ax = plt.subplots(figsize=(12, 8))

pad_lon = 0.05
pad_lat = 0.03

lon_min = float(np.nanmin(lons)) - pad_lon
lon_max = float(np.nanmax(lons)) + pad_lon

lat_min = float(np.nanmin(lats)) - pad_lat
lat_max = float(np.nanmax(lats)) + pad_lat


ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

ax.set_facecolor("#cce5f0")

ax.set_xlabel("Longitude (°E)", fontsize=11)
ax.set_ylabel("Latitude (°N)", fontsize=11)

ax.grid(True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")


# ==========================================================
# Tracé trajectoires
# ==========================================================

n_plotted = 0

for i in range(n_traj):

    lon = lons[i]
    lat = lats[i]

    valid = np.isfinite(lon) & np.isfinite(lat)

    if valid.sum() < 2:
        continue

    lonv = lon[valid]
    latv = lat[valid]

    # trajectoire
    ax.plot(
        lonv,
        latv,
        color="royalblue",
        alpha=0.7,
        linewidth=1.5
    )

    # point départ
    ax.plot(
        lonv[0],
        latv[0],
        "o",
        color="lime",
        markersize=5,
        zorder=5
    )

    # point arrivée
    ax.plot(
        lonv[-1],
        latv[-1],
        "o",
        color="red",
        markersize=5,
        zorder=5
    )

    n_plotted += 1


print(f"{n_plotted} trajectoires tracées")


# ==========================================================
# Légende
# ==========================================================

legend_elements = [

    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="lime",
        markersize=9,
        label="Départ"
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        markerfacecolor="red",
        markersize=9,
        label="Arrivée"
    ),

    Line2D(
        [0],
        [0],
        color="royalblue",
        linewidth=2,
        label="Trajectoires"
    )

]

ax.legend(
    handles=legend_elements,
    loc="upper right",
    fontsize=10,
    framealpha=0.85
)


# ==========================================================
# Titre
# ==========================================================

ax.set_title(
    f"Trajectoires lagrangiennes – S-111 ({duration_h:.0f}h)\n"
    f"Baie de Seine | {n_plotted}/{n_traj} particules | Parcels 3.1.4",
    fontsize=13,
    fontweight="bold"
)


# ==========================================================
# Format axes
# ==========================================================

ax.xaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, _: f"{x:.3f}°")
)

ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda y, _: f"{y:.3f}°")
)

plt.xticks(rotation=30, fontsize=9)
plt.yticks(fontsize=9)


# ==========================================================
# Sauvegarde image
# ==========================================================

png_out = OUT / f"trajectoires_s111_{int(duration_h)}h.png"

plt.tight_layout()

plt.savefig(
    png_out,
    dpi=150,
    bbox_inches="tight"
)

print("Image sauvegardée :", png_out)


# ==========================================================
# Affichage
# ==========================================================

plt.show()