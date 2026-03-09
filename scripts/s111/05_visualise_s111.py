# -*- coding: utf-8 -*-
"""
05_visualise_s111.py
--------------------
Carte matplotlib des trajectoires S-111.
Version sans cartopy (utilise matplotlib + contextily optionnel).
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
OUT  = ROOT / "output"

zarr_files = sorted(OUT.glob("parcels_s111_*.zarr"))
if not zarr_files:
    raise FileNotFoundError("Aucun parcels_s111_*.zarr dans output/")
ZARR_FILE = zarr_files[-1]
print(f"Lecture : {ZARR_FILE}")

ds = xr.open_zarr(ZARR_FILE)
n_traj = len(ds.trajectory)
lons   = ds.lon.values
lats   = ds.lat.values
times  = ds.time.values

t0 = np.nanmin(times)
t1 = np.nanmax(times)
duration_h = float((t1 - t0) / np.timedelta64(1, "h"))

valid_counts = (~np.isnan(lons)).sum(axis=1)
print(f"{n_traj} trajectoires | {duration_h:.1f} h")
print(f"Observations valides par trajectoire : min={valid_counts.min()} max={valid_counts.max()}")
print(f"Trajectoires avec >= 2 points : {(valid_counts >= 2).sum()}")

# ========================================================== #
# Carte matplotlib pure                                       #
# ========================================================== #
fig, ax = plt.subplots(figsize=(12, 8))

pad_lon, pad_lat = 0.05, 0.03
lon_min = float(np.nanmin(lons)) - pad_lon
lon_max = float(np.nanmax(lons)) + pad_lon
lat_min = float(np.nanmin(lats)) - pad_lat
lat_max = float(np.nanmax(lats)) + pad_lat

ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_facecolor("#cce5f0")  # fond ocean bleu clair
ax.set_xlabel("Longitude (°E)", fontsize=11)
ax.set_ylabel("Latitude (°N)", fontsize=11)

# Grille
ax.grid(True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")

# Trajectoires
n_plotted = 0
for i in range(n_traj):
    lo = lons[i]
    la = lats[i]
    v  = np.isfinite(lo) & np.isfinite(la)
    if v.sum() < 2:
        continue
    ax.plot(lo[v], la[v], color="royalblue", alpha=0.5,
            linewidth=1.0)
    ax.plot(lo[v][0],  la[v][0],  "o", color="lime",
            markersize=5, zorder=5)
    ax.plot(lo[v][-1], la[v][-1], "o", color="red",
            markersize=5, zorder=5)
    n_plotted += 1

print(f"{n_plotted} trajectoires tracees")

legend_elements = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="lime",
           markersize=9, label="Depart"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="red",
           markersize=9, label="Arrivee"),
    Line2D([0],[0], color="royalblue", linewidth=2,
           alpha=0.7, label="Trajectoires"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=10,
          framealpha=0.85)

ax.set_title(
    f"Trajectoires lagrangiennes – S-111 ({duration_h:.0f}h)\n"
    f"Baie de Seine | {n_plotted}/{n_traj} particules | Parcels 3.1.4",
    fontsize=13, fontweight="bold"
)

# Annotations longitude / latitude
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}°"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}°"))
plt.xticks(rotation=30, fontsize=9)
plt.yticks(fontsize=9)

png_out = OUT / f"trajectoires_s111_{int(duration_h)}h.png"
plt.tight_layout()
plt.savefig(png_out, dpi=150, bbox_inches="tight")
print(f"Image sauvegardee : {png_out}")
plt.show()