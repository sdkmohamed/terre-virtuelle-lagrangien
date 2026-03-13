# -*- coding: utf-8 -*-
"""
test2_deux_particules.py
------------------------
IDENTIQUE a 03_parcels_s111.py — seuls changements :
  - 2 particules a positions fixes (au lieu de grille 5x6)
  - sortie zarr renommee test2_...
  - figure matplotlib generee automatiquement
"""

from pathlib import Path
import numpy as np
import xarray as xr
from datetime import timedelta
import shutil
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

try:
    from read_s111 import load_s111
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from read_s111 import load_s111

from parcels import FieldSet, ParticleSet, ScipyParticle

# ========================================================== #
# Chemins                                                     #
# ========================================================== #
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "s111"
OUT  = ROOT / "output"
OUT.mkdir(exist_ok=True)

h5_files = list(DATA.glob("*.h5")) + list(DATA.glob("*.hdf5"))
if not h5_files:
    raise FileNotFoundError(f"Aucun fichier HDF5 trouve dans {DATA}")
H5_FILE = h5_files[0]

# ========================================================== #
# Parametres                                                  #
# ========================================================== #
DT_MINUTES = 10
HOURS      = 72

# ===== SEUL CHANGEMENT : 2 positions fixes =================
LONS0 = np.array([-2.12, -2.06])
LATS0 = np.array([48.71, 48.73])
# ===========================================================

# ========================================================== #
# Kernel RK4 + OOB (identique a 03_parcels_s111.py)          #
# ========================================================== #
def RK4_safe(particle, fieldset, time):
    lon_min = fieldset.U.grid.lon[0]
    lon_max = fieldset.U.grid.lon[-1]
    lat_min = fieldset.U.grid.lat[0]
    lat_max = fieldset.U.grid.lat[-1]
    dt2 = particle.dt * 0.5

    u1 = fieldset.U[time, particle.depth, particle.lat, particle.lon]
    v1 = fieldset.V[time, particle.depth, particle.lat, particle.lon]
    lon2 = particle.lon + u1 * dt2
    lat2 = particle.lat + v1 * dt2
    if lon2 < lon_min or lon2 > lon_max or lat2 < lat_min or lat2 > lat_max:
        particle.delete()
        return

    u2 = fieldset.U[time + dt2, particle.depth, lat2, lon2]
    v2 = fieldset.V[time + dt2, particle.depth, lat2, lon2]
    lon3 = particle.lon + u2 * dt2
    lat3 = particle.lat + v2 * dt2
    if lon3 < lon_min or lon3 > lon_max or lat3 < lat_min or lat3 > lat_max:
        particle.delete()
        return

    u3 = fieldset.U[time + dt2, particle.depth, lat3, lon3]
    v3 = fieldset.V[time + dt2, particle.depth, lat3, lon3]
    lon4 = particle.lon + u3 * particle.dt
    lat4 = particle.lat + v3 * particle.dt
    if lon4 < lon_min or lon4 > lon_max or lat4 < lat_min or lat4 > lat_max:
        particle.delete()
        return

    u4 = fieldset.U[time + particle.dt, particle.depth, lat4, lon4]
    v4 = fieldset.V[time + particle.dt, particle.depth, lat4, lon4]

    particle.lon += (u1 + 2.0*u2 + 2.0*u3 + u4) / 6.0 * particle.dt
    particle.lat += (v1 + 2.0*v2 + 2.0*v3 + v4) / 6.0 * particle.dt

    if (particle.lon < lon_min or particle.lon > lon_max or
            particle.lat < lat_min or particle.lat > lat_max):
        particle.delete()


# ========================================================== #
# 1. Lecture S-111                                            #
# ========================================================== #
print("=" * 60)
print("ETAPE 1 - Lecture du fichier S-111")
print("=" * 60)
data = load_s111(H5_FILE, bbox=None, verbose=True)
lon2d = data["lon"]; lat2d = data["lat"]
u_ms  = data["u"];   v_ms  = data["v"];  times = data["times"]
duration_hours = float((times[-1] - times[0]) / np.timedelta64(1, "h"))

# ========================================================== #
# 2. Conversion m/s -> deg/s                                  #
# ========================================================== #
lat_mean      = float(np.nanmean(lat2d))
cos_lat       = float(np.cos(np.radians(lat_mean)))
m_per_deg_lat = 111320.0
m_per_deg_lon = 111320.0 * cos_lat
u_degs = u_ms / m_per_deg_lon
v_degs = v_ms / m_per_deg_lat

# ========================================================== #
# 3. Dataset + FieldSet                                       #
# ========================================================== #
lons_1d = lon2d[0, :]; lats_1d = lat2d[:, 0]
ds_xr = xr.Dataset(
    {"u": xr.DataArray(u_degs, dims=["time","lat","lon"]),
     "v": xr.DataArray(v_degs, dims=["time","lat","lon"])},
    coords={"time": times, "lat": lats_1d, "lon": lons_1d}
)
fieldset = FieldSet.from_xarray_dataset(
    ds_xr,
    variables={"U": "u", "V": "v"},
    dimensions={"lon": "lon", "lat": "lat", "time": "time"},
    mesh="flat",
    allow_time_extrapolation=True,
)

# ========================================================== #
# 4. ParticleSet — 2 particules fixes                        #
# ========================================================== #
print("\n" + "=" * 60)
print("ETAPE 4 - 2 particules a positions fixes")
print("=" * 60)
print(f"  Particule 1 : lon={LONS0[0]}, lat={LATS0[0]}")
print(f"  Particule 2 : lon={LONS0[1]}, lat={LATS0[1]}")

pset = ParticleSet.from_list(
    fieldset=fieldset, pclass=ScipyParticle,
    lon=LONS0, lat=LATS0,
)

# ========================================================== #
# 5. Simulation                                               #
# ========================================================== #
print("\n" + "=" * 60)
print("ETAPE 5 - Simulation 72h")
print("=" * 60)
hours_to_run = min(duration_hours, float(HOURS))
out_zarr = OUT / "test2_s111_2particules.zarr"
if out_zarr.exists():
    shutil.rmtree(out_zarr)
pfile = pset.ParticleFile(name=str(out_zarr), outputdt=timedelta(minutes=10))
pset.execute(
    pset.Kernel(RK4_safe),
    runtime=timedelta(hours=hours_to_run),
    dt=timedelta(minutes=DT_MINUTES),
    output_file=pfile,
    verbose_progress=True,
)
print("Simulation terminee !")

# ========================================================== #
# 6. Lecture resultats + analyse inversions                   #
# ========================================================== #
print("\n" + "=" * 60)
print("ETAPE 6 - Analyse des inversions de direction")
print("=" * 60)

ds_out = xr.open_zarr(str(out_zarr))
lons_r = ds_out["lon"].values
lats_r = ds_out["lat"].values
t_r    = ds_out["time"].values

colors = ["royalblue", "tomato"]
labels = ["Particule 1", "Particule 2"]

for i in range(lons_r.shape[0]):
    lo   = lons_r[i]
    mask = np.isfinite(lo)
    dlon = np.gradient(lo[mask])
    inversions = int(np.sum(np.diff(np.sign(dlon)) != 0))
    print(f"  {labels[i]} : {inversions} inversions E-O sur 72h "
          f"(soit {inversions/3:.1f}/24h, attendu ~4 pour semi-diurne)")

# ========================================================== #
# FIGURE 1 — Trajectoires spatiales                          #
# ========================================================== #
fig1, ax = plt.subplots(figsize=(8, 6))
ax.set_facecolor("#e8f4f8")

for i in range(lons_r.shape[0]):
    lo = lons_r[i]; la = lats_r[i]
    mask = np.isfinite(lo)
    ax.plot(lo[mask], la[mask], color=colors[i], linewidth=1.5,
            label=labels[i], alpha=0.85)
    ax.plot(lo[mask][0],  la[mask][0],  "o", color=colors[i],
            markersize=9, zorder=5, label=f"Depart {i+1}")
    ax.plot(lo[mask][-1], la[mask][-1], "s", color=colors[i],
            markersize=9, zorder=5, label=f"Arrivee {i+1}")

ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.set_title("Test 2 — Trajectoires S-111 · 2 particules · 72h\n"
             "Cercles = départ · Carrés = arrivée")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.4)
fig1.tight_layout()
path1 = OUT / "test2_trajectoires.png"
fig1.savefig(str(path1), dpi=150)
print(f"\n📊 Figure 1 sauvegardee : {path1}")
plt.show()

# ========================================================== #
# FIGURE 2 — Position longitudinale + inversions             #
# ========================================================== #
fig2, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

for i in range(lons_r.shape[0]):
    lo   = lons_r[i]
    la   = lats_r[i]
    t_i  = t_r[i] if t_r.ndim == 2 else t_r
    mask = np.isfinite(lo)
    t_ok = pd.to_datetime(t_i[mask])
    dlon = np.gradient(lo[mask])

    axes[0].plot(t_ok, lo[mask], color=colors[i],
                 label=labels[i], linewidth=1.3)
    axes[1].plot(t_ok, dlon, color=colors[i],
                 label=labels[i], linewidth=1.0, alpha=0.8)

axes[0].set_ylabel("Longitude (°)")
axes[0].set_title("Position longitudinale au cours du temps")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
axes[1].set_ylabel("dLon/dt — proxy direction E-O")
axes[1].set_title("Inversions de direction (passage par zéro = changement de sens)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))
fig2.autofmt_xdate()
fig2.tight_layout()
path2 = OUT / "test2_inversions.png"
fig2.savefig(str(path2), dpi=150)
print(f"📊 Figure 2 sauvegardee : {path2}")
plt.show()