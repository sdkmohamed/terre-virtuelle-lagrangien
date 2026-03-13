# -*- coding: utf-8 -*-
"""
03_parcels_s111.py — TEST 2 : 2 particules fixes
"""

from pathlib import Path
import numpy as np
import xarray as xr
from datetime import timedelta
import shutil

try:
    from read_s111 import load_s111
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from read_s111 import load_s111

from parcels import FieldSet, ParticleSet, ScipyParticle

# ==========================================================
# Chemins
# ==========================================================
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "s111"
OUT  = ROOT / "output"
OUT.mkdir(exist_ok=True)

h5_files = list(DATA.glob("*.h5")) + list(DATA.glob("*.hdf5"))
if not h5_files:
    raise FileNotFoundError(f"Aucun fichier HDF5 trouvé dans {DATA}")
H5_FILE = h5_files[0]

# ==========================================================
# Paramètres
# ==========================================================
BBOX       = None
DT_MINUTES = 10
HOURS      = 72

# ==========================================================
# Kernel RK4
# ==========================================================
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

    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6 * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6 * particle.dt

# ==========================================================
# 1 Lecture S111
# ==========================================================
print("Lecture S111")

data = load_s111(H5_FILE, bbox=BBOX, verbose=True)

lon2d = data["lon"]
lat2d = data["lat"]
u_ms  = data["u"]
v_ms  = data["v"]
times = data["times"]

nt, ny, nx = u_ms.shape

duration_hours = float((times[-1] - times[0]) / np.timedelta64(1, "h"))

# ==========================================================
# Conversion m/s -> deg/s
# ==========================================================
lat_mean = float(np.nanmean(lat2d))

cos_lat = float(np.cos(np.radians(lat_mean)))

m_per_deg_lat = 111320
m_per_deg_lon = 111320 * cos_lat

u_degs = u_ms / m_per_deg_lon
v_degs = v_ms / m_per_deg_lat

# ==========================================================
# Dataset xarray
# ==========================================================
lons_1d = lon2d[0,:]
lats_1d = lat2d[:,0]

ds_xr = xr.Dataset(
    {
        "u": xr.DataArray(u_degs, dims=["time","lat","lon"]),
        "v": xr.DataArray(v_degs, dims=["time","lat","lon"]),
    },
    coords={
        "time":times,
        "lat":lats_1d,
        "lon":lons_1d
    }
)

# ==========================================================
# FieldSet
# ==========================================================
fieldset = FieldSet.from_xarray_dataset(
    ds_xr,
    variables={"U":"u","V":"v"},
    dimensions={"lon":"lon","lat":"lat","time":"time"},
    mesh="flat",
    allow_time_extrapolation=True
)

# ==========================================================
# 2 particules fixes
# ==========================================================
lons0 = np.array([-2.12, -2.06])
lats0 = np.array([48.71, 48.73])

pset = ParticleSet.from_list(
    fieldset=fieldset,
    pclass=ScipyParticle,
    lon=lons0,
    lat=lats0
)

# ==========================================================
# Simulation
# ==========================================================
out_zarr = OUT / "test2_s111_2particules.zarr"

if out_zarr.exists():
    shutil.rmtree(out_zarr)

pfile = pset.ParticleFile(
    name=str(out_zarr),
    outputdt=timedelta(minutes=10)
)

pset.execute(
    pset.Kernel(RK4_safe),
    runtime=timedelta(hours=duration_hours),
    dt=timedelta(minutes=DT_MINUTES),
    output_file=pfile,
    verbose_progress=True
)

print("Simulation terminée")

# ==========================================================
# Analyse
# ==========================================================
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

ds_out = xr.open_zarr(str(out_zarr))

lons_r = ds_out["lon"].values
lats_r = ds_out["lat"].values
t_vals = ds_out["time"].values

colors = ["royalblue","tomato"]
names = ["Particule 1","Particule 2"]

# ==============================
# FIGURE 1 Trajectoires
# ==============================
fig1, ax = plt.subplots(figsize=(7,6))

for i in range(lons_r.shape[0]):

    lo = lons_r[i]
    la = lats_r[i]

    ok = np.isfinite(lo)

    ax.plot(lo[ok], la[ok], color=colors[i], label=names[i])

    ax.plot(lo[ok][0], la[ok][0], "o", color=colors[i])
    ax.plot(lo[ok][-1], la[ok][-1], "s", color=colors[i])

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax.set_title("Trajectoires S111")

ax.legend()
ax.grid()

fig1.tight_layout()

fig1.savefig(OUT/"test2_trajectoires.png", dpi=150)

# ==============================
# FIGURE 2 Inversions
# ==============================
fig2, (ax1, ax2) = plt.subplots(2,1, figsize=(12,7), sharex=True)

for i in range(lons_r.shape[0]):

    lo = lons_r[i]

    ok = np.isfinite(lo)

    lo_ok = lo[ok]

    t_ok = pd.to_datetime(t_vals[i][ok] if t_vals.ndim==2 else t_vals[ok])

    dlon = np.gradient(lo_ok)

    ax1.plot(t_ok, lo_ok, color=colors[i], label=names[i])
    ax2.plot(t_ok, dlon, color=colors[i])

ax1.set_title("Longitude")
ax1.legend()
ax1.grid()

ax2.axhline(0, color="gray", linestyle="--")

ax2.set_title("Inversions direction")

ax2.grid()

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %Hh"))

fig2.tight_layout()

fig2.savefig(OUT/"test2_inversions.png", dpi=150)

# ==========================================================
# AFFICHAGE FINAL
# ==========================================================
plt.show()