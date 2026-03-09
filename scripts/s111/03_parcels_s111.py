# -*- coding: utf-8 -*-
"""
03_parcels_s111.py
------------------
Simulation lagrangienne S-111 (HDF5) -> Parcels 3.1.4 -> .zarr

CORRECTION CRITIQUE : les vitesses S-111 sont en m/s.
Avec mesh="flat", Parcels attend des deg/s.
On convertit donc u,v : m/s -> deg/s avant de creer le FieldSet.
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
BBOX       = None
N_PART     = 140
RNG_SEED   = 42
DT_MINUTES = 10
HOURS      = 72

# ========================================================== #
# Kernel RK4 en syntaxe Parcels 3.x                          #
# Les vitesses etant en deg/s apres conversion,              #
# le kernel est une addition directe (pas de facteur cos).   #
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

data = load_s111(H5_FILE, bbox=BBOX, verbose=True)

lon2d  = data["lon"]
lat2d  = data["lat"]
u_ms   = data["u"]    # m/s
v_ms   = data["v"]    # m/s
times  = data["times"]

nt, ny, nx = u_ms.shape
duration_hours = float((times[-1] - times[0]) / np.timedelta64(1, "h"))
print(f"\nDuree disponible : {duration_hours:.1f} h")

# ========================================================== #
# 2. Conversion m/s -> deg/s                                  #
# ========================================================== #
print("\n" + "=" * 60)
print("ETAPE 2 - Conversion vitesses m/s -> deg/s")
print("=" * 60)

# Latitude moyenne du domaine
lat_mean = float(np.nanmean(lat2d))
m_per_deg_lat = 111320.0
m_per_deg_lon = 111320.0 * np.cos(np.radians(lat_mean))

print(f"Latitude moyenne : {lat_mean:.3f} deg")
print(f"1 deg lat = {m_per_deg_lat:.0f} m")
print(f"1 deg lon = {m_per_deg_lon:.0f} m")

# Conversion : divise par les metres par degre
u_degs = u_ms / m_per_deg_lon   # deg/s (est-ouest)
v_degs = v_ms / m_per_deg_lat   # deg/s (nord-sud)

print(f"Plage U apres conversion : [{np.nanmin(u_degs):.2e}, {np.nanmax(u_degs):.2e}] deg/s")
print(f"Plage V apres conversion : [{np.nanmin(v_degs):.2e}, {np.nanmax(v_degs):.2e}] deg/s")
print(f"Deplacement max (dt={DT_MINUTES}min) : {np.nanmax(np.abs(u_degs))*DT_MINUTES*60:.5f} deg")

# ========================================================== #
# 3. xarray Dataset avec vitesses converties                  #
# ========================================================== #
print("\n" + "=" * 60)
print("ETAPE 3 - Dataset xarray")
print("=" * 60)

lons_1d = lon2d[0, :]
lats_1d = lat2d[:, 0]

ds_xr = xr.Dataset(
    {
        "u": xr.DataArray(u_degs, dims=["time", "lat", "lon"],
                          attrs={"units": "deg s-1"}),
        "v": xr.DataArray(v_degs, dims=["time", "lat", "lon"],
                          attrs={"units": "deg s-1"}),
    },
    coords={"time": times, "lat": lats_1d, "lon": lons_1d}
)
print(ds_xr)

# ========================================================== #
# 4. FieldSet (mesh=flat car vitesses en deg/s)               #
# ========================================================== #
print("\n" + "=" * 60)
print("ETAPE 4 - FieldSet")
print("=" * 60)

fieldset = FieldSet.from_xarray_dataset(
    ds_xr,
    variables={"U": "u", "V": "v"},
    dimensions={"lon": "lon", "lat": "lat", "time": "time"},
    mesh="flat",
    allow_time_extrapolation=True,
)
print("FieldSet cree")

# ========================================================== #
# 5. Particules                                               #
# ========================================================== #
print("\n" + "=" * 60)
print("ETAPE 5 - Initialisation des particules")
print("=" * 60)

u0 = u_ms[0]
iy, ix = np.where(np.isfinite(u0))
n = min(N_PART, len(iy))
rng = np.random.default_rng(RNG_SEED)
sel = rng.choice(len(iy), size=n, replace=False)
lons0 = lon2d[iy[sel], ix[sel]]
lats0 = lat2d[iy[sel], ix[sel]]

pset = ParticleSet.from_list(
    fieldset=fieldset,
    pclass=ScipyParticle,
    lon=lons0, lat=lats0,
)
print(f"ParticleSet : {len(pset)} particules")

# ========================================================== #
# 6. Simulation                                               #
# ========================================================== #
print("\n" + "=" * 60)
print("ETAPE 6 - Simulation RK4")
print("=" * 60)

hours_to_run = min(duration_hours, float(HOURS))
out_zarr = OUT / f"parcels_s111_{int(hours_to_run)}h.zarr"

if out_zarr.exists():
    shutil.rmtree(out_zarr)

pfile = pset.ParticleFile(
    name=str(out_zarr),
    outputdt=timedelta(minutes=10),
)

print(f"Simulation de {hours_to_run:.0f}h avec {len(pset)} particules...")

pset.execute(
    pset.Kernel(RK4_safe),
    runtime=timedelta(hours=hours_to_run),
    dt=timedelta(minutes=DT_MINUTES),
    output_file=pfile,
    verbose_progress=True,
)

print(f"\nSimulation terminee !")
print(f"Sortie : {out_zarr}")