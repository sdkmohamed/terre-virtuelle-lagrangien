# -*- coding: utf-8 -*-
"""
03b_parcels_s111_longer.py
--------------------------
Meme simulation mais sur 72h (3 jours) pour obtenir plus de trajectoires
qui restent dans le domaine plus longtemps.
Les donnees S-111 couvrent septembre 2021 (744h) donc on peut simuler plus long.
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
from parcels.tools.statuscodes import StatusCode

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "s111"
OUT  = ROOT / "output"
OUT.mkdir(exist_ok=True)

h5_files = list(DATA.glob("*.h5")) + list(DATA.glob("*.hdf5"))
H5_FILE  = h5_files[0]

# Parametres — plus de particules, simulation plus longue
N_PART     = 140   # max disponible (toute la grille)
RNG_SEED   = 42
DT_MINUTES = 5     # pas plus petit = moins de sorties OOB
HOURS      = 72    # 3 jours

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

data = load_s111(H5_FILE, bbox=None, verbose=False)
lon2d = data["lon"]; lat2d = data["lat"]
u_arr = data["u"];   v_arr = data["v"];  times = data["times"]

lons_1d = lon2d[0,:]; lats_1d = lat2d[:,0]
ds_xr = xr.Dataset(
    {"u": xr.DataArray(u_arr, dims=["time","lat","lon"]),
     "v": xr.DataArray(v_arr, dims=["time","lat","lon"])},
    coords={"time": times, "lat": lats_1d, "lon": lons_1d}
)

fieldset = FieldSet.from_xarray_dataset(
    ds_xr, variables={"U":"u","V":"v"},
    dimensions={"lon":"lon","lat":"lat","time":"time"},
    mesh="flat", allow_time_extrapolation=True,
)

# Tous les points valides
u0 = u_arr[0]
iy, ix = np.where(np.isfinite(u0))
n = min(N_PART, len(iy))
rng = np.random.default_rng(RNG_SEED)
sel = rng.choice(len(iy), size=n, replace=False)
lons0 = lon2d[iy[sel], ix[sel]]
lats0 = lat2d[iy[sel], ix[sel]]

pset = ParticleSet.from_list(fieldset=fieldset, pclass=ScipyParticle,
                              lon=lons0, lat=lats0)
print(f"Simulation {HOURS}h avec {len(pset)} particules (dt={DT_MINUTES} min)...")

out_zarr = OUT / f"parcels_s111_{HOURS}h.zarr"
if out_zarr.exists():
    shutil.rmtree(out_zarr)

pfile = pset.ParticleFile(name=str(out_zarr),
                          outputdt=timedelta(minutes=10))
pset.execute(
    pset.Kernel(RK4_safe),
    runtime=timedelta(hours=HOURS),
    dt=timedelta(minutes=DT_MINUTES),
    output_file=pfile,
    verbose_progress=True,
)
print(f"Sortie : {out_zarr}")