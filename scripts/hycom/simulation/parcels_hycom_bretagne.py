# -*- coding: utf-8 -*-
"""
Parcels + HYCOM (donne1.nc)
Région : Bretagne (France)

Version avec particules sur grille structurée
- Grille curviligne HYCOM
- ScipyParticle (pas de compilation)
- RK4
- Sortie .zarr
"""

from pathlib import Path
import numpy as np
import xarray as xr
from datetime import timedelta
import shutil

from parcels import FieldSet, ParticleSet, ScipyParticle, AdvectionRK4

# ==========================================================
# Chemins
# ==========================================================
PROJECT = Path(__file__).resolve().parents[3]

DATA = PROJECT / "data" / "hycom"
OUT  = PROJECT / "output"
OUT.mkdir(exist_ok=True)

FILE = DATA / "donne1.nc"

# ==========================================================
# Paramètres
# ==========================================================
DEPTH_INDEX = 0

BBOX = dict(
    lon_min=-6.0,
    lon_max=-1.5,
    lat_min=47.5,
    lat_max=50.3
)

DT_MINUTES = 10
HOURS      = 23

# Taille de la grille (NX × NY = 150 particules)
GRID_NX = 15
GRID_NY = 10

# ==========================================================
# Utilitaires
# ==========================================================
def crop_bbox_curvilinear(ds, lon_name="lon", lat_name="lat", bbox=None):
    lon2d = ds[lon_name].values
    lat2d = ds[lat_name].values

    mask = (
        np.isfinite(lon2d) & np.isfinite(lat2d) &
        (lon2d >= bbox["lon_min"]) & (lon2d <= bbox["lon_max"]) &
        (lat2d >= bbox["lat_min"]) & (lat2d <= bbox["lat_max"])
    )

    iy, ix = np.where(mask)
    if len(iy) == 0:
        raise ValueError("BBox invalide : aucun point trouvé")

    return ds.isel(
        Y=slice(int(iy.min()), int(iy.max()) + 1),
        X=slice(int(ix.min()), int(ix.max()) + 1)
    )


def prepare_for_parcels(ds):
    ds = ds.rename({"Y": "y", "X": "x"})
    ds = ds.assign_coords({
        "x": (("y", "x"), ds["lon"].values),
        "y": (("y", "x"), ds["lat"].values),
    })
    return ds


def structured_grid_points(ds, nx=15, ny=10):
    """
    Remplace random_ocean_points_safe.
    Génère nx×ny points régulièrement espacés sur le domaine,
    avec une marge de 10% pour rester loin des bords.
    Seuls les points avec u/v valides (océan) sont conservés.
    """
    lon2d = ds["x"].values
    lat2d = ds["y"].values
    u0    = ds["u"].isel(time=0).values
    v0    = ds["v"].isel(time=0).values

    ny_grid, nx_grid = lon2d.shape
    margin_y = max(1, int(ny_grid * 0.10))
    margin_x = max(1, int(nx_grid * 0.10))

    lon_inner = lon2d[margin_y:-margin_y, margin_x:-margin_x]
    lat_inner = lat2d[margin_y:-margin_y, margin_x:-margin_x]

    lons = np.linspace(np.nanmin(lon_inner), np.nanmax(lon_inner), nx)
    lats = np.linspace(np.nanmin(lat_inner), np.nanmax(lat_inner), ny)

    grid_lon, grid_lat = np.meshgrid(lons, lats)
    grid_lon = grid_lon.flatten()
    grid_lat = grid_lat.flatten()

    valid_lon, valid_lat = [], []

    for lo, la in zip(grid_lon, grid_lat):
        dist   = (lon2d - lo)**2 + (lat2d - la)**2
        iy, ix = np.unravel_index(np.argmin(dist), dist.shape)
        if np.isfinite(u0[iy, ix]) and np.isfinite(v0[iy, ix]):
            valid_lon.append(lo)
            valid_lat.append(la)

    print(f"🌊 Particules sur grille structurée ({nx}×{ny}) : {len(valid_lon)} valides")
    return np.array(valid_lon), np.array(valid_lat)


# ==========================================================
# Lecture et préparation
# ==========================================================
print("📌 Dataset :", FILE)
print("📌 Existe ?", FILE.exists())

if not FILE.exists():
    raise FileNotFoundError(f"Fichier introuvable: {FILE}")

ds = xr.open_dataset(FILE)

if "depth" in ds["u"].dims:
    ds = ds.isel(depth=DEPTH_INDEX)

print("📌 dims u (avant crop) :", ds["u"].dims)

ds = crop_bbox_curvilinear(ds, bbox=BBOX)
ds = prepare_for_parcels(ds)

ds["u"] = ds["u"].load()
ds["v"] = ds["v"].load()

print("✅ dims u (après crop) :", ds["u"].dims)
print("✅ shape :", ds["u"].shape)
print(f"   Lon range: {float(np.nanmin(ds['x'].values)):.2f} → {float(np.nanmax(ds['x'].values)):.2f}")
print(f"   Lat range: {float(np.nanmin(ds['y'].values)):.2f} → {float(np.nanmax(ds['y'].values)):.2f}")

time_span = (ds.time[-1] - ds.time[0]).values / np.timedelta64(1, "h")
print(f"⏱️ Données disponibles : {float(time_span):.1f} heures")

# ==========================================================
# FieldSet
# ==========================================================
fieldset = FieldSet.from_xarray_dataset(
    ds,
    variables={"U": "u", "V": "v"},
    dimensions={"lon": "x", "lat": "y", "time": "time"},
    mesh="spherical",
    allow_time_extrapolation=True
)

# ==========================================================
# ParticleSet — grille structurée (au lieu de aléatoire)
# ==========================================================
lons0, lats0 = structured_grid_points(ds, nx=GRID_NX, ny=GRID_NY)

pset = ParticleSet.from_list(
    fieldset=fieldset,
    pclass=ScipyParticle,
    lon=lons0,
    lat=lats0
)

# ==========================================================
# Simulation — identique à la version originale qui marchait
# ==========================================================
out_zarr = OUT / f"parcels_hycom_bretagne_{HOURS}h.zarr"

if out_zarr.exists():
    shutil.rmtree(out_zarr)

pfile = pset.ParticleFile(
    name=str(out_zarr),
    outputdt=timedelta(minutes=DT_MINUTES)
)

print(f"🚀 Démarrage simulation ({HOURS}h)...")

pset.execute(
    AdvectionRK4,
    runtime=timedelta(hours=HOURS),
    dt=timedelta(minutes=DT_MINUTES),
    output_file=pfile,
    verbose_progress=True
)

print("✅ Simulation terminée !")
print("📁 Sortie :", out_zarr)