# -*- coding: utf-8 -*-
"""
Parcels + HYCOM (donne1.nc)
Région : Bretagne (France)
Version simple (celle qui lit donne1.nc et génère un .zarr)
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
# Chemins (adaptés à ton projet)
# ==========================================================
# __file__ = ...\scripts\partie4\parcels_hycom_bretagne.py
SCRIPTS_DIR = Path(__file__).resolve().parents[1]   # ...\scripts
PROJECT = SCRIPTS_DIR.parent                         # racine projet
DATA = PROJECT / "data"
OUT  = PROJECT / "output"
OUT.mkdir(exist_ok=True)

FILE = DATA / "donne1.nc"

# ==========================================================
# Paramètres
# ==========================================================
DEPTH_INDEX = 0

# Bretagne (zone réduite)
BBOX = dict(
    lon_min=-6.0,
    lon_max=-1.5,
    lat_min=47.5,
    lat_max=50.3
)

N_PART = 150
RNG_SEED = 15

DT_MINUTES = 10
HOURS = 23  # tes données font ~23h

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
    # Renommage dims
    ds = ds.rename({"Y": "y", "X": "x"})

    # Création coords x/y à partir de lon/lat (2D)
    ds = ds.assign_coords({
        "x": (("y", "x"), ds["lon"].values),
        "y": (("y", "x"), ds["lat"].values),
    })
    return ds

def random_ocean_points_safe(ds, n, seed=1):
    """
    Points océaniques au centre du domaine (marge 10%)
    """
    u0 = ds["u"].isel(time=0).values
    v0 = ds["v"].isel(time=0).values
    lon2d = ds["x"].values
    lat2d = ds["y"].values

    ny, nx = lon2d.shape
    margin_y = max(1, int(ny * 0.20))
    margin_x = max(1, int(nx * 0.20))

    lon_c = lon2d[margin_y:-margin_y, margin_x:-margin_x]
    lat_c = lat2d[margin_y:-margin_y, margin_x:-margin_x]
    u_c = u0[margin_y:-margin_y, margin_x:-margin_x]
    v_c = v0[margin_y:-margin_y, margin_x:-margin_x]

    mask = np.isfinite(u_c) & np.isfinite(v_c)
    iy, ix = np.where(mask)
    if len(iy) < n:
        print(f"⚠️  Seulement {len(iy)} points valides, demandé {n}. On réduit.")
        n = min(n, len(iy))

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(iy), size=n, replace=False)

    return lon_c[iy[sel], ix[sel]], lat_c[iy[sel], ix[sel]]

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
print(f"⏱️  Données disponibles: {float(time_span):.1f} heures")

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
# ParticleSet
# ==========================================================
print(f"🌊 Initialisation de {N_PART} particules...")
lons0, lats0 = random_ocean_points_safe(ds, N_PART, seed=RNG_SEED)

pset = ParticleSet.from_list(
    fieldset=fieldset,
    pclass=ScipyParticle,
    lon=lons0,
    lat=lats0
)

# ==========================================================
# Exécution
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
