# scripts/partie_2/03_lagrange_demo.py
# Suivi Lagrangien de particules (démo) sur donne1.nc
# - Intégrateur RK2 en coordonnées géographiques (lon/lat)
# - Échantillonnage (u,v) par plus-proche-voisin sur la grille curviligne (lon2D,lat2D)
# - Sauvegarde PNG + (optionnel) GeoJSON des trajectoires

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import json

# --------- chemins robustes -----------
ROOT = Path(__file__).resolve().parents[1]     # .../scripts
DATA = ROOT.parent / "data"
OUT  = ROOT.parent / "output"
OUT.mkdir(exist_ok=True)

FILE = DATA / "donne1.nc"
T_INDEX = 0
D_INDEX = 0

# pas de temps (secondes) et nb d'itérations
DT = 900          # 15 minutes
STEPS = 40        # 40 pas -> 10 heures
N_PART = 30       # nombre de particules

# ---------- utilitaires ----------
def meters_to_deg(lon, lat, u, v):
    """
    Convertit (u,v) m/s -> dlon/dt, dlat/dt en degrés/s.
    """
    m2deg_lat = 1.0 / 111_000.0
    # éviter cos(lat)=0
    m2deg_lon = m2deg_lat / np.clip(np.cos(np.deg2rad(lat)), 1e-6, None)
    return u * m2deg_lon, v * m2deg_lat

def build_sampler(lon2d, lat2d, U2d, V2d):
    """
    Construit un échantillonneur nearest-neighbor sur grille curviligne.
    Retourne une fonction sampler(lon, lat) -> (u, v) (m/s), shape=(N,).
    """
    pts = np.c_[lon2d.ravel(), lat2d.ravel()]
    tree = cKDTree(pts)

    def sampler(lon, lat):
        q = np.c_[lon, lat]
        dist, idx = tree.query(q, k=1)
        jj, ii = np.divmod(idx, lon2d.shape[1])
        u = U2d[jj, ii]
        v = V2d[jj, ii]
        return u, v
    return sampler

def export_geojson(trajs_lonlat, out_path: Path):
    feats=[]
    for T in trajs_lonlat:
        coords = [[float(lon), float(lat)] for lon,lat in T]
        feats.append({"type":"Feature",
                      "geometry":{"type":"LineString","coordinates":coords},
                      "properties":{}})
    fc={"type":"FeatureCollection","features":feats}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f)
    return out_path

# ---------- lecture des données ----------
ds = xr.open_dataset(FILE)

U = ds["u"].isel(time=T_INDEX, depth=D_INDEX).values   # (Y,X)
V = ds["v"].isel(time=T_INDEX, depth=D_INDEX).values   # (Y,X)
lon2d = ds["lon"].values                               # (Y,X)
lat2d = ds["lat"].values                               # (Y,X)

# masque mer: on garde points où u et v sont définis
mask = np.isfinite(U) & np.isfinite(V)

# sampler nearest-neighbor
sampler = build_sampler(lon2d, lat2d, U, V)

# ---------- init particules ----------
rng = np.random.default_rng(0)
yy, xx = np.where(mask)
sel = rng.choice(len(yy), size=N_PART, replace=False)
iy0, ix0 = yy[sel], xx[sel]
P = np.c_[lon2d[iy0, ix0], lat2d[iy0, ix0]]  # (N,2) lon/lat de départ

# pour tracer ensuite
trajectories = [ [p.copy()] for p in P ]

# ---------- intégration RK2 ----------
t0 = 0.0
for k in range(STEPS):
    # k1
    u1, v1 = sampler(P[:,0], P[:,1])
    dlon1, dlat1 = meters_to_deg(P[:,0], P[:,1], u1, v1)
    mid = P + 0.5 * DT * np.c_[dlon1, dlat1]

    # k2
    u2, v2 = sampler(mid[:,0], mid[:,1])
    dlon2, dlat2 = meters_to_deg(mid[:,0], mid[:,1], u2, v2)
    P = P + DT * np.c_[dlon2, dlat2]

    # on clippe légèrement pour éviter de sortir complètement de la grille
    P[:,0] = np.clip(P[:,0], lon2d.min(), lon2d.max())
    P[:,1] = np.clip(P[:,1], lat2d.min(), lat2d.max())

    for i in range(N_PART):
        trajectories[i].append(P[i].copy())

# ---------- figure ----------
speed = np.sqrt(U**2 + V**2)
plt.figure(figsize=(10,8))
plt.imshow(speed, origin="lower")
plt.colorbar(label="m/s")
plt.title("Trajectoires Lagrangiennes (HYCOM, ~10 h)")

# échantillonner flèches pour contexte (optionnel)
step_y, step_x = 24, 24
Y, X = np.mgrid[0:U.shape[0]:step_y, 0:U.shape[1]:step_x]
plt.quiver(X, Y, U[::step_y, ::step_x], V[::step_y, ::step_x],
           color="white", scale=50, width=0.002, alpha=0.6)

# tracer les trajectoires
for T in trajectories:
    # on re-projette en indices d'image via nearest-neighbor pour dessiner par-dessus l'imshow
    # (on utilise encore le KDTree)
    # NB: pour une carte vraie (lon/lat), on fera plus tard un fond géo.
    pts = np.c_[lon2d.ravel(), lat2d.ravel()]
    tree = cKDTree(pts)
    dist, idx = tree.query(np.array(T), k=1)
    jj, ii = np.divmod(idx, lon2d.shape[1])
    plt.plot(ii, jj, lw=1.2)

OUT_IMG = OUT / "hycom_traj_demo.png"
plt.tight_layout()
plt.savefig(OUT_IMG, dpi=150)
print(f"✅ Image enregistrée: {OUT_IMG}")

# ---------- export GeoJSON (optionnel, utile pour Cesium plus tard) ----------
OUT_GJ = OUT / "hycom_traj_demo.geojson"
export_geojson(trajectories, OUT_GJ)
print(f"✅ GeoJSON enregistré: {OUT_GJ}")
