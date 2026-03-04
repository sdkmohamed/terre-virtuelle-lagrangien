# scripts/partie_2/03b_lagrange_multitime.py
# Suivi Lagrangien multi-temps sur donne1.nc (HYCOM)
# - Interpolation temporelle linéaire entre frames t_k et t_{k+1}
# - Grille curviligne (lon/lat 2D) + échantillonnage nearest-neighbor SUR POINTS VALIDES
# - Tolérant aux NaN (mer/terre), positions non finies, et versions xarray anciennes
# - Sorties: PNG + GeoJSON + CZML

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import json, datetime as dt

# ---------- chemins ----------
ROOT = Path(__file__).resolve().parents[1]         # .../scripts
DATA = ROOT.parent / "data"                        # .../TerreVirtuelle/data
OUT  = ROOT.parent / "output"                      # .../TerreVirtuelle/output
OUT.mkdir(exist_ok=True)

FILE = DATA / "donne1.nc"

# ---------- paramètres ----------
D_INDEX = 0            # depth=0
DT_STEP = 900          # 15 min par pas (secondes)
STEPS   = 96           # 24 h (96 * 15 min)
N_PART  = 30           # nb particules
RNG_SEED = 1

# ---------- utilitaires ----------
def meters_to_deg(lon, lat, u, v):
    """Convertit (u,v) en m/s -> dlon/dt, dlat/dt en degrés/s."""
    m2deg_lat = 1.0 / 111_000.0
    m2deg_lon = m2deg_lat / np.clip(np.cos(np.deg2rad(lat)), 1e-6, None)
    return u * m2deg_lon, v * m2deg_lat

def build_tree_valid(lon2d, lat2d, mask):
    """
    Construit un KDTree uniquement sur les points où (u,v) sont valides (non-NaN).
    Retourne (tree, (iy_valid, ix_valid)).
    """
    iy, ix = np.where(mask)
    pts = np.c_[lon2d[iy, ix], lat2d[iy, ix]]
    tree = cKDTree(pts) if len(pts) > 0 else None
    return tree, (iy, ix)

def sample_from_tree_valid(tree, idxs, U2d, V2d, lon, lat, fallback=(0.0, 0.0)):
    """
    Échantillonne (u,v) au plus proche point VALIDE.
    Protège contre les lon/lat non finies.
    """
    lon = np.asarray(lon).astype(float)
    lat = np.asarray(lat).astype(float)
    q = np.c_[lon, lat]

    # Si tout est NaN, renvoie le fallback
    if not np.isfinite(q).any():
        return np.full_like(lon, fallback[0], dtype=float), np.full_like(lat, fallback[1], dtype=float)

    # Remplace les lignes NaN par la première ligne finie (évite crash KDTree)
    good_rows = np.isfinite(q).all(axis=1)
    if not good_rows.all():
        first_good = np.where(good_rows)[0]
        if len(first_good) == 0:
            return np.full_like(lon, fallback[0], dtype=float), np.full_like(lat, fallback[1], dtype=float)
        q[~good_rows] = q[first_good[0]]

    if tree is None:
        return np.full_like(lon, fallback[0], dtype=float), np.full_like(lat, fallback[1], dtype=float)

    _, j = tree.query(q, k=1)
    iy, ix = idxs
    u = U2d[iy[j], ix[j]]
    v = V2d[iy[j], ix[j]]
    return u, v

def export_geojson(trajs, path):
    feats=[]
    for T in trajs:
        feats.append({
            "type":"Feature",
            "geometry":{"type":"LineString",
                        "coordinates":[[float(lon), float(lat)] for lon,lat in T]},
            "properties":{}
        })
    with open(path,"w",encoding="utf-8") as f:
        json.dump({"type":"FeatureCollection","features":feats}, f)
    return path

def export_czml(trajs, t0_iso, step_seconds, path):
    import datetime as dt, json
    t0 = dt.datetime.fromisoformat(t0_iso.replace("Z",""))

    packets=[{
        "id":"document","version":"1.0",
        "clock":{
            "interval":"", "currentTime":t0_iso,
            "multiplier":60, "range":"LOOP_STOP"
        }
    }]

    for i,T in enumerate(trajs):
        # cartographicDegrees time-tagged: [seconds, lon, lat, h, seconds, lon, lat, h, ...]
        flat=[]
        for k,(lon,lat) in enumerate(T):
            sec = k * step_seconds
            flat += [sec, float(lon), float(lat), 0.0]

        packets.append({
            "id": f"traj_{i}",
            "polyline": {
                "positions": {
                    "epoch": t0_iso,
                    "cartographicDegrees": flat
                },
                "width": 2,
                "material": {
                    "polylineOutline": {
                        "color": {"rgba": [0,255,255,255]},
                        "outlineColor": {"rgba": [0,0,0,160]},
                        "outlineWidth": 1
                    }
                }
            }
        })

    t_stop_iso = (t0 + dt.timedelta(seconds=(len(trajs[0])-1)*step_seconds)).isoformat()+"Z"
    packets[0]["clock"]["interval"] = f"{t0_iso}/{t_stop_iso}"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(packets, f)
    return path

# ---------- lecture ----------
ds = xr.open_dataset(FILE)

# (time, Y, X)
Uall = ds["u"].isel(depth=D_INDEX).load()   # DataArray
Vall = ds["v"].isel(depth=D_INDEX).load()
lon2d = ds["lon"].values                    # (Y,X)
lat2d = ds["lat"].values                    # (Y,X)

# temps (numpy datetime64 -> secondes relatives)
times = ds["time"].values                   # compat toutes versions
if times.size < 2:
    raise ValueError("Il faut au moins 2 pas de temps pour l'interpolation temporelle.")
times_s = (times - times[0]) / np.timedelta64(1,"s")

# ---------- init particules (points d'eau valides à t0) ----------
mask0 = np.isfinite(Uall.isel(time=0).values) & np.isfinite(Vall.isel(time=0).values)
yy, xx = np.where(mask0)
rng = np.random.default_rng(RNG_SEED)
sel = rng.choice(len(yy), size=N_PART, replace=False)
P = np.c_[lon2d[yy[sel], xx[sel]], lat2d[yy[sel], xx[sel]]]   # (N,2)

trajs = [[p.copy()] for p in P]
domain = (lon2d.min(), lon2d.max(), lat2d.min(), lat2d.max())

def uv_at_time(tsec, lon, lat):
    """
    Interpolation temporelle linéaire entre frames k et k+1.
    Spatial: nearest-neighbor sur points VALIDES de chaque frame.
    """
    # indices temporels et facteur d'interp
    if tsec <= times_s[0]:
        k0 = 0; a = 0.0
    elif tsec >= times_s[-1]:
        k0 = len(times_s) - 2; a = 1.0
    else:
        k0 = int(np.searchsorted(times_s, tsec) - 1)
        dt = times_s[k0+1] - times_s[k0]
        a  = (tsec - times_s[k0]) / dt

    U0 = Uall.isel(time=k0).values
    V0 = Vall.isel(time=k0).values
    U1 = Uall.isel(time=k0+1).values
    V1 = Vall.isel(time=k0+1).values

    mask0 = np.isfinite(U0) & np.isfinite(V0)
    mask1 = np.isfinite(U1) & np.isfinite(V1)
    tree0, idx0 = build_tree_valid(lon2d, lat2d, mask0)
    tree1, idx1 = build_tree_valid(lon2d, lat2d, mask1)

    u0, v0 = sample_from_tree_valid(tree0, idx0, U0, V0, lon, lat, fallback=(0.0, 0.0))
    u1, v1 = sample_from_tree_valid(tree1, idx1, U1, V1, lon, lat, fallback=(0.0, 0.0))

    u = (1.0 - a) * u0 + a * u1
    v = (1.0 - a) * v0 + a * v1
    return u, v

# ---------- intégration RK2 sur 24h ----------
tsec = 0.0
for k in range(STEPS):
    # positions sûres (évite NaN dans le KDTree)
    P = np.where(np.isfinite(P), P, trajs[0][0])  # remet les NaN à une position valable (départ de la première)

    # k1
    u1, v1 = uv_at_time(tsec, P[:,0], P[:,1])
    dlon1, dlat1 = meters_to_deg(P[:,0], P[:,1], u1, v1)
    Pmid = P + 0.5 * DT_STEP * np.c_[dlon1, dlat1]

    # sécurise Pmid
    Pmid = np.where(np.isfinite(Pmid), Pmid, P)

    # k2
    u2, v2 = uv_at_time(tsec + 0.5*DT_STEP, Pmid[:,0], Pmid[:,1])
    dlon2, dlat2 = meters_to_deg(Pmid[:,0], Pmid[:,1], u2, v2)
    P = P + DT_STEP * np.c_[dlon2, dlat2]

    # clip emprise
    P[:,0] = np.clip(P[:,0], domain[0], domain[1])
    P[:,1] = np.clip(P[:,1], domain[2], domain[3])

    for i in range(N_PART):
        trajs[i].append(P[i].copy())

    tsec += DT_STEP

# ---------- figure de contrôle ----------
speed0 = np.sqrt(Uall.isel(time=0).values**2 + Vall.isel(time=0).values**2)
plt.figure(figsize=(10,8))
plt.imshow(speed0, origin="lower")
plt.colorbar(label="m/s")
plt.title("Trajectoires Lagrangiennes (HYCOM, 24 h, interp. temporelle, robustes)")

# Pour dessiner les lignes sur l'imshow (index-image via NN)
pts = np.c_[lon2d.ravel(), lat2d.ravel()]
tree_img = cKDTree(pts)
shp = lon2d.shape
for T in trajs:
    _, idx = tree_img.query(np.array(T), k=1)
    jj, ii = np.divmod(idx, shp[1])
    plt.plot(ii, jj, lw=1.2)

img_path = OUT / "hycom_traj_24h.png"
plt.tight_layout(); plt.savefig(img_path, dpi=150)
print("✅ Image:", img_path)

# ---------- exports ----------
gj_path = OUT / "hycom_traj_24h.geojson"
cz_path = OUT / "hycom_traj_24h.czml"
export_geojson(trajs, gj_path)
t0_iso = "2025-01-01T12:00:00Z"
export_czml(trajs, t0_iso, DT_STEP, cz_path)
print("✅ GeoJSON:", gj_path)
print("✅ CZML:", cz_path)
