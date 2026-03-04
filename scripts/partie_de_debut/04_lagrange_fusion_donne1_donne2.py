# -*- coding: utf-8 -*-
"""
Trajectoires lagrangiennes "propres" (HYCOM curviligne OK)
- 1 source courants obligatoire: donne1.nc (HYCOM u/v)
- 2e source optionnelle (seulement si elle contient u/v) => sinon ignorée
- Interpolation temporelle linéaire
- Spatial: IDW sur K voisins valides (KDTree) (lisse + évite NaN)
- Intégration: RK4
- Particules inactives si NaN / hors domaine (pas de téléportation)
- Sorties: PNG (lon/lat) + GeoJSON + CZML (points animés + path)
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import json
import datetime as dt

# ----------------- chemins -----------------
ROOT = Path(__file__).resolve().parents[1]        # .../scripts
DATA = ROOT.parent / "data"
OUT  = ROOT.parent / "output"
OUT.mkdir(exist_ok=True)

F1 = DATA / "donne1.nc"
F2 = DATA / "donne2.nc"   # optionnel

# ----------------- paramètres -----------------
DEPTH_INDEX = 0

DT_STEP = 900        # 15 min
HOURS   = 48
STEPS   = int((HOURS * 3600) // DT_STEP)

N_PART   = 120
RNG_SEED = 1

K_NEIGH = 8          # IDW: nb voisins
IDW_P   = 2.0        # poids = 1/d^p

# IMPORTANT : pour obtenir un rendu "comme le prof", démarre dans une zone dynamique.
USE_START_BOX = True
START_BOX = dict(lon_min=-6.5, lon_max=-1.0, lat_min=47.8, lat_max=50.2)  # Bretagne/Manche (à ajuster)

# ----------------- helpers -----------------
def find_var(ds, candidates):
    for c in candidates:
        if c in ds.variables:
            return c
    return None

def meters_to_deg(lon, lat, u, v):
    """m/s -> deg/s (approx locale)"""
    m2deg_lat = 1.0 / 111_000.0
    m2deg_lon = m2deg_lat / np.clip(np.cos(np.deg2rad(lat)), 1e-6, None)
    return u * m2deg_lon, v * m2deg_lat

def build_tree_valid(lon2d, lat2d, mask):
    iy, ix = np.where(mask)
    if len(iy) == 0:
        return None, (iy, ix)
    pts = np.c_[lon2d[iy, ix], lat2d[iy, ix]]
    tree = cKDTree(pts)
    return tree, (iy, ix)

def idw_sample(tree, idxs, U2d, V2d, lon, lat, k=8, p=2.0):
    """IDW sur k voisins (renvoie u,v; NaN si impossible)"""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    q = np.c_[lon, lat]

    u = np.full(lon.shape, np.nan, dtype=float)
    v = np.full(lat.shape, np.nan, dtype=float)

    if tree is None:
        return u, v

    good = np.isfinite(q).all(axis=1)
    if not np.any(good):
        return u, v

    d, j = tree.query(q[good], k=min(k, tree.n))
    if d.ndim == 1:
        d = d[:, None]
        j = j[:, None]

    w = 1.0 / np.power(np.clip(d, 1e-12, None), p)
    wsum = np.sum(w, axis=1)

    iy, ix = idxs
    uu = U2d[iy[j], ix[j]]
    vv = V2d[iy[j], ix[j]]

    # sécurité si voisin NaN (rare car mask, mais on protège)
    ok = np.isfinite(uu) & np.isfinite(vv)
    uu = np.where(ok, uu, 0.0)
    vv = np.where(ok, vv, 0.0)
    w  = np.where(ok, w, 0.0)
    wsum = np.sum(w, axis=1)

    u[good] = np.sum(w * uu, axis=1) / np.clip(wsum, 1e-12, None)
    v[good] = np.sum(w * vv, axis=1) / np.clip(wsum, 1e-12, None)
    return u, v

def export_geojson(trajs, path):
    feats = []
    for T in trajs:
        if len(T) < 2:
            continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "LineString",
                         "coordinates": [[float(lo), float(la)] for lo, la in T]},
            "properties": {}
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"type": "FeatureCollection", "features": feats}, f)
    return path

def export_czml_points_and_paths(trajs, t0_iso, step_seconds, path):
    t0 = dt.datetime.fromisoformat(t0_iso.replace("Z", ""))
    max_len = max((len(T) for T in trajs), default=0)
    if max_len < 2:
        raise ValueError("Pas assez de positions pour exporter CZML.")
    t_stop_iso = (t0 + dt.timedelta(seconds=(max_len - 1) * step_seconds)).isoformat() + "Z"

    packets = [{
        "id": "document", "version": "1.0",
        "clock": {
            "interval": f"{t0_iso}/{t_stop_iso}",
            "currentTime": t0_iso,
            "multiplier": 60,
            "range": "LOOP_STOP"
        }
    }]

    color = [0, 255, 255, 255]  # cyan

    for i, T in enumerate(trajs):
        if len(T) < 2:
            continue
        pos = []
        for k, (lon, lat) in enumerate(T):
            pos += [k * step_seconds, float(lon), float(lat), 0.0]

        packets.append({
            "id": f"p_{i}",
            "position": {"epoch": t0_iso, "cartographicDegrees": pos},
            "point": {"pixelSize": 6, "color": {"rgba": color}},
            "path": {
                "material": {"polylineOutline": {
                    "color": {"rgba": color},
                    "outlineColor": {"rgba": [0, 0, 0, 160]},
                    "outlineWidth": 1
                }},
                "width": 2,
                "leadTime": 0,
                "trailTime": step_seconds * (len(T) - 1),
                "resolution": 60
            }
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(packets, f)
    return path

# ----------------- lecture HYCOM (curviligne ok) -----------------
def load_currents(file, tag="S"):
    ds = xr.open_dataset(file)

    u_name = find_var(ds, ["u", "U", "water_u", "uo", "u_velocity", "eastward_velocity"])
    v_name = find_var(ds, ["v", "V", "water_v", "vo", "v_velocity", "northward_velocity"])
    lon_name = find_var(ds, ["lon", "longitude", "nav_lon", "LON", "xlon"])
    lat_name = find_var(ds, ["lat", "latitude",  "nav_lat", "LAT", "xlat"])
    time_name = find_var(ds, ["time", "Time", "ocean_time", "t"])

    if any(x is None for x in [u_name, v_name, lon_name, lat_name, time_name]):
        print(f"⚠️ [{tag}] u/v introuvables dans {file.name}.")
        print(f"   ➜ Variables disponibles (aperçu): {list(ds.variables)[:40]}")
        return None

    U = ds[u_name]
    V = ds[v_name]
    lon2d = ds[lon_name].values
    lat2d = ds[lat_name].values
    times = ds[time_name].values

    # profondeur si présente
    depth_dim = None
    for d in ["depth", "Depth", "deptht", "z", "lev"]:
        if d in U.dims:
            depth_dim = d
            break
    if depth_dim is not None:
        U = U.isel({depth_dim: DEPTH_INDEX})
        V = V.isel({depth_dim: DEPTH_INDEX})

    # normaliser dim temps si besoin
    if "time" not in U.dims and time_name in U.dims:
        # xarray garde le nom exact, donc ok. On accède avec isel({time_dim: k})
        pass

    # load
    U = U.load()
    V = V.load()

    # temps relatifs en secondes
    times_s = (times - times[0]) / np.timedelta64(1, "s")
    times_s = np.asarray(times_s, dtype=float)

    # domaine
    lon_min, lon_max = np.nanmin(lon2d), np.nanmax(lon2d)
    lat_min, lat_max = np.nanmin(lat2d), np.nanmax(lat2d)

    # dim temps réelle
    time_dim = None
    for d in U.dims:
        if d.lower() in ["time", "ocean_time", "t"]:
            time_dim = d
            break
    if time_dim is None:
        time_dim = U.dims[0]  # fallback raisonnable

    return dict(ds=ds, U=U, V=V, lon2d=lon2d, lat2d=lat2d,
                times=times, times_s=times_s, time_dim=time_dim,
                domain=(lon_min, lon_max, lat_min, lat_max))

S1 = load_currents(F1, tag="S1")
if S1 is None:
    raise ValueError("donne1.nc doit contenir u/v (courants). Vérifie les variables.")

S2 = load_currents(F2, tag="S2")
if S2 is None:
    print("ℹ️ Fallback donne2 désactivé (on continue avec donne1 uniquement).")

# domaine global
domains = [S1["domain"]] + ([S2["domain"]] if S2 else [])
DOMAIN = (
    min(d[0] for d in domains),
    max(d[1] for d in domains),
    min(d[2] for d in domains),
    max(d[3] for d in domains),
)

# ----------------- pré-calcul KDTree par time (rapide + stable) -----------------
def precompute_trees(S, tag="S"):
    trees = []
    idxs  = []
    Uall, Vall = S["U"], S["V"]
    td = S["time_dim"]

    nt = Uall.sizes[td]
    for k in range(nt):
        U2 = Uall.isel({td: k}).values
        V2 = Vall.isel({td: k}).values
        mask = np.isfinite(U2) & np.isfinite(V2)
        tree, idx = build_tree_valid(S["lon2d"], S["lat2d"], mask)
        trees.append(tree)
        idxs.append(idx)
    print(f"✅ [{tag}] KDTree pré-calculés: {nt} time steps")
    return trees, idxs

S1["trees"], S1["idxs"] = precompute_trees(S1, "S1")
if S2:
    S2["trees"], S2["idxs"] = precompute_trees(S2, "S2")

# ----------------- UV (time interp + IDW) -----------------
def uv_one_source(S, tsec, lonp, latp):
    times_s = S["times_s"]
    td = S["time_dim"]
    Uall, Vall = S["U"], S["V"]

    if len(times_s) < 2:
        return np.full_like(lonp, np.nan, float), np.full_like(latp, np.nan, float)

    if tsec <= times_s[0]:
        k0, a = 0, 0.0
    elif tsec >= times_s[-1]:
        k0, a = len(times_s) - 2, 1.0
    else:
        k0 = int(np.searchsorted(times_s, tsec) - 1)
        dtloc = times_s[k0 + 1] - times_s[k0]
        a = (tsec - times_s[k0]) / dtloc

    U0 = Uall.isel({td: k0}).values
    V0 = Vall.isel({td: k0}).values
    U1 = Uall.isel({td: k0 + 1}).values
    V1 = Vall.isel({td: k0 + 1}).values

    tree0, idx0 = S["trees"][k0], S["idxs"][k0]
    tree1, idx1 = S["trees"][k0 + 1], S["idxs"][k0 + 1]

    u0, v0 = idw_sample(tree0, idx0, U0, V0, lonp, latp, k=K_NEIGH, p=IDW_P)
    u1, v1 = idw_sample(tree1, idx1, U1, V1, lonp, latp, k=K_NEIGH, p=IDW_P)

    u = (1 - a) * u0 + a * u1
    v = (1 - a) * v0 + a * v1
    return u, v

def uv_fused(tsec, lonp, latp):
    u1, v1 = uv_one_source(S1, tsec, lonp, latp)
    if not S2:
        return u1, v1
    u2, v2 = uv_one_source(S2, tsec, lonp, latp)
    ok1 = np.isfinite(u1) & np.isfinite(v1)
    ok2 = np.isfinite(u2) & np.isfinite(v2)
    u = np.where(ok1, u1, np.where(ok2, u2, np.nan))
    v = np.where(ok1, v1, np.where(ok2, v2, np.nan))
    return u, v

# ----------------- init particules -----------------
def random_water_points(S, n, seed=1, box=None):
    td = S["time_dim"]
    U0 = S["U"].isel({td: 0}).values
    V0 = S["V"].isel({td: 0}).values
    lon2d, lat2d = S["lon2d"], S["lat2d"]

    mask = np.isfinite(U0) & np.isfinite(V0)

    if box is not None:
        mask = mask & (lon2d >= box["lon_min"]) & (lon2d <= box["lon_max"]) \
                    & (lat2d >= box["lat_min"]) & (lat2d <= box["lat_max"])

    yy, xx = np.where(mask)
    if len(yy) < n:
        raise ValueError("Pas assez de points valides pour init (box trop petite ?).")

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(yy), size=n, replace=False)
    return np.c_[lon2d[yy[sel], xx[sel]], lat2d[yy[sel], xx[sel]]]

box = START_BOX if USE_START_BOX else None
P = random_water_points(S1, N_PART, seed=RNG_SEED, box=box)

active = np.ones(N_PART, dtype=bool)
trajs = [[] for _ in range(N_PART)]
for i in range(N_PART):
    trajs[i].append(P[i].copy())

# ----------------- intégration RK4 -----------------
def f(t, Pxy):
    u, v = uv_fused(t, Pxy[:, 0], Pxy[:, 1])
    dlon, dlat = meters_to_deg(Pxy[:, 0], Pxy[:, 1], u, v)
    return np.c_[dlon, dlat]

tsec = 0.0
for step in range(STEPS):
    if not np.any(active):
        break

    P_act = P[active]

    k1 = f(tsec, P_act)
    k2 = f(tsec + 0.5 * DT_STEP, P_act + 0.5 * DT_STEP * k1)
    k3 = f(tsec + 0.5 * DT_STEP, P_act + 0.5 * DT_STEP * k2)
    k4 = f(tsec + 1.0 * DT_STEP, P_act + 1.0 * DT_STEP * k3)

    P_next = P_act + (DT_STEP / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    bad = ~np.isfinite(P_next).all(axis=1)
    out = (P_next[:, 0] < DOMAIN[0]) | (P_next[:, 0] > DOMAIN[1]) | (P_next[:, 1] < DOMAIN[2]) | (P_next[:, 1] > DOMAIN[3])

    P[active] = P_next

    act_idx = np.where(active)[0]
    active[act_idx[bad | out]] = False

    for j, gi in enumerate(act_idx):
        if active[gi]:
            trajs[gi].append(P[gi].copy())

    tsec += DT_STEP

# ----------------- PNG (lon/lat) -----------------
td = S1["time_dim"]
U0 = S1["U"].isel({td: 0}).values
V0 = S1["V"].isel({td: 0}).values
speed0 = np.sqrt(U0**2 + V0**2)

lon2d = S1["lon2d"]
lat2d = S1["lat2d"]

plt.figure(figsize=(12, 8))
plt.pcolormesh(lon2d, lat2d, speed0, shading="auto")
plt.colorbar(label="m/s")
plt.title(f"Trajectoires Lagrangiennes ({HOURS} h, IDW+RK4, fused={'oui' if S2 else 'non'})")
plt.xlabel("Longitude"); plt.ylabel("Latitude")

for T in trajs:
    if len(T) < 2:
        continue
    T = np.asarray(T)
    plt.plot(T[:, 0], T[:, 1], lw=1.0)
    # flèche direction (dernier segment)
    plt.annotate("", xy=(T[-1,0], T[-1,1]), xytext=(T[-2,0], T[-2,1]),
                 arrowprops=dict(arrowstyle="->", lw=1.0))

# zoom auto sur box si activée (rend le rendu "prof" beaucoup plus lisible)
if USE_START_BOX:
    plt.xlim(START_BOX["lon_min"], START_BOX["lon_max"])
    plt.ylim(START_BOX["lat_min"], START_BOX["lat_max"])

img_path = OUT / f"hycom_traj_{HOURS}h_clean.png"
plt.tight_layout()
plt.savefig(img_path, dpi=170)
print("✅ PNG :", img_path)

# ----------------- exports -----------------
gj_path = OUT / f"hycom_traj_{HOURS}h_clean.geojson"
cz_path = OUT / f"hycom_traj_{HOURS}h_clean.czml"

export_geojson(trajs, gj_path)

# t0 ISO : tu peux aussi mettre la vraie date du fichier si tu veux
t0_iso = "2025-11-17T20:00:00Z"
export_czml_points_and_paths(trajs, t0_iso, DT_STEP, cz_path)

print("✅ GeoJSON :", gj_path)
print("✅ CZML    :", cz_path)
print("🎯 Ouvre le CZML dans Cesium/NaVisu4D : points animés + path => direction claire.")
