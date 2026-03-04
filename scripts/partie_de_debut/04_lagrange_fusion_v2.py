# -*- coding: utf-8 -*-
"""
Lagrangien PRO (HYCOM curviligne 2D)
- Grille curviligne lon2d/lat2d -> KDTree
- IDW k-neighbors (évite NaN, lisse)
- Interpolation temporelle linéaire
- Intégration RK4
- Stop particules si: NaN / hors domaine / sur terre (u,v non-finies)
- Init dans une box + filtre vitesse min (pour éviter "spaghetti")
- Plot lon/lat (compatible comparaison)
- Exports: PNG + GeoJSON + CZML (points animés + path)
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import json
import datetime as dt

# =========================
# Chemins
# =========================
ROOT = Path(__file__).resolve().parents[1]  # .../scripts
DATA = ROOT.parent / "data"
OUT  = ROOT.parent / "output"
OUT.mkdir(exist_ok=True)

F1 = DATA / "donne1.nc"
F2 = DATA / "donne2.nc"  # optionnel, ignoré s'il n'a pas u/v

# =========================
# Paramètres (à ajuster)
# =========================
DEPTH_INDEX = 0

DT_STEP = 900          # 15 minutes
HOURS   = 48
STEPS   = int((HOURS * 3600) // DT_STEP)

N_PART   = 80          # baisse à 40-60 si trop chargé
RNG_SEED = 2

# IDW
K_NEIGH = 8
IDW_P   = 2.0

# Zone "comme le prof" -> mets ici la box exacte
USE_START_BOX = True
START_BOX = dict(
    lon_min=-6.5, lon_max=-1.0,
    lat_min=47.8, lat_max=50.2
)

# Filtrage init pour éviter bruit
MIN_SPEED_INIT = 0.20   # m/s (0.10 à 0.30 typique)
MIN_TRAJ_LEN   = 20     # points minimum pour afficher/export (évite flèches manquantes)
ARROW_EVERY    = 1      # 1=flèche sur dernier segment ; 2=flèche moins fréquente si tu ajoutes plusieurs flèches

# =========================
# Helpers
# =========================
def find_var(ds, candidates):
    for c in candidates:
        if c in ds.variables:
            return c
    return None

def _pick_dim(da, candidates):
    for d in candidates:
        if d in da.dims:
            return d
    return None

def meters_to_deg(lon, lat, u, v):
    """m/s -> deg/s (approx locale)"""
    m2deg_lat = 1.0 / 111_000.0
    m2deg_lon = m2deg_lat / np.clip(np.cos(np.deg2rad(lat)), 1e-6, None)
    return u * m2deg_lon, v * m2deg_lat

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
    t0 = dt.datetime.fromisoformat(t0_iso.replace("Z",""))
    max_len = max((len(T) for T in trajs), default=0)
    if max_len < 2:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([{"id":"document","version":"1.0"}], f)
        return path

    t_stop_iso = (t0 + dt.timedelta(seconds=(max_len-1)*step_seconds)).isoformat() + "Z"
    packets = [{
        "id": "document", "version": "1.0",
        "clock": {
            "interval": f"{t0_iso}/{t_stop_iso}",
            "currentTime": t0_iso,
            "multiplier": 60,
            "range": "LOOP_STOP"
        }
    }]

    for i, T in enumerate(trajs):
        if len(T) < 2:
            continue

        pos = []
        for k, (lon, lat) in enumerate(T):
            pos += [k*step_seconds, float(lon), float(lat), 0.0]

        color = [0, 255, 255, 255]  # cyan
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
                "trailTime": step_seconds * (len(T)-1),
                "resolution": 60
            }
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(packets, f)
    return path

# =========================
# Lecture HYCOM (curviligne ou rect)
# =========================
def load_current_dataset(file: Path, tag="S"):
    ds = xr.open_dataset(file)

    u_name   = find_var(ds, ["u", "U", "water_u", "uo", "u_eastward", "eastward_sea_water_velocity"])
    v_name   = find_var(ds, ["v", "V", "water_v", "vo", "v_northward", "northward_sea_water_velocity"])
    lon_name = find_var(ds, ["lon", "longitude", "nav_lon", "LON", "x"])
    lat_name = find_var(ds, ["lat", "latitude", "nav_lat", "LAT", "y"])
    time_name= find_var(ds, ["time", "Time", "ocean_time", "t"])
    dep_name = find_var(ds, ["depth", "Depth", "deptht", "z", "lev"])

    if u_name is None or v_name is None:
        print(f"⚠️ [{tag}] u/v introuvables dans {file.name}.")
        print(f"   ➜ Variables disponibles (aperçu): {list(ds.variables)[:30]}")
        return None

    if lon_name is None or lat_name is None or time_name is None:
        raise ValueError(f"[{tag}] Variables manquantes (lon/lat/time) dans {file.name}")

    U = ds[u_name]
    V = ds[v_name]

    # dimensions
    tdim = _pick_dim(U, ["time", time_name, "Time", "ocean_time"])
    if tdim is None:
        raise ValueError(f"[{tag}] Dimension time introuvable dans {file.name} (dims U={U.dims})")

    # depth optionnel
    ddim = _pick_dim(U, ["depth", dep_name, "z", "lev"])
    if ddim is not None:
        U = U.isel({ddim: DEPTH_INDEX})
        V = V.isel({ddim: DEPTH_INDEX})

    # charge lon/lat
    lon = ds[lon_name].values
    lat = ds[lat_name].values

    # lon/lat peuvent être 1D ou 2D
    if lon.ndim == 1 and lat.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    elif lon.ndim == 2 and lat.ndim == 2:
        lon2d, lat2d = lon, lat
    else:
        raise ValueError(f"[{tag}] lon/lat dimensions inattendues: lon{lon.shape}, lat{lat.shape}")

    # temps en secondes relatives
    times = ds[time_name].values
    times_s = (times - times[0]) / np.timedelta64(1, "s")
    times_s = np.asarray(times_s, dtype=float)

    # met en mémoire (stabilité + vitesse)
    U = U.load()
    V = V.load()

    # domaine
    dom = (np.nanmin(lon2d), np.nanmax(lon2d), np.nanmin(lat2d), np.nanmax(lat2d))

    return dict(
        ds=ds, U=U, V=V,
        lon2d=lon2d, lat2d=lat2d,
        time_name=time_name, times=times, times_s=times_s,
        domain=dom,
        tdim=tdim
    )

S1 = load_current_dataset(F1, tag="S1")
if S1 is None:
    raise ValueError("donne1.nc doit contenir u/v. Arrêt.")

S2 = load_current_dataset(F2, tag="S2")
if S2 is None:
    print("ℹ️ Fallback donne2 désactivé (on continue avec donne1 uniquement).")

# domaine global (union)
d1 = S1["domain"]
if S2 is not None:
    d2 = S2["domain"]
    DOMAIN = (min(d1[0], d2[0]), max(d1[1], d2[1]), min(d1[2], d2[2]), max(d1[3], d2[3]))
else:
    DOMAIN = d1

# =========================
# KDTree global de la grille (accélère)
# =========================
def build_grid_tree(lon2d, lat2d):
    pts = np.c_[lon2d.ravel(), lat2d.ravel()]
    return cKDTree(pts), lon2d.shape

TREE1, SHP1 = build_grid_tree(S1["lon2d"], S1["lat2d"])
if S2 is not None:
    TREE2, SHP2 = build_grid_tree(S2["lon2d"], S2["lat2d"])

def idw_from_flat(tree, shp, U2d, V2d, lonp, latp, k=8, p=2.0):
    """
    IDW sur grille curviligne via KDTree sur (lon,lat) ravel.
    Retourne u,v (1D arrays).
    """
    lonp = np.asarray(lonp, dtype=float)
    latp = np.asarray(latp, dtype=float)
    q = np.c_[lonp, latp]

    u = np.full(lonp.shape, np.nan, dtype=float)
    v = np.full(latp.shape, np.nan, dtype=float)

    good = np.isfinite(q).all(axis=1)
    if not np.any(good):
        return u, v

    d, idx = tree.query(q[good], k=min(k, tree.n))
    if d.ndim == 1:
        d = d[:, None]
        idx = idx[:, None]

    # idx -> (j,i)
    jj, ii = np.divmod(idx, shp[1])
    uu = U2d[jj, ii]
    vv = V2d[jj, ii]

    # masque voisinage (terre -> NaN)
    ok = np.isfinite(uu) & np.isfinite(vv)
    # poids
    w = 1.0 / np.power(np.clip(d, 1e-12, None), p)
    w = np.where(ok, w, 0.0)

    wsum = np.sum(w, axis=1)
    # évite division 0
    good2 = wsum > 0

    out_u = np.full(wsum.shape, np.nan, dtype=float)
    out_v = np.full(wsum.shape, np.nan, dtype=float)
    out_u[good2] = np.sum(w[good2] * uu[good2], axis=1) / wsum[good2]
    out_v[good2] = np.sum(w[good2] * vv[good2], axis=1) / wsum[good2]

    u[good] = out_u
    v[good] = out_v
    return u, v

def uv_one_source(S, tree, shp, tsec, lonp, latp):
    times_s = S["times_s"]
    Uall = S["U"]
    Vall = S["V"]

    if len(times_s) < 2:
        return np.full_like(lonp, np.nan, dtype=float), np.full_like(latp, np.nan, dtype=float)

    if tsec <= times_s[0]:
        k0, a = 0, 0.0
    elif tsec >= times_s[-1]:
        k0, a = len(times_s)-2, 1.0
    else:
        k0 = int(np.searchsorted(times_s, tsec) - 1)
        dtloc = times_s[k0+1] - times_s[k0]
        a = (tsec - times_s[k0]) / dtloc

    U0 = Uall.isel(time=k0).values
    V0 = Vall.isel(time=k0).values
    U1 = Uall.isel(time=k0+1).values
    V1 = Vall.isel(time=k0+1).values

    u0, v0 = idw_from_flat(tree, shp, U0, V0, lonp, latp, k=K_NEIGH, p=IDW_P)
    u1, v1 = idw_from_flat(tree, shp, U1, V1, lonp, latp, k=K_NEIGH, p=IDW_P)

    u = (1-a)*u0 + a*u1
    v = (1-a)*v0 + a*v1
    return u, v

def uv_fused(tsec, lonp, latp):
    # priorité S1
    u1, v1 = uv_one_source(S1, TREE1, SHP1, tsec, lonp, latp)
    ok1 = np.isfinite(u1) & np.isfinite(v1)
    if S2 is None:
        return u1, v1

    u2, v2 = uv_one_source(S2, TREE2, SHP2, tsec, lonp, latp)
    ok2 = np.isfinite(u2) & np.isfinite(v2)

    u = np.where(ok1, u1, np.where(ok2, u2, np.nan))
    v = np.where(ok1, v1, np.where(ok2, v2, np.nan))
    return u, v

# =========================
# Initialisation particules (dans eau + vitesse min)
# =========================
def init_particles_in_box(S, n, seed=1, box=None, min_speed=0.0):
    U0 = S["U"].isel(time=0).values
    V0 = S["V"].isel(time=0).values
    lon2d = S["lon2d"]
    lat2d = S["lat2d"]

    speed0 = np.sqrt(U0**2 + V0**2)
    mask = np.isfinite(U0) & np.isfinite(V0) & np.isfinite(lon2d) & np.isfinite(lat2d)
    if min_speed > 0:
        mask &= (speed0 >= min_speed)

    if box is not None:
        mask &= (lon2d >= box["lon_min"]) & (lon2d <= box["lon_max"]) \
             & (lat2d >= box["lat_min"]) & (lat2d <= box["lat_max"])

    yy, xx = np.where(mask)
    if len(yy) < n:
        raise ValueError(f"Pas assez de points valides dans la box (dispo={len(yy)}, demandé={n}). "
                         f"➜ Diminue N_PART ou MIN_SPEED_INIT ou agrandis la box.")

    rng = np.random.default_rng(seed)
    sel = rng.choice(len(yy), size=n, replace=False)
    P0 = np.c_[lon2d[yy[sel], xx[sel]], lat2d[yy[sel], xx[sel]]]
    return P0

box = START_BOX if USE_START_BOX else None
P = init_particles_in_box(S1, N_PART, seed=RNG_SEED, box=box, min_speed=MIN_SPEED_INIT)

active = np.ones(N_PART, dtype=bool)
trajs = [[] for _ in range(N_PART)]
for i in range(N_PART):
    trajs[i].append(P[i].copy())

# =========================
# Dynamique RK4
# =========================
def f(t, Pxy):
    u, v = uv_fused(t, Pxy[:,0], Pxy[:,1])
    dlon, dlat = meters_to_deg(Pxy[:,0], Pxy[:,1], u, v)
    return np.c_[dlon, dlat], u, v

tsec = 0.0
for step in range(STEPS):
    if not np.any(active):
        break

    act_idx = np.where(active)[0]
    P_act = P[act_idx]

    k1, _, _ = f(tsec, P_act)
    k2, _, _ = f(tsec + 0.5*DT_STEP, P_act + 0.5*DT_STEP*k1)
    k3, _, _ = f(tsec + 0.5*DT_STEP, P_act + 0.5*DT_STEP*k2)
    k4, _, _ = f(tsec + 1.0*DT_STEP, P_act + 1.0*DT_STEP*k3)

    P_next = P_act + (DT_STEP/6.0)*(k1 + 2*k2 + 2*k3 + k4)

    # 1) NaN / inf
    bad = ~np.isfinite(P_next).all(axis=1)

    # 2) hors domaine global
    out = (P_next[:,0] < DOMAIN[0]) | (P_next[:,0] > DOMAIN[1]) | (P_next[:,1] < DOMAIN[2]) | (P_next[:,1] > DOMAIN[3])

    # 3) sur "terre" -> si u/v non finies au point suivant (test sur S1 prioritaire)
    u_test, v_test = uv_one_source(S1, TREE1, SHP1, tsec, P_next[:,0], P_next[:,1])
    land = ~np.isfinite(u_test) | ~np.isfinite(v_test)

    # applique mise à jour
    P[act_idx] = P_next
    kill = bad | out | land
    active[act_idx[kill]] = False

    # stocke uniquement celles toujours actives
    for j, gi in enumerate(act_idx):
        if active[gi]:
            trajs[gi].append(P[gi].copy())

    tsec += DT_STEP

# =========================
# Nettoyage trajs (évite flèches manquantes + bruit)
# =========================
trajs_clean = [T for T in trajs if len(T) >= MIN_TRAJ_LEN]

if len(trajs_clean) == 0:
    raise RuntimeError("Aucune trajectoire assez longue. "
                       "➜ Diminue MIN_TRAJ_LEN ou MIN_SPEED_INIT, ou élargis la box.")

# =========================
# Figure PNG (fond vitesse à t0, tracé lon/lat)
# =========================
U0 = S1["U"].isel(time=0).values
V0 = S1["V"].isel(time=0).values
speed0 = np.sqrt(U0**2 + V0**2)

lon2d = S1["lon2d"]
lat2d = S1["lat2d"]

plt.figure(figsize=(12, 9))
# pcolormesh en lon/lat (plus "scientifique" que imshow indices)
plt.pcolormesh(lon2d, lat2d, speed0, shading="auto")
plt.colorbar(label="m/s")
plt.title(f"Trajectoires Lagrangiennes ({HOURS} h, IDW+RK4, fused={'oui' if S2 else 'non'})")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# Trace
for T in trajs_clean:
    T = np.asarray(T)
    plt.plot(T[:,0], T[:,1], lw=1.6)

    # flèche direction (dernier segment)
    if len(T) >= 2:
        a = T[-2]
        b = T[-1]
        plt.annotate(
            "", xy=(b[0], b[1]), xytext=(a[0], a[1]),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="black")
        )

# zoom sur box si active
if USE_START_BOX:
    plt.xlim(START_BOX["lon_min"], START_BOX["lon_max"])
    plt.ylim(START_BOX["lat_min"], START_BOX["lat_max"])

img_path = OUT / f"hycom_traj_{HOURS}h_clean_pro2.png"
plt.tight_layout()
plt.savefig(img_path, dpi=160)
print("✅ PNG :", img_path)

# =========================
# Exports GeoJSON + CZML
# =========================
gj_path = OUT / f"hycom_traj_{HOURS}h_clean_pro2.geojson"
cz_path = OUT / f"hycom_traj_{HOURS}h_clean_pro2.czml"

export_geojson(trajs_clean, gj_path)

# time start (tu peux aussi prendre ds.time[0] si tu veux)
t0_iso = "2025-11-17T20:00:00Z"
export_czml_points_and_paths(trajs_clean, t0_iso, DT_STEP, cz_path)

print("✅ GeoJSON :", gj_path)
print("✅ CZML    :", cz_path)
print("🎯 Ouvre le CZML dans Cesium/NaVisu4D : points animés + path => direction claire.")
print(f"ℹ️ Trajs gardées: {len(trajs_clean)} / {N_PART} (filtre MIN_TRAJ_LEN={MIN_TRAJ_LEN}, MIN_SPEED_INIT={MIN_SPEED_INIT})")
