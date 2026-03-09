# -*- coding: utf-8 -*-
"""
02_read_s111.py
---------------
Lecture d'un fichier S-111 (HDF5) et extraction des courants U/V.

Le format S-111 stocke les courants en (speed, direction) dans le groupe :
    SurfaceCurrent/SurfaceCurrent.01/Group_NNN/values

La grille peut être :
  - régulière  → attributs gridOriginLongitude/Latitude + gridSpacing
  - non-régulière (non traité ici, mais le script le détecte)

Sortie : dictionnaire Python avec clés lon, lat, u, v, times

Usage :
    python scripts/s111/02_read_s111.py
    # ou import depuis un autre script :
    from scripts.s111.read_s111 import load_s111
"""

from pathlib import Path
import numpy as np
import h5py

# ========================================================== #
# Chemins                                                     #
# ========================================================== #
ROOT    = Path(__file__).resolve().parents[2]
DATA    = ROOT / "data" / "s111"
OUT     = ROOT / "output"
OUT.mkdir(exist_ok=True)

h5_files = list(DATA.glob("*.h5")) + list(DATA.glob("*.hdf5"))
if not h5_files:
    raise FileNotFoundError(f"Aucun fichier HDF5 trouvé dans {DATA}")
H5_FILE = h5_files[0]


# ========================================================== #
# Fonctions utilitaires                                       #
# ========================================================== #

def decode_attr(val):
    """Décode un attribut HDF5 qui peut être bytes ou str."""
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8")
    return val


def build_regular_grid(f, group_path):
    """
    Construit lon2d / lat2d à partir des attributs de grille régulière S-111.
    Retourne (lon2d, lat2d) de shape (ny, nx).
    """
    grp = f[group_path]

    # ---- Origine et espacement ----
    lon0 = float(grp.attrs.get("gridOriginLongitude",
                 f.attrs.get("westBoundLongitude", 0.0)))
    lat0 = float(grp.attrs.get("gridOriginLatitude",
                 f.attrs.get("southBoundLatitude", 0.0)))

    # gridSpacing peut être un scalaire ou un vecteur [dlon, dlat]
    spacing = grp.attrs.get("gridSpacing", None)
    if spacing is None:
        spacing = grp.attrs.get("gridSpacingLongitudinal", 0.01)
    if hasattr(spacing, "__len__"):
        dlon, dlat = float(spacing[0]), float(spacing[1])
    else:
        dlon = dlat = float(spacing)

    # ---- Nombre de points ----
    nx = int(grp.attrs.get("numPointsLongitudinal",
             grp.attrs.get("numPointsX", 0)))
    ny = int(grp.attrs.get("numPointsLatitudinal",
             grp.attrs.get("numPointsY", 0)))

    if nx == 0 or ny == 0:
        raise ValueError(
            "Impossible de déterminer la taille de la grille. "
            "Vérifiez les attributs numPointsLongitudinal / numPointsLatitudinal."
        )

    lons = lon0 + np.arange(nx) * dlon   # (nx,)
    lats = lat0 + np.arange(ny) * dlat   # (ny,)
    lon2d, lat2d = np.meshgrid(lons, lats)  # (ny, nx)
    return lon2d, lat2d


def speed_dir_to_uv(speed, direction_deg):
    """
    Convertit (speed, direction_météo) → (u, v).
    Convention S-111 : direction = cap depuis lequel le courant vient (0°=N, 90°=E).
    → Certains fichiers S-111 utilisent la direction vers laquelle va le courant.
    Adaptez le signe si besoin après vérification.
    """
    theta = np.deg2rad(direction_deg)
    # Direction "vers laquelle va le courant" (convention océanographique)
    u =  speed * np.sin(theta)   # composante est
    v =  speed * np.cos(theta)   # composante nord
    return u.astype(np.float32), v.astype(np.float32)


def parse_datetime(date_str, time_str):
    """
    Convertit les dates S111 vers numpy.datetime64.
    Supporte les formats :
        20210901T000000
        20210901T000000Z
        20210901 + 000000
    """

    def decode(x):
        if isinstance(x, (bytes, np.bytes_)):
            return x.decode()
        return str(x)

    if date_str is None:
        return np.datetime64("2000-01-01T00:00:00")

    s = decode(date_str).replace("Z", "")

    # format 20210901T000000
    if "T" in s and len(s) >= 15:
        date = s[:8]
        time = s[9:15]

        iso = f"{date[:4]}-{date[4:6]}-{date[6:8]}T{time[:2]}:{time[2:4]}:{time[4:6]}"
        return np.datetime64(iso)

    # format séparé
    if time_str is not None:
        d = decode(date_str)
        t = decode(time_str).replace("Z", "")

        iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}T{t[:2]}:{t[2:4]}:{t[4:6]}"
        return np.datetime64(iso)

    return np.datetime64("2000-01-01T00:00:00")


# ========================================================== #
# Fonction principale : load_s111                             #
# ========================================================== #

def load_s111(h5_path: Path, bbox: dict = None, verbose: bool = True):
    """
    Charge un fichier S-111 HDF5 et retourne un dict avec :
      {
        'lon':   np.ndarray (ny, nx)  – longitude
        'lat':   np.ndarray (ny, nx)  – latitude
        'u':     np.ndarray (nt, ny, nx) – vitesse est (m/s)
        'v':     np.ndarray (nt, ny, nx) – vitesse nord (m/s)
        'times': np.ndarray (nt,) datetime64  – instants
      }

    Paramètres
    ----------
    h5_path : Path
        Chemin du fichier .h5
    bbox : dict optionnel
        {'lon_min', 'lon_max', 'lat_min', 'lat_max'} pour rogner la grille
    verbose : bool
        Affiche des infos de progression
    """
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)

    if verbose:
        print(f"📂 Ouverture : {h5_path.name}")

    with h5py.File(h5_path, "r") as f:

        # ---- Localiser le groupe SurfaceCurrent.01 ----
        sc_key = None
        for k in f.keys():
            if "SurfaceCurrent" in k:
                sc_key = k
                break
        if sc_key is None:
            raise KeyError(
                "Groupe 'SurfaceCurrent' introuvable. "
                "Lancez 01_explore_s111.py pour inspecter le fichier."
            )

        sc_grp_name = f"{sc_key}/{sc_key}.01"
        if sc_grp_name not in f:
            # Fallback : premier sous-groupe
            first_sub = list(f[sc_key].keys())[0]
            sc_grp_name = f"{sc_key}/{first_sub}"

        sc_grp = f[sc_grp_name]
        if verbose:
            print(f"   Groupe courant : {sc_grp_name}")

        # ---- Construction de la grille ----
        lon2d, lat2d = build_regular_grid(f, sc_grp_name)
        ny, nx = lon2d.shape
        if verbose:
            print(f"   Grille : {ny} × {nx}")
            print(f"   Lon : [{lon2d.min():.3f}, {lon2d.max():.3f}]")
            print(f"   Lat : [{lat2d.min():.3f}, {lat2d.max():.3f}]")

        # ---- Récupération des pas de temps (Group_000, Group_001, …) ----
        time_groups = sorted(
            [k for k in sc_grp.keys() if k.startswith("Group")],
            key=lambda x: int(x.split("_")[-1])
        )
        if verbose:
            print(f"   Pas de temps : {len(time_groups)}")

        if not time_groups:
            raise KeyError(
                "Aucun groupe 'Group_NNN' trouvé dans le groupe SurfaceCurrent. "
                "Structure du fichier non standard."
            )

        # ---- Temps de référence ----
        dt_first = sc_grp.attrs.get("dateTimeOfFirstRecord", None)
        dt_date  = sc_grp.attrs.get("dateOfFirstRecord", None)
        dt_time  = sc_grp.attrs.get("timeOfFirstRecord", None)

        if dt_first is not None:
            t0 = parse_datetime(dt_first, None)
        elif dt_date is not None:
            t0 = parse_datetime(dt_date, dt_time)
        else:
            t0 = np.datetime64("2000-01-01T00:00:00")
            if verbose:
                print("   ⚠️  Pas de date de référence – t0 = 2000-01-01T00:00:00")

        # Intervalle entre pas de temps (secondes)
        dt_interval_s = float(sc_grp.attrs.get("timeRecordInterval", 3600))
        if verbose:
            print(f"   t0 = {t0}")
            print(f"   Intervalle = {dt_interval_s:.0f} s ({dt_interval_s/3600:.2f} h)")

        # ---- Lecture des champs de vitesse ----
        u_list, v_list, times = [], [], []
        for i, grp_name in enumerate(time_groups):
            grp = sc_grp[grp_name]
            # Dans S-111 les données sont dans un dataset nommé 'values'
            # avec des champs composés : surfaceCurrentSpeed, surfaceCurrentDirection
            if "values" not in grp:
                # Essaye des noms alternatifs
                alt_names = [k for k in grp.keys()]
                if verbose and i == 0:
                    print(f"   ⚠️  Pas de 'values', datasets trouvés : {alt_names}")
                val_ds_name = alt_names[0] if alt_names else None
            else:
                val_ds_name = "values"

            if val_ds_name is None:
                continue

            vals = grp[val_ds_name][:]
            # Le dataset peut être structuré (compound) ou (ny*nx, 2)
            if vals.dtype.names:
                # Dataset compound (le plus courant en S-111)
                speed_names = [n for n in vals.dtype.names
                               if "speed" in n.lower() or "Speed" in n]
                dir_names   = [n for n in vals.dtype.names
                               if "direction" in n.lower() or "Direction" in n]
                if not speed_names or not dir_names:
                    raise KeyError(
                        f"Champs compound inconnus : {vals.dtype.names}. "
                        "Adaptez speed_names / dir_names dans le script."
                    )
                speed_flat = vals[speed_names[0]].astype(np.float32)
                dir_flat   = vals[dir_names[0]].astype(np.float32)
            elif vals.ndim == 2 and vals.shape[1] == 2:
                speed_flat = vals[:, 0].astype(np.float32)
                dir_flat   = vals[:, 1].astype(np.float32)
            else:
                # Suppose que c'est déjà (ny, nx, 2)
                speed_flat = vals[..., 0].flatten().astype(np.float32)
                dir_flat   = vals[..., 1].flatten().astype(np.float32)

            # Remplacement des valeurs fill (souvent 9999 ou -9999)
            fill_val = float(grp[val_ds_name].attrs.get("fillValue", 9999.0))
            speed_flat = np.where(np.abs(speed_flat) >= fill_val * 0.9, np.nan, speed_flat)
            dir_flat   = np.where(np.abs(dir_flat)   >= fill_val * 0.9, np.nan, dir_flat)

            # Reshape en (ny, nx)
            speed_2d = speed_flat.reshape(ny, nx)
            dir_2d   = dir_flat.reshape(ny, nx)

            u_2d, v_2d = speed_dir_to_uv(speed_2d, dir_2d)
            u_list.append(u_2d)
            v_list.append(v_2d)

            t_i = t0 + np.timedelta64(int(i * dt_interval_s), "s")
            times.append(t_i)

        u_arr = np.stack(u_list, axis=0)  # (nt, ny, nx)
        v_arr = np.stack(v_list, axis=0)
        times = np.array(times)

        if verbose:
            print(f"\n✅ Données extraites :")
            print(f"   U shape : {u_arr.shape}  (nt, ny, nx)")
            print(f"   Plage U : [{np.nanmin(u_arr):.3f}, {np.nanmax(u_arr):.3f}] m/s")
            print(f"   Plage V : [{np.nanmin(v_arr):.3f}, {np.nanmax(v_arr):.3f}] m/s")
            print(f"   Temps   : {times[0]} → {times[-1]}")

        # ---- Rognage BBox optionnel ----
        if bbox:
            mask_lon = (lon2d[0, :] >= bbox["lon_min"]) & (lon2d[0, :] <= bbox["lon_max"])
            mask_lat = (lat2d[:, 0] >= bbox["lat_min"]) & (lat2d[:, 0] <= bbox["lat_max"])
            ix = np.where(mask_lon)[0]
            iy = np.where(mask_lat)[0]
            if len(ix) == 0 or len(iy) == 0:
                print("⚠️  BBox vide – données non rognées")
            else:
                lon2d = lon2d[np.ix_(iy, ix)]
                lat2d = lat2d[np.ix_(iy, ix)]
                u_arr = u_arr[:, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1]
                v_arr = v_arr[:, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1]
                if verbose:
                    print(f"   Après rognage : {u_arr.shape}")

        return {
            "lon":   lon2d,
            "lat":   lat2d,
            "u":     u_arr,
            "v":     v_arr,
            "times": times,
        }


# ========================================================== #
# Test standalone                                             #
# ========================================================== #
if __name__ == "__main__":
    data = load_s111(H5_FILE, verbose=True)
    print("\n🎯 Clés du dictionnaire retourné :", list(data.keys()))
    print("   lon   :", data["lon"].shape)
    print("   lat   :", data["lat"].shape)
    print("   u     :", data["u"].shape)
    print("   v     :", data["v"].shape)
    print("   times :", data["times"].shape)
