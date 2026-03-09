# -*- coding: utf-8 -*-
"""
01_explore_s111.py
------------------
Exploration de la structure d'un fichier HDF5 au format S-111.
À lancer en premier pour comprendre l'organisation du fichier.

Usage :
    python scripts/s111/01_explore_s111.py
"""

from pathlib import Path
import numpy as np
import h5py

# ========================================================== #
# Chemins                                                     #
# ========================================================== #
ROOT  = Path(__file__).resolve().parents[2]   # racine projet
DATA  = ROOT / "data" / "s111"
OUT   = ROOT / "output"
OUT.mkdir(exist_ok=True)

# Cherche automatiquement le premier .h5 dans data/s111/
h5_files = list(DATA.glob("*.h5")) + list(DATA.glob("*.hdf5"))
if not h5_files:
    raise FileNotFoundError(f"Aucun fichier HDF5 trouvé dans {DATA}")
H5_FILE = h5_files[0]
print(f"📂 Fichier S-111 : {H5_FILE}")
print(f"   Taille        : {H5_FILE.stat().st_size / 1024**2:.1f} Mo\n")

# ========================================================== #
# Exploration récursive                                       #
# ========================================================== #
def print_h5_tree(obj, indent=0):
    """Affiche l'arborescence groupes/datasets d'un fichier HDF5."""
    prefix = "  " * indent
    if isinstance(obj, h5py.File):
        print(f"{prefix}📁 [RACINE]")
        for key in obj.keys():
            print_h5_tree(obj[key], indent + 1)
    elif isinstance(obj, h5py.Group):
        print(f"{prefix}📁 {obj.name.split('/')[-1]}/")
        # Attributs du groupe
        for attr_name, attr_val in obj.attrs.items():
            print(f"{prefix}   @{attr_name} = {attr_val}")
        for key in obj.keys():
            print_h5_tree(obj[key], indent + 1)
    elif isinstance(obj, h5py.Dataset):
        shape = obj.shape
        dtype = obj.dtype
        print(f"{prefix}📊 {obj.name.split('/')[-1]}  shape={shape}  dtype={dtype}")
        # Attributs du dataset
        for attr_name, attr_val in obj.attrs.items():
            print(f"{prefix}   @{attr_name} = {attr_val}")

with h5py.File(H5_FILE, "r") as f:
    print("=" * 60)
    print("ARBORESCENCE DU FICHIER S-111")
    print("=" * 60)
    print_h5_tree(f)

# ========================================================== #
# Affichage des attributs globaux (métadonnées IHO S-100)    #
# ========================================================== #
print("\n" + "=" * 60)
print("ATTRIBUTS GLOBAUX (métadonnées)")
print("=" * 60)
with h5py.File(H5_FILE, "r") as f:
    for k, v in f.attrs.items():
        print(f"  {k:40s} = {v}")

# ========================================================== #
# Repérage automatique des groupes SurfaceCurrent            #
# ========================================================== #
print("\n" + "=" * 60)
print("GROUPES SurfaceCurrent détectés")
print("=" * 60)

def find_surface_current_groups(f):
    """Retourne les chemins des groupes contenant les données de courant."""
    found = []
    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and "SurfaceCurrent" in name:
            found.append(name)
    f.visititems(visitor)
    return found

with h5py.File(H5_FILE, "r") as f:
    sc_groups = find_surface_current_groups(f)
    if not sc_groups:
        # Fallback : cherche tous les datasets nommés 'speed' ou 'direction' ou 'u/v'
        print("  (pas de groupe SurfaceCurrent explicite – scan des datasets...)")
        def list_datasets(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  dataset: {name}  shape={obj.shape}")
        f.visititems(list_datasets)
    else:
        for g in sc_groups:
            print(f"  ✅  {g}")

print("\n✅ Exploration terminée. Relisez la sortie pour identifier :")
print("   • Le groupe contenant speed/direction ou surfaceCurrentSpeed/Direction")
print("   • Les variables de grille : lon/lat ou les attributs gridOriginLongitude etc.")
print("   • La variable temps (timeRecordInterval, dateTimeOfFirstRecord, ...)")
