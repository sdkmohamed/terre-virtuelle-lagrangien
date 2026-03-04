# scripts/01c_list_hdf5_tree.py
from pathlib import Path
import h5py

H5_PATH = Path("../data/donne3.h5")

def walk(name, obj):
    if isinstance(obj, h5py.Dataset):
        shape = "×".join(str(s) for s in obj.shape)
        print(f"  • {name}  -> dataset  shape={shape}  dtype={obj.dtype}")
        # affiche quelques attributs utiles s'ils existent
        if obj.attrs:
            keys = ", ".join([str(k) for k in obj.attrs.keys()])
            print(f"      attrs: {keys}")
    elif isinstance(obj, h5py.Group):
        print(f"[Group] {name}")

def main():
    if not H5_PATH.exists():
        print(f"Fichier introuvable: {H5_PATH}")
        return
    print("="*80)
    print(f"[HDF5] {H5_PATH.name}")
    with h5py.File(H5_PATH, "r") as f:
        f.visititems(walk)  # parcourt tout l'arbre

if __name__ == "__main__":
    main()
