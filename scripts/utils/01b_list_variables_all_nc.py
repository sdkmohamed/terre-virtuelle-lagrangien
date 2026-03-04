# scripts/01b_list_variables_all_nc.py
from pathlib import Path
import xarray as xr

DATA_DIR = Path("../data")
FILES = [DATA_DIR/"donne1.nc", DATA_DIR/"donne2.nc"]

def dump_nc(nc_path: Path):
    print("="*80)
    print(f"[NetCDF] {nc_path.name}")
    ds = xr.open_dataset(nc_path)
    print("- Dimensions:")
    for k, v in ds.dims.items():
        print(f"  • {k}: {int(v)}")
    print("- Variables:")
    for name, var in ds.variables.items():
        dims = ", ".join(var.dims)
        shape = "×".join(str(s) for s in var.shape)
        units = var.attrs.get("units", "")
        print(f"  • {name:20s}  dims=({dims:})  shape={shape}  units={units}")
    print("- Attributs globaux:", ds.attrs)
    ds.close()

for f in FILES:
    if f.exists():
        dump_nc(f)
    else:
        print(f"Fichier introuvable: {f}")
