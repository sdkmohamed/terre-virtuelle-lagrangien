import xarray as xr

# change le nom du fichier ici si tu veux tester un autre
FILE = "../data/donne1.nc"

ds = xr.open_dataset(FILE)
print("Variables :")
print(list(ds.variables))
print("\nDimensions :")
print(ds.dims)
