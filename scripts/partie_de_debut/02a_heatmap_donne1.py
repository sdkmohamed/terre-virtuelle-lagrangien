import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

FILE = "../../data/donne1.nc"
T_INDEX = 0  # premier pas de temps
D_INDEX = 0  # profondeur 0

ds = xr.open_dataset(FILE)
U = ds["u"].isel(time=T_INDEX, depth=D_INDEX)
V = ds["v"].isel(time=T_INDEX, depth=D_INDEX)
lon = ds["lon"].values
lat = ds["lat"].values

speed = np.sqrt(U.values**2 + V.values**2)

print("Shape:", speed.shape)
print("Min/Max speed (m/s):", float(speed.min()), float(speed.max()))

plt.figure(figsize=(8, 6))
plt.imshow(speed, origin="lower")
plt.colorbar(label="m/s")
plt.title("Vitesse des courants - HYCOM t0")
plt.tight_layout()
plt.savefig("../../output/hycom_speed_t0.png")
print("✅ Heatmap enregistrée dans ../output/hycom_speed_t0.png")
