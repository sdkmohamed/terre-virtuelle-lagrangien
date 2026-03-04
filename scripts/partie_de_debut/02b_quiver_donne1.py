import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

FILE = "../../data/donne1.nc"
T_INDEX = 0
D_INDEX = 0

# Lecture des données
ds = xr.open_dataset(FILE)
U = ds["u"].isel(time=T_INDEX, depth=D_INDEX)
V = ds["v"].isel(time=T_INDEX, depth=D_INDEX)
lon = ds["lon"].values
lat = ds["lat"].values
speed = np.sqrt(U.values**2 + V.values**2)

# Affichage de la heatmap
plt.figure(figsize=(10, 8))
plt.imshow(speed, origin="lower")
plt.colorbar(label="m/s")
plt.title("Direction et vitesse des courants - HYCOM t0")

# Ajout des flèches (quiver)
step_y, step_x = 20, 20  # une flèche toutes les 20 cases pour lisibilité
Y, X = np.mgrid[0:U.shape[0]:step_y, 0:U.shape[1]:step_x]
plt.quiver(X, Y, U.values[::step_y, ::step_x], V.values[::step_y, ::step_x],
           color="white", scale=50, width=0.002)

plt.tight_layout()
plt.savefig("../../output/hycom_quiver_t0.png")
plt.show()
print("✅ Quiver enregistré dans ../../output/hycom_quiver_t0.png")
