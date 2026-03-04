# -*- coding: utf-8 -*-
"""
Visualisation des résultats Parcels (Bretagne)
"""

from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ==========================================================
# Chemins
# ==========================================================
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "output"
ZARR_FILE = OUT / "parcels_hycom_bretagne_23h.zarr"

print("📁 Lecture de :", ZARR_FILE)
print("   Existe ?", ZARR_FILE.exists())

# ==========================================================
# Chargement des données
# ==========================================================
ds = xr.open_zarr(ZARR_FILE)

print("\n📊 Contenu du fichier :")
print(ds)

# ==========================================================
# Statistiques
# ==========================================================
n_traj = len(ds.trajectory)
n_obs_dim = len(ds.obs)

print(f"\n✅ Nombre de trajectoires : {n_traj}")
print(f"✅ Dimension obs : {n_obs_dim}")

# Calcul des observations valides (non-NaN) par trajectoire
valid_obs_per_traj = (~np.isnan(ds.lon.values)).sum(axis=1)
print(f"\n📈 Observations valides par trajectoire :")
print(f"   Min : {valid_obs_per_traj.min()}")
print(f"   Max : {valid_obs_per_traj.max()}")
print(f"   Moyenne : {valid_obs_per_traj.mean():.1f}")

# Durée de la simulation
if 'time' in ds.variables:
    # Le temps est en secondes depuis le début
    # Durée de la simulation (en heures)
    times = ds.time.values  # datetime64
    t0 = np.nanmin(times)
    t1 = np.nanmax(times)

    duration_hours = (t1 - t0) / np.timedelta64(1, 'h')

    print(f"\n⏱️  Durée de la simulation : {duration_hours:.2f} heures")

# ==========================================================
# Visualisation
# ==========================================================
print("\n🎨 Création de la carte...")

fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

# Limites géographiques
lon_min, lon_max = -6.5, -1.0
lat_min, lat_max = 47.0, 50.8
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

# Fond de carte
ax.add_feature(cfeature.LAND, color='lightgray', edgecolor='black', linewidth=0.5)
ax.add_feature(cfeature.OCEAN, color='lightblue')
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=':')

# Grille
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Trajectoires
print("   Tracé des trajectoires...")
n_plotted = 0
for i in range(n_traj):
    lon_traj = ds.lon.values[i, :]
    lat_traj = ds.lat.values[i, :]
    
    # Filtrer les NaN
    valid = ~np.isnan(lon_traj) & ~np.isnan(lat_traj)
    
    if valid.sum() > 1:  # Au moins 2 points
        ax.plot(lon_traj[valid], lat_traj[valid], 
                color='red', alpha=0.3, linewidth=0.8, 
                transform=ccrs.PlateCarree())
        
        # Point de départ (vert)
        ax.plot(lon_traj[valid][0], lat_traj[valid][0], 
                'go', markersize=3, transform=ccrs.PlateCarree())
        
        # Point d'arrivée (bleu)
        ax.plot(lon_traj[valid][-1], lat_traj[valid][-1], 
                'bo', markersize=3, transform=ccrs.PlateCarree())
        
        n_plotted += 1

print(f"   → {n_plotted} trajectoires tracées")

# Légende
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=8, label='Départ'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=8, label='Arrivée'),
    Line2D([0], [0], color='red', linewidth=2, alpha=0.5, label='Trajectoires')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

ax.set_title('Trajectoires de particules - Bretagne (23h)\n' +
             f'{n_traj} particules - HYCOM', 
             fontsize=14, fontweight='bold')

# Sauvegarde
output_png = OUT / "trajectoires_bretagne_23h.png"
plt.tight_layout()
plt.savefig(output_png, dpi=150, bbox_inches='tight')
print(f"\n💾 Carte sauvegardée : {output_png}")

plt.show()

print("\n✅ Visualisation terminée !")