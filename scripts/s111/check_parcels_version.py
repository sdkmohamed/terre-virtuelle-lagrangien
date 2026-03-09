"""Lance ce script pour connaitre la version exacte de Parcels et ses APIs disponibles."""
import parcels
print("Version Parcels :", parcels.__version__)

# Verifie si 'recovery' est accepte par execute()
import inspect
sig = inspect.signature(parcels.ParticleSet.execute)
print("Parametres de execute() :", list(sig.parameters.keys()))

# Verifie les codes d'erreur disponibles
print("\n--- statuscodes ---")
import parcels.tools.statuscodes as sc
print(dir(sc))