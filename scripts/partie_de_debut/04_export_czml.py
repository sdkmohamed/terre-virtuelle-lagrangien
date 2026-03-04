from pathlib import Path
import json, datetime as dt

# chemins
ROOT = Path(__file__).resolve().parents[1]     # .../scripts
OUT  = ROOT.parent / "output"
GJ = OUT / "hycom_traj_demo.geojson"           # produit par 03_lagrange_demo.py
CZ = OUT / "hycom_traj_demo.czml"

# paramètres temps pour l’animation
t0 = dt.datetime(2025, 1, 1, 12, 0, 0)          # début arbitraire
step_seconds = 900                               # le même DT que dans 03 (15 min)

with open(GJ, "r", encoding="utf-8") as f:
    geo = json.load(f)

packets = []
clock_packet = {
    "id": "document",
    "version": "1.0",
    "clock": {
        "interval": "",  # rempli après
        "currentTime": "",
        "multiplier": 60,
        "range": "LOOP_STOP"
    }
}
packets.append(clock_packet)

# Construire CZML: une polyline temporelle par trajectoire
for i, feat in enumerate(geo["features"]):
    coords = feat["geometry"]["coordinates"]  # [ [lon,lat], ... ]
    # construire une liste time-tagged: [iso, lon, lat, height, iso, lon, lat, ...]
    # on fait un timestamp par point, espacé de step_seconds
    flat = []
    for k, (lon, lat) in enumerate(coords):
        t = (t0 + dt.timedelta(seconds=k*step_seconds)).isoformat()+"Z"
        flat += [t, float(lon), float(lat), 0.0]

    packets.append({
        "id": f"traj_{i}",
        "polyline": {
            "positions": {"epoch": t0.isoformat()+"Z", "cartographicDegrees": flat},
            "width": 2,
            "material": {
                "polylineOutline": {
                    "color": {"rgba": [0, 255, 255, 255]},
                    "outlineColor": {"rgba": [0, 0, 0, 180]},
                    "outlineWidth": 1
                }
            }
        }
    })

# Définir l’intervalle du clock
t_start = t0
t_stop  = t0 + dt.timedelta(seconds=(len(geo["features"][0]["geometry"]["coordinates"])-1)*step_seconds)
packets[0]["clock"]["interval"] = f"{t_start.isoformat()}Z/{t_stop.isoformat()}Z"
packets[0]["clock"]["currentTime"] = t_start.isoformat()+"Z"

with open(CZ, "w", encoding="utf-8") as f:
    json.dump(packets, f)

print(f"✅ CZML écrit: {CZ}")
print("Ouvre-le ensuite depuis une petite page Cesium.")
