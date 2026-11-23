import os
import xarray as xr
import numpy as np

u_dir = "C:/Users/peter/Desktop/UNI/Masterarbeit/program_copy/data/ERA5/u_component_of_wind/"
v_dir = "C:/Users/peter/Desktop/UNI/Masterarbeit/program_copy/data/ERA5/v_component_of_wind/"
output_dir = "C:/Users/peter/Desktop/UNI/Masterarbeit/program_copy/data/ERA5/wind_speed/"

os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(u_dir):
    if file.endswith(".nc"):
        
        # Jahr extrahieren (anpassen falls nötig)
        year = file.split("_")[-1].split(".")[0]
        print(year)
        year_int = int(year)
        if year_int<2022:
            continue
        u_path = os.path.join(u_dir, file)
        v_path = os.path.join(v_dir, file.replace("u", "v"))
        
        if not os.path.exists(v_path):
            print(f"Keine passende v-Datei für {file}")
            continue
        
        print(f"Verarbeite Jahr: {year}")
        
        # Datensätze laden
        ds_u = xr.open_dataset(u_path)
        ds_v = xr.open_dataset(v_path)
        print(ds_u["time"].shape)
        print(ds_v["time"].shape)

        print((ds_u["time"] == ds_v["time"]).all())
        # Variablen auswählen (dein Format!)
        u = ds_u["u_component_of_wind"]
        v = ds_v["v_component_of_wind"]
        ds_u.close()
        ds_v.close()
        # Windgeschwindigkeit berechnen
        wind_speed = np.sqrt(u**2 + v**2)
        
        # Metadaten hinzufügen
        wind_speed.name = "wind_speed"
        wind_speed.attrs["units"] = "m/s"
        wind_speed.attrs["long_name"] = "Windgeschwindigkeit berechnet aus u und v"
        
        # Dataset behalten (inkl. coords wie latitude, longitude, time, level)
        ds_out = wind_speed.to_dataset()
        
        # Datei speichern
        out_path = os.path.join(output_dir, f"era5_daily_weatherbench2_300hPa_wind_speed_{year}.nc")
        ds_out.to_netcdf(out_path)

print("Fertig!")