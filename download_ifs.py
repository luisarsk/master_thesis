import gcsfs
import xarray as xr
import time
import os
import numpy as np

var = "2m_temperature"
start_year = [2019, 2020, 2021]
end_year = [2020, 2021, 2022]
level_var = None

fs = gcsfs.GCSFileSystem(token="anon")

ds_pangu = xr.open_zarr(
    'gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr',
    consolidated=True,
    storage_options={"token": "anon"}
)

print(ds_pangu)
for start_year, end_year in zip(start_year, end_year):
    print(f"Processing period: {start_year}-{end_year}")
    start_time = time.time()
    # einzelne variabeln und zeiträume als nc speichern
    if level_var is not None:
        temperature = ds_pangu[var].sel(
            time=slice(f"{start_year}-12-01", f"{end_year}-02-28"),
            prediction_timedelta=slice(6, 96),
            latitude=slice(12, 84),
            longitude=((ds_pangu.longitude <= 50) | (ds_pangu.longitude >= 270)),
            level=level_var
        )
    else:
        temperature = ds_pangu[var].sel(
            time=slice(f"{start_year}-12-01", f"{end_year}-02-28"),
            prediction_timedelta=slice(6, 96),
            latitude=slice(12, 84),
            longitude=((ds_pangu.longitude <= 50) | (ds_pangu.longitude >= 270))
        )

    # Longitude von -90° bis 50° umrechnen und sortieren
    temperature = temperature.assign_coords(
        longitude=((temperature.longitude + 180) % 360) - 180
    ).sortby("longitude")

    savepath = f'data/ifs_hres/{var}/ifs_hres_{var}_{start_year}-{end_year}_DJF.nc'
    os.makedirs(os.path.dirname(savepath), exist_ok=True)

    temperature.to_netcdf(savepath)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")