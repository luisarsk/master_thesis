import os
import numpy as np
import xarray as xr
# from weatherbenchX.data_loaders import xarray_loaders
from dask.diagnostics import ProgressBar
from pathlib import Path

ERA5_ZARR = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
VARIABLE = "relative_humidity"  # Beispielvariable, z.B. "2m_temperature", "total_precipitation", etc.
LEVEL = 700  # None falls keine Levels
LAT_MIN, LAT_MAX = 84, 12
LON_MIN1, LON_MAX1 = 270, 360
LON_MIN2, LON_MAX2 = 0, 50

START_YEAR, END_YEAR = 2018, 2022
output_dir = Path(f"data/ERA5/{VARIABLE}{LEVEL}")
output_dir.mkdir(parents=True, exist_ok=True)

ds = xr.open_zarr(ERA5_ZARR, consolidated=True, decode_times=True, chunks={})[VARIABLE]

for year in range(START_YEAR, END_YEAR + 1):
    print(f"[INFO] Processing year {year}...")
    ds_sel1 = ds.sel(
        time=slice(f"{year}-01-01", f"{year}-12-31"),
        latitude=slice(LAT_MIN, LAT_MAX),   # Zuerst max, dann min falls Daten abnehmend
        longitude=slice(LON_MIN1, LON_MAX1),
        level=LEVEL
    )
    # print(ds_sel1)
    ds_sel2 = ds.sel(
        time=slice(f"{year}-01-01", f"{year}-12-31"),
        latitude=slice(LAT_MIN, LAT_MAX),   # Zuerst max, dann min falls Daten abnehmend
        longitude=slice(LON_MIN2, LON_MAX2),
        level=LEVEL
    )
    # print(ds_sel2)
    ds_sel = xr.concat([ds_sel1, ds_sel2], dim="longitude")
    print(ds_sel)

    print('Now resample...')
    ds_daily = ds_sel.resample(time="1D").mean()
    print(ds_daily)

    ds_daily = ds_daily.chunk({'time': 30, 'latitude': 50, 'longitude': 50})
    # -------------------------------
    # Als NetCDF speichern (lazy, Dask chunks)
    # -------------------------------

    OUTPUT_NC_YEAR = output_dir / f'era5_daily_weatherbench2_{VARIABLE}{LEVEL}_{year}.nc'
    with ProgressBar():
        ds_daily.to_netcdf(OUTPUT_NC_YEAR, mode="w", format="NETCDF4")
    print(f"Gespeichert: {OUTPUT_NC_YEAR}")
