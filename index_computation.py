from pathlib import Path
import datetime as dt

import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import zarr
import pandas as pd

from data_registry import Bibliography
from domino.core import LaggedAnalyser, PatternFilter, IndexGenerator
from domino.prediction import PredictionTest
from domino import agg
from coords_workflow_utils import PreparedCoordsLoader, NUTSMaskTools
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, auc

import cartopy.crs as ccrs
import cmocean.cm as cm

# -----CONFIGURATION----------
bib = Bibliography(Path("data"))
bib.add("energy", "WON", "daily", "subcountry", "prepared/WON_daily_subcountry_prepared_DJF.csv")
# bib.add("energy", "SPV", "daily", "subcountry", "prepared/SPV_daily_subcountry_prepared_DJF.csv")
bib.add("atmosphere", "mean_sea_level_pressure", "daily", "subcountry", "ERA5/prepared/DJF_anom/mean_sea_level_pressure")
# bib.add("atmosphere", "geopotential", "daily", "subcountry", "ERA5/prepared/DJF_anom/geopotential500")
# bib.add("atmosphere", "wind_speed", "daily", "subcountry", "ERA5/prepared/DJF_anom/wind_speed")
# bib.add("atmosphere", "v_component_of_wind", "daily", "subcountry", "ERA5/prepared/DJF_anom/v_component_of_wind")
# bib.add('atmosphere', 'total_column_water_vapour', 'daily', 'global', 'ERA5/prepared/DJF_anom/total_column_water_vapour')
# bib.add('atmosphere', '2m_temperature', 'daily', 'global', 'ERA5/prepared/DJF_anom/2m_temperature')
# bib.add('atmosphere', 'relative_humidity', 'daily', 'global', 'ERA5/prepared/DJF_anom/relative_humidity700')

# -----VARIABLES-----------
erstelldatum = dt.date.today()
lags = [-4,-3,-2,-1,0] # [-6, -5, -4, -3, -2, -1, 0]
region_ziel = "TenneT"  # region der ziel variablen
region_prec = "EUROPE"  # welche Daten werden als precursor verwendet
timeperiod = "1979-2025"  # zeitraum der analysierten daten
quant = 0.1  # welches quantile => def extreme event
extreme_event = "l"  # l=low 0.1, h=high 0.9
cv_version = ''
maskierung = '_noMask'
test_version = 'tso_cv' # anpassen im pfad zu index_parameter
test_version_region = '_TenneT-with-Amprion'

# -----OPTIONS-----------
use_deseasonalization = True  # Deseasonalisierung durchführen?
deseasonalization_agg = "dayofyear"  # Aggregationsmethode: 'dayofyear', 'month', etc.
deseasonalization_smooth = 30  # Glättungsfenster für deseasonalization
use_bootstrap = False  # Bootstrap Significance Test durchführen?
bootstrap_num = 100  # Anzahl Bootstrap-Samples
bootstrap_p = 0.05  # Signifikanzniveau
min_connected_size = 10

energy_entries = bib.get_by_category("energy")
atm_entries = bib.get_by_category("atmosphere")

# load event and precursor data to set up analyser
energy_entry = energy_entries[0]
atm_entry = atm_entries[0]

event = PreparedCoordsLoader.load_energy_event_from_prepared(
    path=energy_entry.path,
    energy_name=energy_entry.name,
    region_ziel=region_ziel,
    quant=quant,
    extreme_event=extreme_event,
)
print(event)

from regionalmeancalculator import RegionalMeanCalculator
@staticmethod
def load_energy_values_from_prepared(
    path: Path,
    energy_name: str,
    region_ziel: str | None,
) -> xr.DataArray:
    """
    Load prepared energy data (long format) and return absolute values
    (no thresholding, no binarization).

    Expected CSV format:
    Date, NUTS_ID, <energy_name>, lon, lat
    """
    import pandas as pd
    import xarray as xr

    df = pd.read_csv(path, parse_dates=["Date"])

    if energy_name not in df.columns:
        raise ValueError(f"Spalte '{energy_name}' fehlt in {path}")

    # long → wide
    df_wide = (
        df.pivot_table(
            index="Date",
            columns="NUTS_ID",
            values=energy_name,
            aggfunc="mean"
        )
        .sort_index()
    )

    # Regionale Aggregation
    if region_ziel is not None:
        calculator = RegionalMeanCalculator()
        series = calculator.calculate_mean(df=df_wide, region_name=region_ziel)
    else:
        series = df_wide.mean(axis=1)
        series.name = energy_name

    values_da = xr.DataArray(
        series.to_numpy(),
        dims="time",
        coords={"time": series.index.to_numpy()},
        name=energy_name,
    )

    return values_da

event_abs = load_energy_values_from_prepared(
    path=energy_entry.path,
    energy_name=energy_entry.name,
    region_ziel=region_ziel
)
print(event_abs)

# lade einzelne atm variablen als xr.Dataset
atm_entry = atm_entries[0]
ATM_NAME = atm_entry.name # "geopotential", "mean_sea_level_pressure", "2m_temperature"
LEVEL = '' # '' for non level data
WEATHERBENCH_DAILY_PATH = atm_entry.path
print(WEATHERBENCH_DAILY_PATH)
print("[INFO] Merging all years into a single dataset...")
files = sorted(WEATHERBENCH_DAILY_PATH.glob("*.nc")) #era5_daily_
ds_all = xr.open_mfdataset(
    files, combine="by_coords",
    chunks={"time": 50, "latitude":50, "longitude":50} # changed time from 365 for deseason
).load()
print("[INFO] Loaded and merged WeatherBench dailymeans dataset")

ds_djf = ds_all.sel(time=ds_all.time.dt.month.isin([12, 1, 2]))#ds_all.sel(init_time=ds_all.init_time.dt.month.isin([12, 1, 2]))
ds_all = ds_djf
month_idx = '_DJF'
del ds_djf
ds_msl = ds_all

ds_all = ds_msl#xr.merge([ds_msl, ds_z500, ds_ws])

# split into training and test data
training_predictor=ds_all.sel(time=slice('1980','2010'))
training_event=event.sel(time=slice('1980','2010'))
test_predictor=ds_all.sel(time=slice('2011','2022'))
test_event=event.sel(time=slice('2011','2022')) 

analyser = LaggedAnalyser(event=training_event, variables=training_predictor)
analyser.lag_variables(lags, offset_unit='days', mode='past')

composites = xr.open_dataset(f'data//domino_output_coords_weatherbench/{energy_entry.name}/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{"_".join(map(str, lags))}_composites_coords_deseasonalized_DJF_anom.nc')#analyser.compute_composites(as_anomaly=True)
lagged_means = xr.open_dataset(f'data//domino_output_coords_weatherbench/{energy_entry.name}/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{"_".join(map(str, lags))}_lagged-means_coords_deseasonalized_DJF_anom.nc')

analyser.composites = composites
analyser.lagged_means = lagged_means
analyser.composites_are_anomaly = True
analyser.composite_func = (agg.mean_ds, agg.cat_occ_ds)

atm_entry = atm_entries[0]
ATM_NAME = atm_entry.name
S = xr.open_dataset(f'data//domino_output_coords_weatherbench/{energy_entry.name}/100/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{"_".join(map(str, lags))}_bootstrap-composites_deseasonalized_DJF_anom.nc')

S = S.assign_coords(
    longitude=((S.longitude + 180) % 360) - 180
).sortby("longitude")
analyser.composite_sigs = S
masked_composite=S

PF = PatternFilter(analyser=analyser)
thresh_dict=dict(relative_humidity=0.05, total_column_water_vapour=1.0, **{'2m_temperature': 1.}, mean_sea_level_pressure=250, geopotential=300, wind_speed=2.5)

def abs_amplitude_over_threshold(val_ds,thresh_dict):
    return abs(val_ds)>xr.Dataset(thresh_dict)

PF.apply_value_mask(abs_amplitude_over_threshold,thresh_dict,mode='intersection')
mask1=PF.get_mask() #retrieve the new mask for plotting later.

IG=IndexGenerator()
data_vars = [atm_entries[0].name]#, 'geopotential', 'wind_speed'] mean_sea_level_pressure 2m_temperature total_column_water_vapour relative_humidity
var_name = data_vars[0]
training_predictor_partial = training_predictor[var_name]
test_predictor_partial = test_predictor[var_name]
masked_composite_partial = masked_composite[[var_name]]
masked_ds = composites #masked_composite_partial.where(mask1) #mask1 anwenden

# index berechnen für lags mit trainings daten
# index parameter abspeichern nach system: data/index_parameters/energy_variable/event/WON-l-predictor-index-1-training.nc
index_timelags = [-4,-3,-2,-1] #-4,-3,-2,-1,
for lag_id in index_timelags:
    print(f"Calculating index for lag {lag_id}...")
    index_slices=[dict(index_val=1,lag=lag_id)]
    indices=IG.generate(masked_ds,training_predictor_partial,dim='time',slices=index_slices)
    print(indices.data_vars)
    standardisation_params=IG.get_standardisation_params()
    print(standardisation_params)
    index_params_path = Path(f"data/index_parameters/{test_version}/{region_ziel}/{energy_entry.name}/{extreme_event}/{energy_entry.name}-{extreme_event}-{var_name}-index-{lag_id}-training{cv_version}{maskierung}{test_version_region}.nc")
    index_params_path.parent.mkdir(parents=True, exist_ok=True)
    standardisation_params.to_netcdf(index_params_path)
    print(f"Saved index parameters to {index_params_path}")
