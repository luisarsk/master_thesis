from pathlib import Path
import datetime as dt

import xarray as xr
import matplotlib.pyplot as plt
import joblib

from data_registry import Bibliography
from regionalmeancalculator import RegionalMeanCalculator
from domino.core import LaggedAnalyser, PatternFilter, IndexGenerator
from domino import agg
from coords_workflow_utils import PreparedCoordsLoader
from sklearn.metrics import roc_auc_score, roc_curve

import time


start = time.time()
# -----CONFIGURATION----------
bib = Bibliography(Path("data"))
bib.add("energy", "WON", "daily", "subcountry", "prepared/WON_daily_subcountry_prepared_DJF.csv")
# bib.add("energy", "SPV", "daily", "subcountry", "prepared/SPV_daily_subcountry_prepared_DJF.csv")
bib.add("atmosphere", "mean_sea_level_pressure", "daily", "subcountry", "ERA5/prepared/DJF_anom/mean_sea_level_pressure")
bib.add("atmosphere", "geopotential", "daily", "subcountry", "ERA5/prepared/DJF_anom/geopotential500")
bib.add("atmosphere", "v_component_of_wind", "daily", "subcountry", "ERA5/prepared/DJF_anom/v_component_of_wind")
# bib.add("atmosphere", "2m_temperature", "daily", "global", "ERA5/prepared/DJF_anom/2m_temperature")
# bib.add("atmosphere", "relative_humidity", "daily", "global", "ERA5/prepared/DJF_anom/relative_humidity700")

# -----VARIABLES-----------
erstelldatum = dt.date.today()
lags = [-4, -3, -2, -1, 0]
region_ziel = "TenneT"
region_prec = "EUROPE"
timeperiod = "1979-2025"
quant = 0.9
extreme_event = "h"
maskierung = "_noMask"
model = 'pangu'
test_version = 'tso_cv' # falls leer, unten im pfad anpassen zu den index_parameters
test_version_region = '_TenneT-with-Amprion'

energy_entries = bib.get_by_category("energy")
atm_entries = bib.get_by_category("atmosphere")

## -----LOAD DATA----------
energy_entry = energy_entries[0]

event = PreparedCoordsLoader.load_energy_event_from_prepared(
    path=energy_entry.path,
    energy_name=energy_entry.name,
    region_ziel=region_ziel,
    quant=quant,
    extreme_event=extreme_event,
)
print(event)

# Load each variable as xr.Dataset
atm_entry = atm_entries[0]
WEATHERBENCH_DAILY_PATH = atm_entry.path
print(WEATHERBENCH_DAILY_PATH)
print("[INFO] Merging all years into a single dataset...")
files = sorted(WEATHERBENCH_DAILY_PATH.glob("*.nc"))
ds_all = xr.open_mfdataset(
    files,
    combine="by_coords",
    chunks={"time": 50, "latitude": 50, "longitude": 50},
).load()
print("[INFO] Loaded and merged WeatherBench dailymeans dataset")
ds_msl = ds_all.sel(time=ds_all.time.dt.month.isin([12, 1, 2]))

atm_entry = atm_entries[1]
WEATHERBENCH_DAILY_PATH = atm_entry.path
print(WEATHERBENCH_DAILY_PATH)
print("[INFO] Merging all years into a single dataset...")
files = sorted(WEATHERBENCH_DAILY_PATH.glob("*.nc"))
ds_all = xr.open_mfdataset(
    files,
    combine="by_coords",
    chunks={"time": 50, "latitude": 50, "longitude": 50},
).load()
print("[INFO] Loaded and merged WeatherBench dailymeans dataset")
ds_z500 = ds_all.sel(time=ds_all.time.dt.month.isin([12, 1, 2]))

atm_entry = atm_entries[2]
WEATHERBENCH_DAILY_PATH = atm_entry.path
print(WEATHERBENCH_DAILY_PATH)
print("[INFO] Merging all years into a single dataset...")
files = sorted(WEATHERBENCH_DAILY_PATH.glob("*.nc"))
ds_all = xr.open_mfdataset(
    files,
    combine="by_coords",
    chunks={"time": 50, "latitude": 50, "longitude": 50},
).load()
print("[INFO] Loaded and merged WeatherBench dailymeans dataset")
ds_ws = ds_all.sel(time=ds_all.time.dt.month.isin([12, 1, 2]))

# Normalize longitude convention
ds_ws = ds_ws.assign_coords(longitude=((ds_ws.longitude + 180) % 360) - 180).sortby("longitude")
ds_msl = ds_msl.assign_coords(longitude=((ds_msl.longitude + 180) % 360) - 180).sortby("longitude")
ds_z500 = ds_z500.assign_coords(longitude=((ds_z500.longitude + 180) % 360) - 180).sortby("longitude")

ds_all = xr.merge([ds_msl, ds_z500, ds_ws], compat="override")

calculator = RegionalMeanCalculator()
energy_abs_vals = calculator.load_energy_values_from_prepared(
    path=energy_entry.path,
    energy_name=energy_entry.name,
    region_name=region_ziel,
)
print(energy_abs_vals)

# ---------Analyser-----------------
# Use original split only (no CV variants)
training_predictor = ds_all.sel(time=slice("1980", "2010"))
training_event = event.sel(time=slice("1980", "2010"))
test_predictor = ds_all.sel(time=slice("2011", "2022"))
test_event = event.sel(time=slice("2011", "2022"))

analyser = LaggedAnalyser(event=training_event, variables=training_predictor)
analyser.lag_variables(lags, offset_unit="days", mode="past")

atm_entry = atm_entries[0]
ATM_NAME = atm_entry.name
composites0 = xr.open_dataset(
    f"data//domino_output_coords_weatherbench/{energy_entry.name}/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{'_'.join(map(str, lags))}_composites_coords_deseasonalized_DJF_anom.nc"
)
lagged_means0 = xr.open_dataset(
    f"data//domino_output_coords_weatherbench/{energy_entry.name}/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{'_'.join(map(str, lags))}_lagged-means_coords_deseasonalized_DJF_anom.nc"
)

atm_entry = atm_entries[1]
ATM_NAME = atm_entry.name
composites1 = xr.open_dataset(
    f"data//domino_output_coords_weatherbench/{energy_entry.name}/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{'_'.join(map(str, lags))}_composites_coords_deseasonalized_DJF_anom.nc"
)
lagged_means1 = xr.open_dataset(
    f"data//domino_output_coords_weatherbench/{energy_entry.name}/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{'_'.join(map(str, lags))}_lagged-means_coords_deseasonalized_DJF_anom.nc"
)

atm_entry = atm_entries[2]
ATM_NAME = atm_entry.name
composites2 = xr.open_dataset(
    f"data//domino_output_coords_weatherbench/{energy_entry.name}/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{'_'.join(map(str, lags))}_composites_coords_deseasonalized_DJF_anom.nc"
)
lagged_means2 = xr.open_dataset(
    f"data//domino_output_coords_weatherbench/{energy_entry.name}/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{'_'.join(map(str, lags))}_lagged-means_coords_deseasonalized_DJF_anom.nc"
)

composites_all = xr.merge([composites0, composites1, composites2])
lagged_means_all = xr.merge([lagged_means0, lagged_means1, lagged_means2])

composites_all = composites_all.assign_coords(longitude=((composites_all.longitude + 180) % 360) - 180).sortby("longitude")
lagged_means_all = lagged_means_all.assign_coords(longitude=((lagged_means_all.longitude + 180) % 360) - 180).sortby("longitude")

analyser.composites = composites_all
analyser.lagged_means = lagged_means_all
analyser.composites_are_anomaly = True
analyser.composite_func = (agg.mean_ds, agg.cat_occ_ds)

print("[INFO] Loading significance mask for composites...")
atm_entry = atm_entries[0]
ATM_NAME = atm_entry.name
S0 = xr.open_dataset(
    f"data//domino_output_coords_weatherbench/{energy_entry.name}/100/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{'_'.join(map(str, lags))}_bootstrap-composites_deseasonalized_DJF_anom.nc"
)
atm_entry = atm_entries[1]
ATM_NAME = atm_entry.name
S1 = xr.open_dataset(
    f"data//domino_output_coords_weatherbench/{energy_entry.name}/100/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{'_'.join(map(str, lags))}_bootstrap-composites_deseasonalized_DJF_anom.nc"
)
atm_entry = atm_entries[2]
ATM_NAME = atm_entry.name
S2 = xr.open_dataset(
    f"data//domino_output_coords_weatherbench/{energy_entry.name}/100/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{'_'.join(map(str, lags))}_bootstrap-composites_deseasonalized_DJF_anom.nc"
)
S = xr.merge([S0, S1, S2])
S = S.assign_coords(longitude=((S.longitude + 180) % 360) - 180).sortby("longitude")
analyser.composite_sigs = S
masked_composite = S

if maskierung == "_noMask":
    masked_composite = composites_all
if maskierung == "":
    PF = PatternFilter(analyser=analyser)
    thresh_dict = dict(
        relative_humidity=0.05,
        total_column_water_vapour=1.0,
        **{"2m_temperature": 1.0},
        mean_sea_level_pressure=250,
        geopotential=300,
        wind_speed=2.5,
    )

    def abs_amplitude_over_threshold(val_ds, thresh_dict):
        return abs(val_ds) > xr.Dataset(thresh_dict)

    PF.apply_value_mask(abs_amplitude_over_threshold, thresh_dict, mode="intersection")
    PF.apply_convolution(3, ["latitude", "longitude"], mode="intersection")
    mask1 = PF.get_mask()

end = time.time()
print(f"[INFO] Data and analyser prepared in {end - start:.2f} seconds.")

#-------------Composites for index generation--------------------------------
if maskierung == "":
    masked_ds = masked_composite.where(mask1)
else:
    masked_ds = masked_composite
plot_ds = masked_ds.sel(index_val=1)

variables = list(plot_ds.data_vars)
print(f"[INFO] Composite variables used for indices: {variables}")

#-------------Index Generator --------------------------------
IG = IndexGenerator()
var_names = [atm_entries[0].name, atm_entries[1].name, atm_entries[2].name]
lags_index = [-4, -3, -2, -1]

index_vals_train = []
index_vals_test = []

for var_name in var_names:
    for lag_id in lags_index:
        print(f"Generating index for {var_name} at lag {lag_id}...")
        index_slices = [dict(index_val=1, lag=lag_id)]
        index_params_path = Path(
            f"data/index_parameters/{test_version}/{region_ziel}/{energy_entry.name}/{extreme_event}/{energy_entry.name}-{extreme_event}-{var_name}-index-{lag_id}-training{maskierung}{test_version_region}.nc"
        )
        params = xr.open_dataset(index_params_path)
        idx_mean = params[f"{var_name}_index_val1_lag{lag_id}"].sel(param="mean").item()
        idx_std = params[f"{var_name}_index_val1_lag{lag_id}"].sel(param="std").item()

        if idx_mean == 0 and idx_std == 0:
            print(f"[INFO] Skipping {var_name} at lag {lag_id} because mean and std are both 0.")
            continue

        index_name = f"{var_name}_index_val1_lag{lag_id}"
        indices = IG.generate(
            masked_ds[[f"{var_name}"]],
            training_predictor[[f"{var_name}"]],
            dim="time",
            slices=index_slices,
            ix_means={index_name: idx_mean},
            ix_stds={index_name: idx_std},
        )
        index_vals_train.append(indices)

        indices_test = IG.generate(
            masked_ds[[f"{var_name}"]],
            test_predictor[[f"{var_name}"]],
            dim="time",
            slices=index_slices,
            ix_means={index_name: idx_mean},
            ix_stds={index_name: idx_std},
        )
        index_vals_test.append(indices_test)
# print(index_vals_test)
X = xr.concat([ds.to_array().squeeze() for ds in index_vals_train], dim="feature").transpose("time", "feature")
feature_names = X.coords["variable"].values
X = X.drop_vars("variable")
X = X.assign_coords(feature=("feature", feature_names))

X_test = xr.concat([ds.to_array().squeeze() for ds in index_vals_test], dim="feature").transpose("time", "feature")
feature_names = X_test.coords["variable"].values
X_test = X_test.drop_vars("variable")
X_test = X_test.assign_coords(feature=("feature", feature_names))

Y = training_event
Y_test = test_event
Y_test["time"] = Y_test.time.astype("datetime64[ns]")
Y["time"] = Y.time.astype("datetime64[ns]")
X, Y = xr.align(X, Y, join="inner")

nan_count = X.isnull().sum(dim="time")
valid_features = nan_count < X.sizes["time"]

X_reduced = X.sel(feature=valid_features)
X_test_reduced = X_test.sel(feature=valid_features)
print(f"[INFO] Reduced features with non-nan data: {valid_features.feature.values}")

X_test_reduced, Y_test = xr.align(X_test_reduced, Y_test, join="inner")

# -----------------------------------------------------
# ROC-Kurven fuer alle Lags in einem Plot
# -----------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

for lag in lags:
    model_path = Path(
        f"fig/logRes_new/prepPangu/{test_version}/{region_ziel}/{energy_entry.name}/{extreme_event}/{energy_entry.name}_{extreme_event}_logistic_model_lag{lag}{maskierung}_prepPangu{test_version_region}.pkl"
    )

    if not model_path.exists():
        print(f"[INFO] Modell fuer Lag {lag} nicht gefunden. Ueberspringe.")
        continue

    saved = joblib.load(model_path)
    model = saved["model"]
    selected_features = saved["features"]

    X_test_reduced_features = X_test_reduced.sel(feature=selected_features)

    Y_prob = model.predict_proba(X_test_reduced_features)[:, 1]
    auc = roc_auc_score(Y_test, Y_prob)
    fpr, tpr, _ = roc_curve(Y_test, Y_prob)

    label = f"Lag {lag} (AUC = {auc:.2f})"
    print(f"[RESULTS] {label}")

    ax.plot(fpr, tpr, lw=2, label=label)

ax.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves Across Lags")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

lag_name = "_".join(map(str, lags))
out_path = Path(
    f"fig/logRes_new/prepPangu/{test_version}/{region_ziel}/{energy_entry.name}/{extreme_event}/{energy_entry.name}_{extreme_event}_ROC_alllags_{lag_name}{maskierung}{test_version_region}.png"
)
out_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_path, dpi=300)
print(f"[INFO] Saved ROC plot to: {out_path}")

# -----------------------------------------------------
# ROC-Kurven pro Indexvariable aus index_vals_test
# (ein Plot je Variable mit mehreren Lag-Kurven)
# -----------------------------------------------------
for var_name in var_names:
    fig_var, ax_var = plt.subplots(figsize=(10, 10))
    plotted_any = False

    for lag_id in lags_index:
        feature_name = f"{var_name}_index_val1_lag{lag_id}"
        if feature_name not in X_test_reduced.feature.values:
            print(f"[INFO] Feature {feature_name} nicht gefunden. Ueberspringe.")
            continue

        score_da = X_test_reduced.sel(feature=feature_name)
        score_da, y_da = xr.align(score_da, Y_test, join="inner")

        valid_mask = score_da.notnull()
        score_vals = score_da.where(valid_mask, drop=True).values
        y_vals = y_da.where(valid_mask, drop=True).values

        if len(y_vals) == 0 or len(set(y_vals.tolist())) < 2:
            print(f"[INFO] Zu wenige gueltige Klassen fuer {feature_name}. Ueberspringe.")
            continue

        auc_single = roc_auc_score(y_vals, score_vals)
        fpr_single, tpr_single, _ = roc_curve(y_vals, score_vals)
        ax_var.plot(fpr_single, tpr_single, lw=2, label=f"Lag {lag_id} (AUC = {auc_single:.2f})")
        plotted_any = True

    if not plotted_any:
        plt.close(fig_var)
        print(f"[INFO] Keine ROC-Kurven fuer {var_name} erstellt.")
        continue

    ax_var.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
    ax_var.set_xlabel("False Positive Rate")
    ax_var.set_ylabel("True Positive Rate")
    ax_var.set_title(f"ROC Curves by Lag for {var_name}")
    ax_var.grid(alpha=0.3)
    ax_var.legend()
    fig_var.tight_layout()

    out_var_path = Path(
        f"fig/logRes_new/prepPangu/{test_version}/{region_ziel}/{energy_entry.name}/{extreme_event}/{energy_entry.name}_{extreme_event}_ROC_indexvals_{var_name}_lags_{'_'.join(map(str, lags_index))}{maskierung}{test_version_region}.png"
    )
    out_var_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_var_path, dpi=300)
    print(f"[INFO] Saved per-variable index ROC plot to: {out_var_path}")
