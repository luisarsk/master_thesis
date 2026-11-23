from pathlib import Path
import datetime as dt

import geopandas as gpd
import sklearn
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import zarr
import pandas as pd
import time
import joblib

from data_registry import Bibliography
from regionalmeancalculator import RegionalMeanCalculator
from domino.core import LaggedAnalyser, PatternFilter, IndexGenerator
from domino.prediction import PredictionTest
from domino import agg
from coords_workflow_utils import PreparedCoordsLoader, NUTSMaskTools
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve

import cartopy.crs as ccrs
import cmocean.cm as cm

# -----CONFIGURATION----------
bib = Bibliography(Path("data"))
# bib.add("energy", "WON", "daily", "subcountry", "prepared/WON_daily_subcountry_prepared_DJF.csv")
bib.add("energy", "SPV", "daily", "subcountry", "prepared/SPV_daily_subcountry_prepared_DJF.csv")
bib.add("atmosphere", "mean_sea_level_pressure", "daily", "subcountry", "ERA5/prepared/DJF_anom/mean_sea_level_pressure")
# bib.add("atmosphere", "geopotential", "daily", "subcountry", "ERA5/prepared/DJF_anom/geopotential")
# bib.add("atmosphere", "v_component_of_wind", "daily", "subcountry", "ERA5/prepared/DJF_anom/v_component_of_wind")
# bib.add("atmosphere", "wind_speed", "daily", "subcountry", "ERA5/prepared/DJF_anom/wind_speed")
# bib.add('atmosphere', 'total_column_water_vapour', 'daily', 'global', 'ERA5/prepared/DJF_anom/total_column_water_vapour')
bib.add('atmosphere', '2m_temperature', 'daily', 'global', 'ERA5/prepared/DJF_anom/2m_temperature')
bib.add('atmosphere', 'relative_humidity', 'daily', 'global', 'ERA5/prepared/DJF_anom/relative_humidity')

# -----VARIABLES-----------
erstelldatum = dt.date.today()
lags = [-4,-3,-2,-1,0] # [-6, -5, -4, -3, -2, -1, 0]
region_ziel = "Norddeutschland"  # region der ziel variablen
region_prec = "EUROPE"  # welche Daten werden als precursor verwendet
timeperiod = "1979-2025"  # zeitraum der analysierten daten
quant = 0.9 # welches quantile => def extreme event
extreme_event = "h"  # l=low 0.1, h=high 0.9
plotting = False  # sollen die composites geplottet werden?
ks_roc_plots = False # sollen die KS und ROC Plots erstellt werden? für die einzelnen Indizes, ohne logRes
model_erstellen = True  # soll das logistische Modell erstellt werden?
plot_zeitreihen = True
cv_version = '' # '_cv2' 3, 4 for cross-validation version, '' for original version (Train 1980-2010, Test 2011-2022)
maskierung = '_noMask' # '_noMask' for no mask, '' for mask with pattern filter
test_version = '' # anpassen im pfad zu index_parameter, model save_dir
# test_version_region = '_TenneT-with-Amprion'

energy_entries = bib.get_by_category("energy")
atm_entries = bib.get_by_category("atmosphere")

## -----LOAD DATA----------
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

# lade einzelne atm variablen als xr.Dataset
atm_entry = atm_entries[1]
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
ds_z500 = ds_all

# lade einzelne atm variablen als xr.Dataset
atm_entry = atm_entries[2]
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
ds_ws = ds_all

# rechne longitudes bei allen um
ds_ws = ds_ws.assign_coords(
    longitude=((ds_ws.longitude + 180) % 360) - 180
).sortby("longitude")
ds_msl = ds_msl.assign_coords(
    longitude=((ds_msl.longitude + 180) % 360) - 180
).sortby("longitude")
ds_z500 = ds_z500.assign_coords(
    longitude=((ds_z500.longitude + 180) % 360) - 180
).sortby("longitude")

ds_all = xr.merge([ds_msl, ds_z500, ds_ws], compat="override")

calculator = RegionalMeanCalculator()

energy_abs_vals = calculator.load_energy_values_from_prepared(
    path=energy_entry.path,
    energy_name=energy_entry.name,
    region_name=region_ziel
)
print(energy_abs_vals)
# ---------Analyser-----------------
# split into training and test data
if cv_version=='':
    training_predictor=ds_all.sel(time=slice('1980','2010'))
    training_event=event.sel(time=slice('1980','2010'))
    test_predictor=ds_all.sel(time=slice('2011','2022'))
    test_event=event.sel(time=slice('2011','2022'))
elif cv_version=='_cv2':
    training_predictor=ds_all.sel(time=slice('1992','2022'))
    training_event=event.sel(time=slice('1992','2022'))
    test_predictor=ds_all.sel(time=slice('1980','1991'))
    test_event=event.sel(time=slice('1980','1991'))
elif cv_version=='_cv3':
    test_predictor = ds_all.sel(time=slice("1990", "2001"))
    test_event = event.sel(time=slice("1990", "2001"))
    training_predictor = xr.concat([ds_all.sel(time=slice("1979", "1989")), ds_all.sel(time=slice("2002", "2022"))],dim="time")
    training_event = xr.concat([event.sel(time=slice("1979", "1989")), event.sel(time=slice("2002", "2022"))], dim="time")
elif cv_version=='_cv4':
    test_predictor = ds_all.sel(time=slice("2000", "2011"))
    test_event = event.sel(time=slice("2000", "2011"))
    training_predictor = xr.concat([ds_all.sel(time=slice("1979", "1999")), ds_all.sel(time=slice("2012", "2022"))],dim="time")
    training_event = xr.concat([event.sel(time=slice("1979", "1999")), event.sel(time=slice("2012", "2022"))], dim="time")

analyser = LaggedAnalyser(event=training_event, variables=training_predictor)
analyser.lag_variables(lags, offset_unit='days', mode='past')

atm_entry = atm_entries[0]
ATM_NAME = atm_entry.name
composites0 = xr.open_dataset(f'data//domino_output_coords_weatherbench/{energy_entry.name}/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{"_".join(map(str, lags))}_composites_coords_deseasonalized_DJF_anom.nc')#analyser.compute_composites(as_anomaly=True)
lagged_means0 = xr.open_dataset(f'data//domino_output_coords_weatherbench/{energy_entry.name}/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{"_".join(map(str, lags))}_lagged-means_coords_deseasonalized_DJF_anom.nc')
atm_entry = atm_entries[1]
ATM_NAME = atm_entry.name
composites1 = xr.open_dataset(f'data//domino_output_coords_weatherbench/{energy_entry.name}/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{"_".join(map(str, lags))}_composites_coords_deseasonalized_DJF_anom.nc')#analyser.compute_composites(as_anomaly=True)
lagged_means1 = xr.open_dataset(f'data//domino_output_coords_weatherbench/{energy_entry.name}/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{"_".join(map(str, lags))}_lagged-means_coords_deseasonalized_DJF_anom.nc')
atm_entry = atm_entries[2]
ATM_NAME = atm_entry.name
composites2 = xr.open_dataset(f'data//domino_output_coords_weatherbench/{energy_entry.name}/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{"_".join(map(str, lags))}_composites_coords_deseasonalized_DJF_anom.nc')#analyser.compute_composites(as_anomaly=True)
lagged_means2 = xr.open_dataset(f'data//domino_output_coords_weatherbench/{energy_entry.name}/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{"_".join(map(str, lags))}_lagged-means_coords_deseasonalized_DJF_anom.nc')

composites_all = xr.merge([composites0, composites1, composites2])
lagged_means_all = xr.merge([lagged_means0, lagged_means1, lagged_means2])

composites_all = composites_all.assign_coords(
    longitude=((composites_all.longitude + 180) % 360) - 180
).sortby("longitude")
lagged_means_all = lagged_means_all.assign_coords(
    longitude=((lagged_means_all.longitude + 180) % 360) - 180
).sortby("longitude")

analyser.composites = composites_all
analyser.lagged_means = lagged_means_all
analyser.composites_are_anomaly = True
analyser.composite_func = (agg.mean_ds, agg.cat_occ_ds)

print("[INFO] Loading significance mask for composites...")
atm_entry = atm_entries[0]
ATM_NAME = atm_entry.name
S0 = xr.open_dataset(f'data//domino_output_coords_weatherbench/{energy_entry.name}/100/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{"_".join(map(str, lags))}_bootstrap-composites_deseasonalized_DJF_anom.nc')
atm_entry = atm_entries[1]
ATM_NAME = atm_entry.name
S1 = xr.open_dataset(f'data//domino_output_coords_weatherbench/{energy_entry.name}/100/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{"_".join(map(str, lags))}_bootstrap-composites_deseasonalized_DJF_anom.nc')
atm_entry = atm_entries[2]
ATM_NAME = atm_entry.name
S2 = xr.open_dataset(f'data//domino_output_coords_weatherbench/{energy_entry.name}/100/{energy_entry.name}-{extreme_event}_{region_ziel}_{ATM_NAME}_{region_prec}_1979_lags_{"_".join(map(str, lags))}_bootstrap-composites_deseasonalized_DJF_anom.nc')
S = xr.merge([S0, S1, S2])
S = S.assign_coords(
    longitude=((S.longitude + 180) % 360) - 180
).sortby("longitude")
analyser.composite_sigs = S
masked_composite=S#composites_all.where(S==1)

if maskierung == '_noMask':
    masked_composite = composites_all
if maskierung == '':
    PF = PatternFilter(analyser=analyser)
    thresh_dict=dict(relative_humidity=0.05, total_column_water_vapour=1.0, **{'2m_temperature': 1.}, mean_sea_level_pressure=250, geopotential=300, wind_speed=2.5, v_component_of_wind=2.5)

    def abs_amplitude_over_threshold(val_ds,thresh_dict):
        return abs(val_ds)>xr.Dataset(thresh_dict)

    PF.apply_value_mask(abs_amplitude_over_threshold,thresh_dict,mode='intersection')
    PF.apply_convolution(3,['latitude','longitude'],mode='intersection')
    mask1=PF.get_mask() #retrieve the new mask for plotting later.
    print("[INFO] Loading significance mask for lagged means...")

#-------------Composites Plotten, die in Index verwendet werden sollen--------------------------------
if maskierung == '':
    masked_ds = masked_composite.where(mask1)
else:
    masked_ds = masked_composite
plot_ds = masked_ds.sel(index_val=1)

# alle Variablen
variables = list(plot_ds.data_vars)
print(f"[INFO] Plotting composite patterns for variables: {variables}")

# über Variablen einzeln plotten
if plotting == True:
    for var in variables:

        lags = plot_ds.lag.values

        fig, axes = plt.subplots(
            1,
            len(lags),
            figsize=(4.5 * len(lags), 4),
            subplot_kw={"projection": ccrs.PlateCarree()}
        )

        axes = np.atleast_1d(axes)

        # gleiche Farbskala für alle Lags dieser Variable
        data_all = plot_ds[var]
        vabs = np.nanmax(np.abs(data_all.values))

        for ax, lag in zip(axes, lags):

            data = plot_ds[var].sel(lag=lag)

            im = data.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap="RdBu_r",
                vmin=-vabs,
                vmax=vabs,
                add_colorbar=False
            )

            ax.coastlines()
            ax.set_title(f"Lag {lag}")

        # Platz für Colorbar unten reservieren
        fig.subplots_adjust(bottom=0.25, top=0.85, wspace=0.05)

        # eigene Position der Colorbar
        cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.025])
        cbar = fig.colorbar(
            im,
            cax=cbar_ax,
            orientation="horizontal"
        )

        cbar.set_label(var)

        plt.suptitle(
            f"{var} Composite Pattern for {energy_entry.name} ({extreme_event}) in {region_ziel}",
            fontsize=16
        )

        plt.tight_layout(rect=[0, 0.12, 1, 0.95])
        save_dir = Path(
            f"fig/logRes/{energy_entry.name}/{extreme_event}/Composites_patterns_bootstrap_filter"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        plt.savefig(
            save_dir / f"{energy_entry.name}_{extreme_event}_{region_ziel}_{var}_composite_pattern{maskierung}.png",
            dpi=300
        )
        plt.close(fig)

#-------------Index Generator --------------------------------

IG=IndexGenerator()

#--------Index Zeitreihen für Train und Test Set erstellen --------
# erstelle zeitreihen von indexwerten
var_names =[atm_entries[0].name, atm_entries[1].name, atm_entries[2].name]

index_vals_train = []
index_vals_test = []
lags = [-4,-3,-2,-1]

for var_name in var_names:
    for lag_id in lags:
        print(f"Generating index for {var_name} at lag {lag_id}...")
        index_slices=[dict(index_val=1,lag=lag_id)]
        index_params_path = Path(f"data/index_parameters/{region_ziel}/{energy_entry.name}/{extreme_event}/{energy_entry.name}-{extreme_event}-{var_name}-index-{lag_id}-training{cv_version}{maskierung}.nc")
        params = xr.open_dataset(index_params_path)
        idx_mean = params[f"{var_name}_index_val1_lag{lag_id}"].sel(param="mean").item()
        idx_std = params[f"{var_name}_index_val1_lag{lag_id}"].sel(param="std").item()
        print(idx_mean, idx_std)

        if idx_mean == 0 and idx_std == 0:
            print(f"[INFO] Skipping {var_name} at lag {lag_id} because mean and std are both 0.")
            continue

        index_name = f"{var_name}_index_val1_lag{lag_id}"
        indices=IG.generate(masked_ds[[f"{var_name}"]],training_predictor[[f"{var_name}"]],dim='time',slices=index_slices, ix_means={index_name: idx_mean}, ix_stds={index_name: idx_std})
        index_vals_train.append(indices)

        indices_test=IG.generate(masked_ds[[f"{var_name}"]],test_predictor[[f"{var_name}"]],dim='time',slices=index_slices, ix_means={index_name: idx_mean}, ix_stds={index_name: idx_std})
        index_vals_test.append(indices_test)

X = xr.concat(
    [ds.to_array().squeeze() for ds in index_vals_train],
    dim="feature"
).transpose("time", "feature")

feature_names = X.coords["variable"].values

X = X.drop_vars("variable")

X = X.assign_coords(
    feature=("feature", feature_names)
)

X_test = xr.concat(
    [ds.to_array().squeeze() for ds in index_vals_test],
    dim="feature"
).transpose("time", "feature")

feature_names = X_test.coords["variable"].values

X_test = X_test.drop_vars("variable")

X_test = X_test.assign_coords(
    feature=("feature", feature_names)
)

#-------------------------------------------------------------
#--------X und Y erstellen für Train und Test Set --------
#-------------------------------------------------------------

Y = training_event
Y_test = test_event
Y_test["time"] = Y_test.time.astype("datetime64[ns]")
Y["time"] = Y.time.astype("datetime64[ns]")
X, Y = xr.align(X, Y, join="inner")
nan_count = X.isnull().sum(dim="time")
print(nan_count)
valid_features = nan_count < X.sizes["time"]

X_reduced = X.sel(feature=valid_features)
X_test_reduced = X_test.sel(feature=valid_features)
print(f'[INFO] Reduced features with non nan data: {valid_features.feature.values}')

if model_erstellen ==True:
    #-------------------------------------------------------------
    #--------Model trainieren mit alles n Variablen und Lags --------
    #-------------------------------------------------------------

    model = LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        max_iter=1000
    )

    model.fit(
        X_reduced,
        Y
    )

    Y_prob = model.predict_proba(X_test_reduced)[:, 1]
    auc = roc_auc_score(
        Y_test,
        Y_prob
    )
    print("[RESULTS] LogRes all vars. For Testperiod AUC = ", auc)

    weights = pd.Series(
        model.coef_[0],
        index=X_reduced.feature.values
    )

    weights = weights.sort_values(ascending=False)
    print(f'[INFO] Feature weights: {weights}')

    # plotte zugehörige ROC Kurve
    fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
    # AUC berechnen
    auc = roc_auc_score(Y_test, Y_prob)

    # Plot
    plt.figure(figsize=(6,6))

    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc:.3f})")

    # Zufallsklassifikator
    plt.plot([0,1], [0,1], "--", color="gray", label="Random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: AUC = {:.3f}".format(auc))
    plt.legend()
    plt.grid(alpha=0.3)
    save_dir = Path(f"fig/logRes_new/prepPangu/{region_ziel}/{energy_entry.name}/{extreme_event}")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{energy_entry.name}_{extreme_event}_ROC_curve_allVars{cv_version}{maskierung}_prepPangu.png",dpi=300)

    # model abspeichern
    model_path = save_dir / f"{energy_entry.name}_{extreme_event}_logistic_model_allVars{cv_version}{maskierung}_prepPangu.pkl"
    joblib.dump(
        {
            "model": model,
            "features": X_reduced.feature.values,
        },
        model_path,
    )

    print(f"Modell gespeichert unter:\n{model_path}")

    #-------------------------------------------------------------
    #--------Model trainieren mit ausgewählten Variablen und Lags: Forward Predictor Selection --------
    #-------------------------------------------------------------
    model = LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        max_iter=1000,
        random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    sfs = SequentialFeatureSelector(
        estimator=model,
        n_features_to_select='auto',
        direction="forward",
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )

    # Feature-Maske
    sfs.fit(X_reduced, Y)
    mask = sfs.get_support()

    # Trainings- und Testdaten reduzieren
    X_train_sel = X_reduced.sel(feature=mask)      # falls X_reduced ein xarray/DataArray oder NumPy-Array ist
    X_test_sel = X_test_reduced.sel(feature=mask)

    # Modell neu trainieren
    model.fit(X_train_sel, Y)

    # Wahrscheinlichkeiten berechnen
    Y_prob = model.predict_proba(X_test_sel)[:, 1]

    fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
    auc = roc_auc_score(Y_test, Y_prob)
    print("ROC-AUC:", auc)

    # Plot
    plt.figure(figsize=(6,6))

    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc:.3f})")

    # Zufallsklassifikator
    plt.plot([0,1], [0,1], "--", color="gray", label="Random classifier")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: AUC = {:.3f}".format(auc))
    plt.legend()
    plt.grid(alpha=0.3)
    save_dir = Path(f"fig/logRes_new/prepPangu/{region_ziel}/{energy_entry.name}/{extreme_event}")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"{energy_entry.name}_{extreme_event}_ROC_curve_forward-predictor-selection{cv_version}{maskierung}_prepPangu.png",dpi=300)

    # model abspeichern
    model_path = save_dir / f"{energy_entry.name}_{extreme_event}_logistic_model_forward-predictor-selection{cv_version}{maskierung}_prepPangu.pkl"
    joblib.dump(
        {
            "model": model,
            "features": X_train_sel.feature.values,
        },
        model_path,
    )

    print(f"Modell gespeichert unter:\n{model_path}")

    #-------------------------------------------------------------
    # -------Model trainieren mit jeweils dem gleichen lag für alle Variablen ----------
    #-------------------------------------------------------------
    for lag in lags:
        print(f"[INFO] Training model with lag {lag} for all variables...")
        X_train_lag = X_reduced.sel(feature=X_reduced.feature.str.endswith(f"lag{lag}"))
        X_test_lag = X_test_reduced.sel(feature=X_test_reduced.feature.str.endswith(f"lag{lag}"))
        # Lag überspringen, falls keine Features vorhanden sind
        if X_train_lag.sizes["feature"] == 0 or X_test_lag.sizes["feature"] == 0:
            print(f"[INFO] Keine Features für Lag {lag} vorhanden. Überspringe diesen Lag.")
            continue

        model.fit(X_train_lag, Y)

        Y_prob = model.predict_proba(X_test_lag)[:, 1]
        auc = roc_auc_score(Y_test, Y_prob)
        print(f"[RESULTS] LogRes with lag {lag}. For Testperiod AUC = {auc}")

        # ROC-Kurve plotten
        fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc:.3f})")
        plt.plot([0,1], [0,1], "--", color="gray", label="Random classifier")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: AUC = {auc:.3f} (Lag {lag})")
        plt.legend()
        plt.grid(alpha=0.3)
        
        save_dir = Path(f"fig/logRes_new/prepPangu/{region_ziel}/{energy_entry.name}/{extreme_event}")
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / f"{energy_entry.name}_{extreme_event}_ROC_curve_lag{lag}{cv_version}{maskierung}_prepPangu.png", dpi=300)
        plt.close()
        # model abspeichern
        model_path = save_dir / f"{energy_entry.name}_{extreme_event}_logistic_model_lag{lag}{cv_version}{maskierung}_prepPangu.pkl"
        joblib.dump(
        {
            "model": model,
            "features": X_train_lag.feature.values,
        },
        model_path,
    )
        print(f"Modell gespeichert unter:\n{model_path}")

##########--------------------------------------------------------------
#----------- Model Zeitreihen für alle Lags und Variablen erstellen und abspeichern ----------------
##########--------------------------------------------------------------
if plot_zeitreihen == True and cv_version=='':
    for lag in lags:
        # model laden
        model_path = Path(f"fig/logRes_new/prepPangu/{region_ziel}/{energy_entry.name}/{extreme_event}/{energy_entry.name}_{extreme_event}_logistic_model_lag{lag}{cv_version}{maskierung}_prepPangu.pkl")
        if not model_path.exists():
            print(f"[INFO] Modell für Lag {lag} nicht gefunden. Überspringe.")
            continue
        saved = joblib.load(model_path)
        model = saved["model"]
        selected_features = saved["features"]

        X_test_reduced_features = X_test_reduced.sel(feature=selected_features)

        # Vorhersagen machen für X_test_reduced mit dem geladenen Modell
        Y_prob = model.predict_proba(X_test_reduced_features)[:, 1]
        auc = roc_auc_score(
            Y_test,
            Y_prob
        )
        print(f"[RESULTS] LogRes with lag {lag}. For Testperiod AUC = {auc}")
        index_prediction = model.predict(
        X_test_reduced_features)
        index_prediction_xr = xr.DataArray(
            index_prediction,
            coords={"time": X_test_reduced_features.time},
            dims=["time"],
            name="prediction"
        )

        SPV_index = xr.DataArray(
            Y_prob,
            coords={"time": X_test_reduced_features.time},
            dims="time",
            name="SPV_probability"
        )

        years = np.unique(X_test_reduced.time.dt.year.values)
        print(years)
        for year in years:
            start = f"{year}-12-01"
            end   = f"{year+1}-02-28"

            # Zeitraum auswählen
            prob = SPV_index.sel(time=slice(start, end))
            event = Y_test.sel(time=slice(start, end))
            prediction = index_prediction_xr.sel(time=slice(start, end))
            features = X_test_reduced_features.sel(time=slice(start, end))
            energy = energy_abs_vals.sel(time=slice(start, end))


            fig, ax = plt.subplots(
                3, 1,
                figsize=(15, 11),
                sharex=True,
                gridspec_kw={"height_ratios": [2, 3, 2]}
            )


            # =================================================
            # 1. Plot: Modelloutput
            # =================================================

            ax1 = ax[0]

            ax1.plot(
                prob.time,
                prob,
                color="tab:blue",
                lw=2,
                label=f"{energy_entry.name} probability"
            )

            ax1.set_ylabel("Probability")
            ax1.set_ylim(0, 1)
            ax1.grid(alpha=0.3)


            ax2 = ax1.twinx()

            # Beobachtete Events
            ax2.bar(
                event.time.values,
                event.values,
                width=1.0,
                color="tab:red",
                alpha=0.5,
                label=f"Observed {energy_entry.name} event"
            )

            # Modellprediction als grüne Marker
            pred_times = prediction.time.where(prediction == 1, drop=True)

            ax2.scatter(
                pred_times,
                np.ones(len(pred_times)),
                marker="v",
                s=70,
                color="tab:green",
                label="Prediction",
                zorder=5
            )

            ax2.set_ylim(0, 1.2)
            ax2.set_ylabel("Event")
            ax2.set_yticks([0, 1])


            # Legende rechts außerhalb
            lines, labels = ax1.get_legend_handles_labels()
            bars, labels2 = ax2.get_legend_handles_labels()

            ax1.legend(
                lines + bars,
                labels + labels2,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                frameon=True
            )


            # =================================================
            # 2. Plot: Eingangsindizes
            # =================================================

            ax3 = ax[1]

            for feature in features.feature.values:
                ax3.plot(
                    features.time,
                    features.sel(feature=feature),
                    lw=1,
                    label=str(feature)
                )

            ax3.axhline(
                0,
                color="black",
                linewidth=0.8
            )

            ax3.set_ylabel("Index value")
            ax3.grid(alpha=0.3)


            ax3.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                fontsize=8,
                frameon=True
            )


            # =================================================
            # 3. Plot: Energiezeitreihe
            # =================================================

            ax4 = ax[2]

            ax4.plot(
                energy.time,
                energy,
                color="tab:purple",
                lw=2,
                label=f"{energy_entry.name} capacity factor"
            )

            ax4.set_ylabel(f"{energy_entry.name} capacity factor")
            ax4.set_xlabel("Date")
            ax4.grid(alpha=0.3)

            ax4.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                frameon=True
            )


            # =================================================
            # Gesamttitel
            # =================================================

            fig.suptitle(
                f"Winter {year}/{year+1}",
                fontsize=14,
                x=0.5,
                y=0.995
            )


            # Platz für rechte Legenden lassen
            plt.tight_layout(rect=[0, 0, 0.82, 0.96])
            path2save = Path(f'fig/logRes_new/prepPangu/{test_version}/{region_ziel}/{energy_entry.name}/{extreme_event}')
            path2save.mkdir(parents=True, exist_ok=True)
            plt.savefig(path2save / f'{energy_entry.name}_{extreme_event}_logistic_regression_results_lag{lag}{cv_version}{maskierung}_{year}_prepPangu.png')
            # plt.show()
