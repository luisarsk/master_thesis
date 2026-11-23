import os
from pathlib import Path
import datetime as dt

import xarray as xr
import matplotlib.pyplot as plt
import joblib
import pandas as pd
import numpy as np

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
region_ziel = "Norddeutschland"
region_prec = "EUROPE"
timeperiod = "1979-2025"
quant = 0.9
extreme_event = "h"
maskierung = "_noMask"
cv_version = ''
model_forecast = 'pangu'

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
IG=IndexGenerator()
var_names = []
for entry in atm_entries:
    var_names.append(entry.name)
index_vals_train = []
index_vals_test = []
lags = [-4,-3,-2,-1]

pangu_start_year = 2019
pangu_end_year = 2022

pangu_data = []
clim_smooth_list = []
for entry in atm_entries:
    print(entry)
    pangu = xr.open_dataset(f"data/{model_forecast}/{entry.name}/{model_forecast}_{entry.name}_{pangu_start_year}-{pangu_end_year}_DJF.nc")
    pangu = pangu.assign_coords(
        longitude=((pangu.longitude + 180) % 360) - 180
    ).sortby("longitude")
    pangu_data.append(pangu)

    clim_smooth = xr.open_dataset(f'data/climsmooth/clim_smooth_{entry.name}.nc')
    clim_smooth = clim_smooth.assign_coords(
        longitude=((clim_smooth.longitude + 180) % 360) - 180
    ).sortby("longitude")
    clim_smooth_list.append(clim_smooth)

# Ergebnisse für alle var_ids speichern
pangu_daily_means_all = {}

for var_id in range(len(pangu_data)):

    pangu_sel = pangu_data[var_id]
    clim_sel = clim_smooth_list[var_id]

    # Nur die Initialzeiten um 00 UTC
    pangu_sel = pangu_sel.sel(time=pangu_sel.time.dt.hour == 0)

    pangu_daily_means = []

    for initial_time in pangu_sel.time:

        # Prediction time in richtiges Zeitformat umwandeln
        prediction_times = (
            pangu_sel.prediction_timedelta * np.timedelta64(1, "h")
            + initial_time
        )

        pangu_timestep_data = pangu_sel.sel(time=initial_time)

        pangu_timestep_data = pangu_timestep_data.assign_coords(
            prediction_time=prediction_times
        )

        # Tagesmittel für jeden Prediction-Tag
        daily_mean = pangu_timestep_data.groupby(
            "prediction_time.date"
        ).mean()

        daily_mean = (
            daily_mean
            .assign_coords(
                prediction_time=(
                    "date",
                    pd.to_datetime(daily_mean.date.values)
                )
            )
            .swap_dims({"date": "prediction_time"})
            .drop_vars("date")
        )

        # Klimatologie nach day of year auswählen
        doy = daily_mean["prediction_time"].dt.dayofyear
        clim_for_prediction = clim_sel.sel(dayofyear=doy)

        # Anomalie relativ zur Klimatologie
        daily_mean = daily_mean - clim_for_prediction

        pangu_daily_means.append(daily_mean)

    # Für diese var_id abspeichern
    pangu_daily_means_all[var_id] = pangu_daily_means

forecast_indices = {}
var_ids = [0, 1, 2]
lag_ids = [-4,-3,-2,-1]
for var_id in var_ids:

    print(f"Processing variable ID: {var_id}")

    var_name = atm_entries[var_id].name

    # Platz für alle Lags dieser Variable
    forecast_indices[var_id] = {}

    for lag_id in lag_ids:

        index_name = f"{var_name}_index_val1_lag{lag_id}"

        index_slices = [
            dict(index_val=1, lag=lag_id)
        ]

        index_params_path = Path(
            f"data/index_parameters/"
            f"{energy_entry.name}/"
            f"{extreme_event}/"
            f"{energy_entry.name}-{extreme_event}-"
            f"{var_name}-index-{lag_id}-training"
            f"{cv_version}{maskierung}.nc"
        )

        params = xr.open_dataset(index_params_path)

        idx_mean = params[index_name].sel(param="mean").item()
        idx_std = params[index_name].sel(param="std").item()

        forecast_indices_for_lag = []

        for pangu_daily_mean in pangu_daily_means_all[var_id]:

            forecast_index = IG.generate(
                masked_ds,
                pangu_daily_mean,
                dim="prediction_time",
                slices=index_slices,
                ix_means={index_name: idx_mean},
                ix_stds={index_name: idx_std},
            )

            forecast_indices_for_lag.append(forecast_index)

        forecast_indices[var_id][lag_id] = forecast_indices_for_lag

        params.close()

lag_ids = [-4, -3, -2, -1]

# Für ROC-Kurve:
roc_results = {}

for LAG_ID in lag_ids:

    print("\n" + "=" * 70)
    print(f"Bearbeite LAG_ID = {LAG_ID}")
    print("=" * 70)


    # -----------------------------------------------------
    # 1. Logistic-Regression-Modell laden
    # -----------------------------------------------------

    path_to_model = (
        f"fig/logRes_new/prepPangu/"
        f"{energy_entry.name}/{extreme_event}/"
        f"{energy_entry.name}_{extreme_event}_"
        f"logistic_model_lag{LAG_ID}{maskierung}_prepPangu.pkl"
    )

    model = joblib.load(path_to_model)

    features = model["features"]
    clf = model["model"]

    print("Features:", features)


    # -----------------------------------------------------
    # 2. Forecast-Datasets für diesen Lag auswählen
    # -----------------------------------------------------

    feature_lists = {
        features[0]: forecast_indices[0][LAG_ID],
        features[1]: forecast_indices[1][LAG_ID],
        features[2]: forecast_indices[2][LAG_ID],
    }


    # -----------------------------------------------------
    # 3. Initialzeitpunkt -> Dataset
    # -----------------------------------------------------

    by_time = {}

    for feature, ds_list in feature_lists.items():

        by_time[feature] = {
            pd.Timestamp(ds.time.values): ds[feature]
            for ds in ds_list
        }


    # -----------------------------------------------------
    # 4. Initialzeiten, die bei ALLEN Features vorhanden sind
    # -----------------------------------------------------

    common_times = sorted(
        set.intersection(
            *(set(d.keys()) for d in by_time.values())
        )
    )

    print(
        f"Gemeinsame Initialzeiten für Lag {LAG_ID}: "
        f"{len(common_times)}"
    )


    # -----------------------------------------------------
    # 5. Modell-Samples erzeugen
    # -----------------------------------------------------

    rows = []

    for init_time in common_times:

        # Die drei Features für denselben Initialzeitpunkt
        arrays = [
            by_time[feature][init_time]
            for feature in features
        ]

        # Über prediction_time ausrichten
        arrays = xr.align(
            *arrays,
            join="inner"
        )

        # Jede prediction_time ist ein Modell-Sample
        for i in range(len(arrays[0].prediction_time)):

            row = {
                "time": init_time,
                "prediction_time": pd.Timestamp(
                    arrays[0].prediction_time.values[i]
                ),
            }

            for feature, arr in zip(features, arrays):

                row[feature] = float(
                    arr.values[i]
                )

            rows.append(row)


    X_df = pd.DataFrame(rows)

    print(
        f"X_df für Lag {LAG_ID}:",
        X_df.shape
    )


    # -----------------------------------------------------
    # 6. Logistic Regression Prediction
    # -----------------------------------------------------

    X = X_df[features].to_numpy()

    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)

    X_df["prediction"] = y_pred
    X_df["probability"] = y_prob


    # -----------------------------------------------------
    # 7. Tatsächliche Events laden
    # -----------------------------------------------------

    event_sel = event.sel(
        time=slice(
            f"{pangu_start_year}-12-01",
            f"{pangu_end_year}-02-28"
        )
    )

    event_times = pd.to_datetime(
        event_sel.time.values
    )

    event_values = np.asarray(
        event_sel.values
    )


    # -----------------------------------------------------
    # 8. Absolute Event-Zeitreihe
    # -----------------------------------------------------

    event_abs = (
        PreparedCoordsLoader.load_energy_values_from_prepared(
            path=energy_entry.path,
            energy_name=energy_entry.name,
            region_ziel=region_ziel
        )
    )

    event_abs_sel = event_abs.sel(
        time=slice(
            f"{pangu_start_year}-12-01",
            f"{pangu_end_year}-02-28"
        )
    )

    event_abs_times = pd.to_datetime(
        event_abs_sel.time.values
    )

    event_abs_values = np.asarray(
        event_abs_sel.values
    )


    # =====================================================
    # 9. ROC-Daten für diesen Lag erzeugen
    # =====================================================

    roc_y_true = []
    roc_y_score = []

    for prediction_time, probability in zip(
        X_df["prediction_time"],
        X_df["probability"]
    ):

        prediction_time = pd.Timestamp(
            prediction_time
        )

        # Passendes binäres Event zum Prediction-Zeitpunkt
        try:

            event_value = (
                event_sel
                .sel(
                    time=prediction_time,
                    method="nearest"
                )
                .item()
            )

            roc_y_true.append(
                int(event_value)
            )

            roc_y_score.append(
                float(probability)
            )

        except Exception:

            # Falls kein passender Event-Zeitpunkt
            # gefunden werden kann
            continue


    roc_y_true = np.asarray(
        roc_y_true
    )

    roc_y_score = np.asarray(
        roc_y_score
    )


    # Nur berechnen, wenn beide Klassen vorhanden sind
    if len(np.unique(roc_y_true)) == 2:

        roc_auc = roc_auc_score(
            roc_y_true,
            roc_y_score
        )

        fpr, tpr, thresholds = roc_curve(
            roc_y_true,
            roc_y_score
        )

        roc_results[LAG_ID] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": roc_auc,
            "y_true": roc_y_true,
            "y_score": roc_y_score
        }

        print(
            f"Lag {LAG_ID}: "
            f"ROC-AUC = {roc_auc:.3f}"
        )

    else:

        print(
            f"Lag {LAG_ID}: "
            f"ROC nicht berechenbar – "
            f"nur eine Event-Klasse vorhanden."
        )


    # =====================================================
    # 10. ZEITREIHENPLOT
    # =====================================================

    pred_times = pd.to_datetime(
        X_df["prediction_time"]
    )

    pred_prob = X_df["probability"].values


    fig, ax = plt.subplots(
        figsize=(14, 6)
    )


    # -----------------------------------------------------
    # Alle Prediction-Zeitpunkte sammeln
    # -----------------------------------------------------

    all_times = pd.DatetimeIndex(
        pd.to_datetime(pred_times)
    ).unique().sort_values()


    # Position jedes Zeitpunktes auf künstlicher x-Achse
    time_to_x = {
        pd.Timestamp(time): i
        for i, time in enumerate(all_times)
    }


    # Künstliche x-Positionen
    pred_x = [
        time_to_x[pd.Timestamp(t)]
        for t in pd.to_datetime(pred_times)
    ]


    # -----------------------------------------------------
    # 11. Prediction Indices
    # -----------------------------------------------------

    for feature in features:

        ax.plot(
            pred_x,
            X_df[feature].values,
            label=feature,
            linewidth=1.5
        )

    ax.set_xlabel(
        "Prediction Time"
    )

    ax.set_ylabel(
        "Index Value"
    )


    # -----------------------------------------------------
    # 12. Zweite y-Achse
    # -----------------------------------------------------

    ax2 = ax.twinx()


    # Modell-Wahrscheinlichkeit
    ax2.plot(
        pred_x,
        pred_prob,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Predicted event probability"
    )


    # -----------------------------------------------------
    # 13. Absolute Events
    # -----------------------------------------------------

    event_x = []
    event_y = []

    for time, value in zip(
        event_abs_times,
        event_abs_values
    ):

        time = pd.Timestamp(time)

        if time in time_to_x:

            event_x.append(
                time_to_x[time]
            )

            event_y.append(
                value
            )


    ax2.bar(
        event_x,
        event_y,
        width=0.8,
        alpha=0.25,
        color="yellow",
        label="Event amplitude"
    )

    ax2.set_ylabel(
        "Event / Probability"
    )


    # -----------------------------------------------------
    # 14. Skalierung der rechten Achse
    # -----------------------------------------------------

    # Wahrscheinlichkeit liegt immer zwischen 0 und 1.
    #
    # Falls die absoluten Eventwerte ebenfalls <= 1 sind,
    # verwenden wir 0-1.
    #
    # Andernfalls wird die Achse automatisch skaliert.

    max_event_value = (
        np.nanmax(event_abs_values)
        if len(event_abs_values) > 0
        else 1
    )

    if max_event_value <= 1:

        ax2.set_ylim(
            0,
            1
        )

    else:

        ax2.set_ylim(
            0,
            max(1, max_event_value * 1.05)
        )


    # -----------------------------------------------------
    # 15. Beobachtete binäre Events
    # -----------------------------------------------------

    event_label_added = False

    for time, value in zip(
        event_times,
        event_values
    ):

        if value == 1:

            time = pd.Timestamp(time)

            if time in time_to_x:

                x = time_to_x[time]

                ax.axvspan(
                    x - 0.5,
                    x + 0.5,
                    alpha=0.15,
                    color="m",
                    label=(
                        "Observed event"
                        if not event_label_added
                        else None
                    )
                )

                event_label_added = True

    # ---------------------------------------------------------
    # X-ACHSE:
    # Prediction-Zeitreihen ohne Lücken hintereinander darstellen
    # ---------------------------------------------------------

    # all_times enthält ausschließlich tatsächlich vorhandene
    # Prediction-Zeitpunkte aus dem Datensatz.
    all_times = pd.DatetimeIndex(
        pd.to_datetime(pred_times)
    ).unique().sort_values()

    # Künstliche Position für jeden tatsächlich vorhandenen
    # Zeitpunkt
    time_to_x = {
        pd.Timestamp(t): i
        for i, t in enumerate(all_times)
    }

    # Positionen der Prediction-Zeitpunkte
    pred_x = [
        time_to_x[pd.Timestamp(t)]
        for t in pred_times
    ]


    # ---------------------------------------------------------
    # Wenige X-Ticks auswählen
    # ---------------------------------------------------------
    #
    # WICHTIG:
    # Die Tick-Positionen werden aus all_times genommen.
    # Dadurch stimmen Label und tatsächlicher Datensatz
    # garantiert überein.
    #
    # Wir zeigen maximal ca. 8-10 Beschriftungen.

    n_ticks_max = 10

    if len(all_times) <= n_ticks_max:

        tick_indices = np.arange(
            len(all_times)
        )

    else:

        # Gleichmäßig über den tatsächlich vorhandenen
        # Datenpunkten verteilte Indizes
        tick_indices = np.linspace(
            0,
            len(all_times) - 1,
            n_ticks_max,
            dtype=int
        )

        # doppelte Indizes vermeiden
        tick_indices = np.unique(
            tick_indices
        )


    tick_positions = tick_indices

    tick_times = [
        all_times[i]
        for i in tick_indices
    ]


    # ---------------------------------------------------------
    # Tick-Labels
    # ---------------------------------------------------------
    #
    # Die Beschriftung kommt DIREKT aus all_times.
    #
    # Bei vielen Jahren:
    #     Dez 2020
    #     Jan 2021
    #     ...
    #
    # Dadurch gibt es keine künstlich erzeugten Jahreslabels.

    tick_labels = [
        pd.Timestamp(t).strftime("%b %Y")
        for t in tick_times
    ]


    ax.set_xticks(
        tick_positions
    )

    ax.set_xticklabels(
        tick_labels,
        rotation=45,
        ha="right"
    )


    # ---------------------------------------------------------
    # Winterwechsel markieren
    # ---------------------------------------------------------
    #
    # Der Wechsel wird ebenfalls anhand der echten
    # prediction_time-Werte bestimmt.

    for i in range(
        1,
        len(all_times)
    ):

        previous = pd.Timestamp(
            all_times[i - 1]
        )

        current = pd.Timestamp(
            all_times[i]
        )

        previous_winter = (
            previous.year + 1
            if previous.month == 12
            else previous.year
        )

        current_winter = (
            current.year + 1
            if current.month == 12
            else current.year
        )

        if current_winter != previous_winter:

            ax.axvline(
                i - 0.5,
                linestyle="--",
                linewidth=0.8,
                alpha=0.5
            )


    # ---------------------------------------------------------
    # Achsenbereich
    # ---------------------------------------------------------

    if len(all_times) > 1:

        ax.set_xlim(
            -0.5,
            len(all_times) - 0.5
        )

    else:

        ax.set_xlim(
            -1,
            1
        )

    # -----------------------------------------------------
    # 18. Gemeinsame Legende
    # -----------------------------------------------------

    lines1, labels1 = (
        ax.get_legend_handles_labels()
    )

    lines2, labels2 = (
        ax2.get_legend_handles_labels()
    )

    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper left",
        bbox_to_anchor=(1.04, 1)
    )


    # -----------------------------------------------------
    # 19. Titel
    # -----------------------------------------------------

    ax.set_title(
        f"{energy_entry.name} {extreme_event} – "
        f"Prediction Indices, Event Probability and Events "
        f"for Logistic Regression Model lag {LAG_ID}"
    )


    # -----------------------------------------------------
    # 20. Grid
    # -----------------------------------------------------

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    ax.grid(
        axis="x",
        visible=False
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()


    # -----------------------------------------------------
    # 21. Speichern
    # -----------------------------------------------------

    os.makedirs(
        f"fig/{model_forecast}_predictability",
        exist_ok=True
    )

    savefile = (
        f"fig/{model_forecast}_predictability/"
        f"{model_forecast}_{energy_entry.name}_"
        f"{extreme_event}_"
        f"prediction_indices-withProbabilityAndEvents_"
        f"{pangu_start_year}-{pangu_end_year}_"
        f"logResModel{LAG_ID}.png"
    )

    plt.savefig(
        savefile,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


# =========================================================
# 22. GEMEINSAME ROC-KURVE FÜR ALLE VIER MODELLE
# =========================================================

plt.figure(
    figsize=(8, 7)
)


for lag_id in lag_ids:

    if lag_id not in roc_results:
        continue

    fpr = roc_results[lag_id]["fpr"]
    tpr = roc_results[lag_id]["tpr"]
    roc_auc = roc_results[lag_id]["auc"]

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=(
            f"Lag {lag_id}: "
            f"ROC-AUC = {roc_auc:.3f}"
        )
    )


# ---------------------------------------------------------
# No-skill-Linie
# ---------------------------------------------------------

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="gray",
    label="Random"
)


# ---------------------------------------------------------
# ROC-Beschriftungen
# ---------------------------------------------------------

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    f"ROC Curves – "
    f"{energy_entry.name} – "
    f"{extreme_event}"
)

plt.legend()

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()


# ---------------------------------------------------------
# ROC speichern
# ---------------------------------------------------------

os.makedirs(
    f"fig/{model_forecast}_predictability",
    exist_ok=True
)

roc_savefile = (
    f"fig/{model_forecast}_predictability/"
    f"{model_forecast}_{energy_entry.name}_"
    f"{extreme_event}_"
    f"roc_curves_all_lags_"
    f"{pangu_start_year}-{pangu_end_year}.png"
)

plt.savefig(
    roc_savefile,
    dpi=300,
    bbox_inches="tight"
)
plt.close()