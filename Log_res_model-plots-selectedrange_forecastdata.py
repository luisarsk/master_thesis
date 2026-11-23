import os
from pathlib import Path
import datetime as dt

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from coords_workflow_utils import PreparedCoordsLoader
from data_registry import Bibliography
from domino import agg
from domino.core import IndexGenerator, LaggedAnalyser, PatternFilter
from regionalmeancalculator import RegionalMeanCalculator

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

erstelldatum = dt.date.today()
lags = [-4, -3, -2, -1, 0]
lag_ids = [-4, -3, -2, -1]
region_ziel = "TenneT"
region_prec = "EUROPE"
quant = 0.9
extreme_event = "h"
maskierung = "_noMask"
cv_version = ""
model_forecast = "pangu"
pangu_start_year = 2022
pangu_end_year = 2022

selected_start = pd.Timestamp("2022-12-10")
selected_end = pd.Timestamp("2022-12-18")

energy_entries = bib.get_by_category("energy")
atm_entries = bib.get_by_category("atmosphere")
energy_entry = energy_entries[0]

output_dir = Path(f"fig/{model_forecast}_predictability/case_study")
output_dir.mkdir(parents=True, exist_ok=True)


def plot_model_timeseries_for_lag(
    x_df: pd.DataFrame,
    features: list[str],
    event_sel: xr.DataArray,
    event_abs_sel: xr.DataArray,
    lag_id: int,
) -> None:
    x_df = x_df.sort_values(["time", "prediction_time"]).copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for feat_idx, feature in enumerate(features):
        color = color_cycle[feat_idx % len(color_cycle)]
        first_segment = True
        for _, grp in x_df.groupby("time"):
            grp = grp.sort_values("prediction_time")
            ax.plot(
                pd.to_datetime(grp["prediction_time"]),
                grp[feature].values,
                color=color,
                linewidth=1.5,
                label=(feature if first_segment else None),
            )
            first_segment = False
    ax.set_xlabel("Prediction Time")
    ax.set_ylabel("Index Value")

    ax2 = ax.twinx()
    first_prob_segment = True
    for _, grp in x_df.groupby("time"):
        grp = grp.sort_values("prediction_time")
        ax2.plot(
            pd.to_datetime(grp["prediction_time"]),
            grp["probability"].values,
            color="black",
            linestyle="--",
            linewidth=1.8,
            label=("Predicted event probability" if first_prob_segment else None),
        )
        first_prob_segment = False

    event_abs_times = pd.to_datetime(event_abs_sel.time.values)
    event_abs_values = np.asarray(event_abs_sel.values)
    ax2.bar(
        event_abs_times,
        event_abs_values,
        width=0.8,
        alpha=0.25,
        color="yellow",
        label="Event amplitude",
    )
    ax2.set_ylabel("Event / Probability")

    max_event_value = np.nanmax(event_abs_values) if len(event_abs_values) > 0 else 1
    if max_event_value <= 1:
        ax2.set_ylim(0, 1)
    else:
        ax2.set_ylim(0, max(1, max_event_value * 1.05))

    event_times = pd.to_datetime(event_sel.time.values)
    event_values = np.asarray(event_sel.values)
    event_label_added = False
    for t, val in zip(event_times, event_values):
        if int(val) != 1:
            continue
        ax.axvspan(
            t - pd.Timedelta(hours=12),
            t + pd.Timedelta(hours=12),
            alpha=0.15,
            color="m",
            label=("Observed event" if not event_label_added else None),
        )
        event_label_added = True

    ax.set_xlim(selected_start - pd.Timedelta(days=0.5), selected_end + pd.Timedelta(days=0.5))

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", bbox_to_anchor=(1.04, 1))

    ax.set_title(
        f"{energy_entry.name} {extreme_event} - Prediction Indices, Event Probability and Events "
        f"(lag {lag_id}, {selected_start.date()} to {selected_end.date()})"
    )
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.grid(axis="x", visible=False)
    fig.autofmt_xdate()
    plt.tight_layout()

    savefile = output_dir / (
        f"{model_forecast}_{energy_entry.name}_{extreme_event}_{region_ziel}_"
        f"selectedRange_{selected_start.date()}_{selected_end.date()}_logResModel{lag_id}.png"
    )
    plt.savefig(savefile, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved model/index time series for lag {lag_id}: {savefile}")


# -----LOAD DATA----------
event = PreparedCoordsLoader.load_energy_event_from_prepared(
    path=energy_entry.path,
    energy_name=energy_entry.name,
    region_ziel=region_ziel,
    quant=quant,
    extreme_event=extreme_event,
)

event_abs = PreparedCoordsLoader.load_energy_values_from_prepared(
    path=energy_entry.path,
    energy_name=energy_entry.name,
    region_ziel=region_ziel,
)

atm_entry = atm_entries[0]
files = sorted(atm_entry.path.glob("*.nc"))
ds_all = xr.open_mfdataset(
    files,
    combine="by_coords",
    chunks={"time": 50, "latitude": 50, "longitude": 50},
).load()
ds_msl = ds_all.sel(time=ds_all.time.dt.month.isin([12, 1, 2]))

atm_entry = atm_entries[1]
files = sorted(atm_entry.path.glob("*.nc"))
ds_all = xr.open_mfdataset(
    files,
    combine="by_coords",
    chunks={"time": 50, "latitude": 50, "longitude": 50},
).load()
ds_z500 = ds_all.sel(time=ds_all.time.dt.month.isin([12, 1, 2]))

atm_entry = atm_entries[2]
files = sorted(atm_entry.path.glob("*.nc"))
ds_all = xr.open_mfdataset(
    files,
    combine="by_coords",
    chunks={"time": 50, "latitude": 50, "longitude": 50},
).load()
ds_ws = ds_all.sel(time=ds_all.time.dt.month.isin([12, 1, 2]))

ds_ws = ds_ws.assign_coords(longitude=((ds_ws.longitude + 180) % 360) - 180).sortby("longitude")
ds_msl = ds_msl.assign_coords(longitude=((ds_msl.longitude + 180) % 360) - 180).sortby("longitude")
ds_z500 = ds_z500.assign_coords(longitude=((ds_z500.longitude + 180) % 360) - 180).sortby("longitude")
ds_all = xr.merge([ds_msl, ds_z500, ds_ws], compat="override")

training_predictor = ds_all.sel(time=slice("1980", "2010"))
training_event = event.sel(time=slice("1980", "2010"))

analyser = LaggedAnalyser(event=training_event, variables=training_predictor)
analyser.lag_variables(lags, offset_unit="days", mode="past")

composites = []
lagged_means = []
for atm_entry in atm_entries:
    atm_name = atm_entry.name
    composites.append(
        xr.open_dataset(
            f"data//domino_output_coords_weatherbench/{energy_entry.name}/"
            f"{energy_entry.name}-{extreme_event}_{region_ziel}_{atm_name}_{region_prec}_1979_"
            f"lags_{'_'.join(map(str, lags))}_composites_coords_deseasonalized_DJF_anom.nc"
        )
    )
    lagged_means.append(
        xr.open_dataset(
            f"data//domino_output_coords_weatherbench/{energy_entry.name}/"
            f"{energy_entry.name}-{extreme_event}_{region_ziel}_{atm_name}_{region_prec}_1979_"
            f"lags_{'_'.join(map(str, lags))}_lagged-means_coords_deseasonalized_DJF_anom.nc"
        )
    )

composites_all = xr.merge(composites).assign_coords(
    longitude=((composites[0].longitude + 180) % 360) - 180
).sortby("longitude")
lagged_means_all = xr.merge(lagged_means).assign_coords(
    longitude=((lagged_means[0].longitude + 180) % 360) - 180
).sortby("longitude")

analyser.composites = composites_all
analyser.lagged_means = lagged_means_all
analyser.composites_are_anomaly = True
analyser.composite_func = (agg.mean_ds, agg.cat_occ_ds)

sig_parts = []
for atm_entry in atm_entries:
    atm_name = atm_entry.name
    sig_parts.append(
        xr.open_dataset(
            f"data//domino_output_coords_weatherbench/{energy_entry.name}/100/"
            f"{energy_entry.name}-{extreme_event}_{region_ziel}_{atm_name}_{region_prec}_1979_"
            f"lags_{'_'.join(map(str, lags))}_bootstrap-composites_deseasonalized_DJF_anom.nc"
        )
    )

S = xr.merge(sig_parts)
S = S.assign_coords(longitude=((S.longitude + 180) % 360) - 180).sortby("longitude")
analyser.composite_sigs = S
masked_composite = composites_all if maskierung == "_noMask" else S

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
    masked_ds = masked_composite.where(mask1)
else:
    masked_ds = masked_composite

IG = IndexGenerator()
var_name_to_var_id = {entry.name: i for i, entry in enumerate(atm_entries)}


def resolve_feature_to_var_id(feature_name: str) -> int:
    for var_name, var_id in var_name_to_var_id.items():
        if str(feature_name).startswith(f"{var_name}_index_"):
            return var_id
    raise KeyError(
        f"Feature '{feature_name}' kann keiner Atmosphaeren-Variable zugeordnet werden."
    )

pangu_data = []
clim_smooth_list = []
for entry in atm_entries:
    pangu = xr.open_dataset(
        f"data/{model_forecast}/{entry.name}/{model_forecast}_{entry.name}_{pangu_start_year}-{pangu_end_year}_case_study.nc"
    )
    pangu = pangu.assign_coords(
        longitude=((pangu.longitude + 180) % 360) - 180
    ).sortby("longitude")
    pangu_data.append(pangu)

    clim_smooth = xr.open_dataset(f"data/climsmooth/clim_smooth_{entry.name}.nc")
    clim_smooth = clim_smooth.assign_coords(
        longitude=((clim_smooth.longitude + 180) % 360) - 180
    ).sortby("longitude")
    clim_smooth_list.append(clim_smooth)
print(clim_smooth_list)

max_prediction_timedelta = pd.to_timedelta(
    np.max(pangu_data[0].prediction_timedelta.values),
    unit="h",
)
init_start_time = selected_start - max_prediction_timedelta
init_end_time = selected_end
print(
    f"[INFO] Initial time window for forecast generation: "
    f"{init_start_time} to {init_end_time}"
)

pangu_daily_means_all = {}
for var_id in range(len(pangu_data)):
    pangu_sel = pangu_data[var_id].sel(time=pangu_data[var_id].time.dt.hour == 0)
    if pangu_sel.sizes["time"] == 0:
        pangu_daily_means_all[var_id] = []
        print(f"[INFO] Keine 00-UTC Initialzeiten fuer var {atm_entries[var_id].name}.")
        continue

    pangu_time_min = pd.Timestamp(pangu_sel.time.min().item())
    pangu_time_max = pd.Timestamp(pangu_sel.time.max().item())
    print(
        f"[INFO] Verfuegbare Pangu-Zeit fuer {atm_entries[var_id].name}: "
        f"{pangu_time_min.date()} bis {pangu_time_max.date()}"
    )

    start_date = np.datetime64(init_start_time)
    end_date = np.datetime64(init_end_time)
    selected_mask = (
        (pangu_sel.time >= start_date)
        & (pangu_sel.time <= end_date)
    )
    pangu_sel = pangu_sel.sel(time=selected_mask)
    clim_sel = clim_smooth_list[var_id]

    pangu_daily_means = []
    for initial_time in pangu_sel.time:
        prediction_times = pangu_sel.prediction_timedelta * np.timedelta64(1, "h") + initial_time
        pangu_timestep_data = pangu_sel.sel(time=initial_time).assign_coords(prediction_time=prediction_times)
        pangu_timestep_data = pangu_timestep_data.sortby("prediction_time")
        doy = pangu_timestep_data["prediction_time"].dt.dayofyear
        clim_for_prediction = clim_sel.sel(dayofyear=doy)
        stepwise_anomaly = pangu_timestep_data - clim_for_prediction
        pangu_daily_means.append(stepwise_anomaly)

    pangu_daily_means_all[var_id] = pangu_daily_means
    print(f"[INFO] Pangu initial times for var {atm_entries[var_id].name}: {len(pangu_daily_means)}")

forecast_indices = {}
for var_id, atm_entry in enumerate(atm_entries):
    var_name = atm_entry.name
    forecast_indices[var_id] = {}
    for lag_id in lag_ids:
        index_name = f"{var_name}_index_val1_lag{lag_id}"
        index_slices = [dict(index_val=1, lag=lag_id)]
        index_params_path = Path(
            f"data/index_parameters/{energy_entry.name}/{extreme_event}/"
            f"{energy_entry.name}-{extreme_event}-{var_name}-index-{lag_id}-training{cv_version}{maskierung}.nc"
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

event_sel = event.sel(time=slice(selected_start, selected_end))
event_abs_sel = event_abs.sel(time=slice(selected_start, selected_end))

for lag_id in lag_ids:
    model_path = Path(
        f"fig/logRes_new/prepPangu/{energy_entry.name}/{extreme_event}/"
        f"{energy_entry.name}_{extreme_event}_logistic_model_lag{lag_id}{maskierung}_prepPangu.pkl"
    )
    if not model_path.exists():
        print(f"[INFO] Model not found for lag {lag_id}. Skipping.")
        continue

    saved = joblib.load(model_path)
    features = saved["features"]
    clf = saved["model"]

    feature_lists = {}
    for feature in features:
        var_id = resolve_feature_to_var_id(str(feature))
        feature_lists[feature] = forecast_indices[var_id][lag_id]

    if any(len(ds_list) == 0 for ds_list in feature_lists.values()):
        print(
            f"[INFO] Keine Forecast-Indizes fuer mindestens ein Feature bei lag {lag_id}. "
            "Wahrscheinlich ist der ausgewaehlte Zeitraum nicht in den Pangu-Initialzeiten enthalten."
        )
        continue

    by_time = {}
    for feature, ds_list in feature_lists.items():
        by_time[feature] = {pd.Timestamp(ds.time.values): ds[feature] for ds in ds_list}

    common_times = sorted(set.intersection(*(set(d.keys()) for d in by_time.values())))
    rows = []
    for init_time in common_times:
        arrays = [by_time[feature][init_time] for feature in features]
        arrays = xr.align(*arrays, join="inner")
        for i in range(len(arrays[0].prediction_time)):
            row = {
                "time": pd.Timestamp(init_time),
                "prediction_time": pd.Timestamp(arrays[0].prediction_time.values[i]),
            }
            for feature, arr in zip(features, arrays):
                row[feature] = float(arr.values[i])
            rows.append(row)

    x_df = pd.DataFrame(rows)
    if x_df.empty:
        print(f"[INFO] Keine Modell-Samples fuer lag {lag_id} im ausgewaehlten Zeitraum.")
        continue

    x_df = x_df.sort_values(["time", "prediction_time"])
    x_df = x_df[
        (pd.to_datetime(x_df["prediction_time"]) >= selected_start)
        & (pd.to_datetime(x_df["prediction_time"]) <= selected_end)
    ].copy()
    if x_df.empty:
        print(f"[INFO] Keine prediction_time im Fenster fuer lag {lag_id}.")
        continue

    x_df["lead_hours"] = (
        pd.to_datetime(x_df["prediction_time"]) - pd.to_datetime(x_df["time"])
    ).dt.total_seconds() / 3600.0

    X = x_df[features].to_numpy()
    x_df["prediction"] = clf.predict(X)
    x_df["probability"] = clf.predict_proba(X)[:, 1]
    x_df = x_df.sort_values("prediction_time")

    plot_model_timeseries_for_lag(
        x_df=x_df,
        features=features,
        event_sel=event_sel,
        event_abs_sel=event_abs_sel,
        lag_id=lag_id,
    )

end = time.time()
print(f"[INFO] Finished in {end - start:.2f} seconds.")
