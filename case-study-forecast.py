from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr

from data_registry import Bibliography
from domino.core import IndexGenerator


# ============================================================
# CONFIGURATION
# ============================================================

# "ifs_hres" oder "pangu"
FORECAST_SOURCE = "ifs_hres"
FORECAST_START_YEAR = 2022
FORECAST_END_YEAR = 2022

# Variablen analog zu den anderen LogReg-Skripten
SELECTED_INDICES = [
    "mean_sea_level_pressure",
    "geopotential",
    "v_component_of_wind",
    "2m_temperature",
    "relative_humidity"
]

# Nur anwenden, wenn level-Dimension/-Koordinate in den Daten existiert
LEVEL_SELECTION = {
    "geopotential": 500,
    "v_component_of_wind": 300,
    "relative_humidity": 700
}

# Case-Study Event-Tage
EVENT_DATES = [
    "2022-12-16",
    "2022-12-17",
    # "2022-12-10",
]

REGIONEN = [
    "Norddeutschland",
    "50Hertz",
    "Amprion",
    "TenneT",
    "TransnetBW",
]

EXTREME_EVENTS = ["l", "h"]
LAGS = [-4, -3, -2, -1]

CV_VERSION = ""
MASKIERUNG = "_noMask"
REGION_PRECURSOR = "EUROPE"
MAX_LEAD_HOURS = 240

OUTPUT_DIR = Path("fig/case_study_forecast")


def setup_bibliography() -> tuple[list, list]:
    bib = Bibliography(Path("data"))
    # bib.add("energy", "WON", "daily", "subcountry", "prepared/WON_daily_subcountry_prepared_DJF.csv")
    bib.add("energy", "SPV", "daily", "subcountry", "prepared/SPV_daily_subcountry_prepared_DJF.csv")
    bib.add("atmosphere", "mean_sea_level_pressure", "daily", "subcountry", "ERA5/prepared/DJF_anom/mean_sea_level_pressure")
    # bib.add("atmosphere", "geopotential", "daily", "subcountry", "ERA5/prepared/DJF_anom/geopotential")
    # bib.add("atmosphere", "v_component_of_wind", "daily", "subcountry", "ERA5/prepared/DJF_anom/v_component_of_wind")
    bib.add('atmosphere', '2m_temperature', 'daily', 'global', 'ERA5/prepared/DJF_anom/2m_temperature')
    bib.add('atmosphere', 'relative_humidity', 'daily', 'global', 'ERA5/prepared/DJF_anom/relative_humidity')       
    return bib.get_by_category("energy"), bib.get_by_category("atmosphere")


def load_masked_composites(
    energy_name: str,
    region: str,
    extreme_event: str,
    atm_entries: list,
) -> xr.Dataset:
    lags_all = [-4, -3, -2, -1, 0]
    lag_str = "_".join(map(str, lags_all))
    composites = []

    for atm_entry in atm_entries:
        var_name = atm_entry.name
        comp_path = Path(
            f"data/domino_output_coords_weatherbench/{energy_name}/"
            f"{energy_name}-{extreme_event}_{region}_{var_name}_{REGION_PRECURSOR}_1979_lags_{lag_str}_"
            "composites_coords_deseasonalized_DJF_anom.nc"
        )
        if not comp_path.exists():
            raise FileNotFoundError(f"Composite-Datei fehlt: {comp_path}")
        composites.append(xr.open_dataset(comp_path))

    masked_ds = xr.merge(composites)
    masked_ds = masked_ds.assign_coords(
        longitude=((masked_ds.longitude + 180) % 360) - 180
    ).sortby("longitude")
    return masked_ds


def _apply_level_if_available(ds: xr.Dataset, variable: str) -> xr.Dataset:
    if variable not in LEVEL_SELECTION:
        return ds

    level_value = LEVEL_SELECTION[variable]
    if "level" in ds.dims:
        level_coord = ds.coords.get("level", None)
        if level_coord is not None and level_coord.ndim == 1:
            level_values = np.asarray(level_coord.values)
            matches = np.where(level_values == level_value)[0]
            if len(matches) == 0:
                raise ValueError(
                    f"Level {level_value} nicht in Forecast-Daten für {variable} vorhanden. "
                    f"Verfügbare level: {level_values.tolist()}"
                )
            return ds.isel(level=int(matches[0]))
        return ds.sel(level=level_value)

    if "level" in ds.coords:
        level_coord = ds["level"]
        if level_coord.ndim == 0:
            existing_level = level_coord.item()
            if existing_level != level_value:
                raise ValueError(
                    f"Forecast-Daten für {variable} haben fixes level={existing_level}, "
                    f"aber konfiguriert ist {level_value}."
                )
            return ds
        return ds.sel(level=level_value)

    return ds


def _find_forecast_file(variable: str) -> Path:
    preferred = Path(
        f"data/{FORECAST_SOURCE}/{variable}/"
        f"{FORECAST_SOURCE}_{variable}_{FORECAST_START_YEAR}-{FORECAST_END_YEAR}_case_study.nc"
    )
    if preferred.exists():
        return preferred

    fallback = Path(
        f"data/{FORECAST_SOURCE}/{variable}/"
        f"{FORECAST_SOURCE}_{variable}_{FORECAST_START_YEAR}-{FORECAST_END_YEAR}_DJF.nc"
    )
    if fallback.exists():
        return fallback

    candidates = sorted(
        Path(f"data/{FORECAST_SOURCE}/{variable}").glob(f"{FORECAST_SOURCE}_{variable}_*.nc")
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        f"Keine Forecast-Datei für {FORECAST_SOURCE}/{variable} gefunden."
    )


def load_forecast_variable(variable: str) -> xr.Dataset:
    nc_path = _find_forecast_file(variable)
    ds = xr.open_dataset(nc_path)

    if "longitude" in ds.coords:
        ds = ds.assign_coords(
            longitude=((ds.longitude + 180) % 360) - 180
        ).sortby("longitude")

    ds = _apply_level_if_available(ds, variable)

    if "prediction_timedelta" not in ds.coords and "prediction_timedelta" not in ds.dims:
        raise ValueError(f"{nc_path} hat keine prediction_timedelta-Dimension.")

    return ds


def load_clim_smooth_variable(variable: str) -> xr.Dataset:
    clim_path = Path(f"data/climsmooth/clim_smooth_{variable}.nc")
    if not clim_path.exists():
        raise FileNotFoundError(f"Klimatologie-Datei fehlt: {clim_path}")

    clim = xr.open_dataset(clim_path)
    if "longitude" in clim.coords:
        clim = clim.assign_coords(
            longitude=((clim.longitude + 180) % 360) - 180
        ).sortby("longitude")

    clim = _apply_level_if_available(clim, variable)
    return clim


def build_daily_forecast_anomalies(
    forecast_by_var: dict[str, xr.Dataset],
    clim_by_var: dict[str, xr.Dataset],
) -> dict[str, list[xr.Dataset]]:
    daily_means_by_var: dict[str, list[xr.Dataset]] = {}

    for variable, forecast_ds in forecast_by_var.items():
        clim_ds = clim_by_var[variable]
        ds_00 = forecast_ds.sel(time=forecast_ds.time.dt.hour == 0)

        if ds_00.sizes.get("time", 0) == 0:
            daily_means_by_var[variable] = []
            continue

        series_list: list[xr.Dataset] = []

        for init_time in ds_00.time:
            prediction_times = ds_00.prediction_timedelta * np.timedelta64(1, "h") + init_time

            step = ds_00.sel(time=init_time)
            step = step.assign_coords(prediction_time=prediction_times)

            daily_mean = step.groupby("prediction_time.date").mean()
            daily_mean = (
                daily_mean
                .assign_coords(prediction_time=("date", pd.to_datetime(daily_mean.date.values)))
                .swap_dims({"date": "prediction_time"})
                .drop_vars("date")
            )

            day_of_year = daily_mean["prediction_time"].dt.dayofyear
            clim_for_prediction = clim_ds.sel(dayofyear=day_of_year)
            daily_mean = daily_mean - clim_for_prediction

            series_list.append(daily_mean)

        daily_means_by_var[variable] = series_list

    return daily_means_by_var


def _extract_feature_meta(feature_name: str) -> tuple[str | None, int | None]:
    match = re.match(r"^(?P<variable>.*?)_index_val1_lag(?P<lag>-?\d+)$", str(feature_name))
    if not match:
        return None, None
    return match.group("variable"), int(match.group("lag"))


def calculate_forecast_index_series(
    forecast_daily_mean: xr.Dataset,
    masked_ds: xr.Dataset,
    region: str,
    energy_name: str,
    extreme_event: str,
    var_name: str,
    lag_id: int,
) -> xr.DataArray | None:
    index_params_path = Path(
        f"data/index_parameters/{region}/{energy_name}/{extreme_event}/"
        f"{energy_name}-{extreme_event}-{var_name}-index-{lag_id}-training{CV_VERSION}{MASKIERUNG}.nc"
    )
    if not index_params_path.exists():
        return None

    index_name = f"{var_name}_index_val1_lag{lag_id}"
    params = xr.open_dataset(index_params_path)
    try:
        idx_mean = params[index_name].sel(param="mean").item()
        idx_std = params[index_name].sel(param="std").item()
    finally:
        params.close()

    if idx_mean == 0 and idx_std == 0:
        return None

    generator = IndexGenerator()
    index_result = generator.generate(
        masked_ds[[var_name]],
        forecast_daily_mean[[var_name]],
        dim="prediction_time",
        slices=[dict(index_val=1, lag=lag_id)],
        ix_means={index_name: idx_mean},
        ix_stds={index_name: idx_std},
    )
    result_variable = list(index_result.data_vars)[0]
    return index_result[result_variable]


def compute_case_study_indices(
    daily_means_by_var: dict[str, list[xr.Dataset]],
    masked_ds: xr.Dataset,
    event_date: str,
    region: str,
    energy_name: str,
    extreme_event: str,
    var_names: list[str],
) -> dict[int, dict[str, list[dict]]]:
    event_start = np.datetime64(event_date, "D")
    event_end = event_start + np.timedelta64(1, "D")

    results: dict[int, dict[str, list[dict]]] = {
        lag: {var_name: [] for var_name in var_names} for lag in LAGS
    }

    for var_name in var_names:
        for forecast_daily_mean in daily_means_by_var[var_name]:
            if "time" not in forecast_daily_mean.coords:
                continue
            init_time = np.datetime64(forecast_daily_mean.time.values, "ns")

            for lag_id in LAGS:
                index_series = calculate_forecast_index_series(
                    forecast_daily_mean=forecast_daily_mean,
                    masked_ds=masked_ds,
                    region=region,
                    energy_name=energy_name,
                    extreme_event=extreme_event,
                    var_name=var_name,
                    lag_id=lag_id,
                )
                if index_series is None:
                    continue

                pred_times = index_series.prediction_time.values.astype("datetime64[ns]")
                mask = (pred_times >= event_start) & (pred_times < event_end)
                if not np.any(mask):
                    continue

                values = index_series.values[mask]
                valid_pred_times = pred_times[mask]
                lead_hours = (valid_pred_times - init_time) / np.timedelta64(1, "h")
                lead_to_event = float((event_start - init_time) / np.timedelta64(1, "h"))

                results[lag_id][var_name].append(
                    {
                        "init_time": init_time,
                        "lead_to_event_hours": lead_to_event,
                        "mean_matching_lead_hours": float(np.nanmean(lead_hours)),
                        "index_value": float(np.nanmean(values)),
                        "n_steps": int(np.count_nonzero(np.isfinite(values))),
                    }
                )

    return results


def _load_forward_model(region: str, energy_name: str, extreme_event: str):
    candidates = [
        Path(
            f"fig/logRes_new/prepPangu/{region}/{energy_name}/{extreme_event}/"
            f"{energy_name}_{extreme_event}_logistic_model_forward-predictor-selection{CV_VERSION}{MASKIERUNG}_prepPangu.pkl"
        ),
        Path(
            f"fig/logRes_new/prepPangu/{energy_name}/{extreme_event}/"
            f"{energy_name}_{extreme_event}_logistic_model_forward-predictor-selection{CV_VERSION}{MASKIERUNG}_prepPangu.pkl"
        ),
    ]
    model_path = next((p for p in candidates if p.exists()), None)
    if model_path is None:
        raise FileNotFoundError("Forward-Predictor-Selection Modell nicht gefunden.")
    saved = joblib.load(model_path)
    return saved["model"], [str(f) for f in saved["features"]], model_path


def _add_prediction_hours_top_axis(ax, x_times: pd.DatetimeIndex, event_date: str) -> None:
    if len(x_times) == 0:
        return

    event_start = np.datetime64(event_date, "D")
    x_ns = x_times.values.astype("datetime64[ns]")
    pred_hours = ((event_start - x_ns) / np.timedelta64(1, "h")).astype(int)

    tick_step = max(1, int(np.ceil(len(x_times) / 10)))
    tick_times = x_times[::tick_step]
    tick_hours = pred_hours[::tick_step]

    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.set_xticks(mdates.date2num(tick_times.to_pydatetime()))
    ax_top.set_xticklabels([str(int(h)) for h in tick_hours], rotation=0)
    ax_top.set_xlabel("Prediction time (hours)")


def plot_indices_per_lag(
    lag_results: dict[int, dict[str, list[dict]]],
    event_date: str,
    save_dir: Path,
    region: str,
    energy_name: str,
    extreme_event: str,
    y_limits: tuple[float, float],
) -> None:
    cmap = plt.get_cmap("tab10")
    color_map = {var: cmap(i % 10) for i, var in enumerate(SELECTED_INDICES)}

    for lag_id in LAGS:
        fig, ax = plt.subplots(figsize=(13, 5))
        plotted_any = False
        all_x_times = []

        for var_name, records in lag_results.get(lag_id, {}).items():
            if not records:
                continue
            records = sorted(records, key=lambda x: x["init_time"])
            x_time = pd.to_datetime([r["init_time"] for r in records])
            y_values = np.array([r["index_value"] for r in records])
            all_x_times.extend(list(x_time))

            ax.plot(
                x_time,
                y_values,
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=var_name,
                color=color_map.get(var_name),
            )
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            print(f"[INFO] Keine Daten für Lag {lag_id} ({region}/{energy_name}/{extreme_event}/{event_date}).")
            continue

        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylim(y_limits)
        ax.set_ylabel("Standardized index amplitude")
        ax.set_xlabel("Forecast initial time")
        ax.set_title(
            f"{FORECAST_SOURCE} | {region} | {energy_name}-{extreme_event} | Event {event_date} | Lag {lag_id}"
        )
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

        unique_times = pd.DatetimeIndex(sorted(set(pd.to_datetime(all_x_times))))
        _add_prediction_hours_top_axis(ax, unique_times, event_date)
        fig.tight_layout()

        save_dir.mkdir(parents=True, exist_ok=True)
        out_file = save_dir / f"{energy_name}_{extreme_event}_{event_date}_indices_lag{lag_id}_{FORECAST_SOURCE}.png"
        fig.savefig(out_file, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] Plot gespeichert: {out_file}")


def build_forward_predictor_series(
    lag_results: dict[int, dict[str, list[dict]]],
    region: str,
    energy_name: str,
    extreme_event: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clf, features, model_path = _load_forward_model(region, energy_name, extreme_event)
    print(f"[INFO] Verwende Forward-Modell: {model_path}")

    per_feature_records: dict[str, dict[np.datetime64, float]] = {}
    for feature_name in features:
        var_name, lag_id = _extract_feature_meta(feature_name)
        if var_name is None or lag_id is None:
            continue
        records = lag_results.get(lag_id, {}).get(var_name, [])
        per_feature_records[feature_name] = {
            np.datetime64(r["init_time"], "ns"): float(r["index_value"]) for r in records
        }

    if len(per_feature_records) != len(features):
        missing = [f for f in features if f not in per_feature_records]
        raise ValueError(f"Forward-Features nicht vollständig in Forecast-Indizes verfügbar: {missing}")

    common_times = sorted(set.intersection(*(set(rec.keys()) for rec in per_feature_records.values())))
    if not common_times:
        raise ValueError("Keine gemeinsamen Initialzeiten für Forward-Features gefunden.")

    x_rows = []
    for init_time in common_times:
        x_rows.append([per_feature_records[f][init_time] for f in features])
    X = np.asarray(x_rows)

    y_prob = clf.predict_proba(X)[:, 1]
    y_score = clf.decision_function(X)
    x_time = pd.to_datetime(common_times)
    return x_time, y_prob, y_score


def plot_forward_predictor_model(
    x_time: np.ndarray,
    y_prob: np.ndarray,
    y_score: np.ndarray,
    event_date: str,
    save_dir: Path,
    region: str,
    energy_name: str,
    extreme_event: str,
    y_limits: tuple[float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(x_time, y_prob, marker="o", linewidth=2, color="black", label="Predicted probability")
    ax.plot(x_time, y_score, marker="s", linewidth=1.3, color="tab:blue", alpha=0.8, label="Decision score")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_ylim(y_limits)
    ax.set_xlabel("Forecast initial time")
    ax.set_ylabel("Forward model output")
    ax.set_title(f"{FORECAST_SOURCE} | {region} | {energy_name}-{extreme_event} | Event {event_date} | Forward model")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    _add_prediction_hours_top_axis(ax, pd.DatetimeIndex(pd.to_datetime(x_time)), event_date)
    fig.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f"{energy_name}_{extreme_event}_{event_date}_forward-model_{FORECAST_SOURCE}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Plot gespeichert: {out_file}")


def main() -> None:
    energy_entries, atm_entries = setup_bibliography()
    atm_entries = [entry for entry in atm_entries if entry.name in SELECTED_INDICES]
    selected_variables = [entry.name for entry in atm_entries]
    if not selected_variables:
        raise ValueError("SELECTED_INDICES passt nicht zu den verfügbaren Atmosphären-Variablen.")

    forecast_by_var = {}
    clim_by_var = {}
    for variable in selected_variables:
        print(f"[LOAD] Forecast {FORECAST_SOURCE}: {variable}")
        forecast_by_var[variable] = load_forecast_variable(variable)
        clim_by_var[variable] = load_clim_smooth_variable(variable)

    daily_means_by_var = build_daily_forecast_anomalies(forecast_by_var, clim_by_var)

    plot_payloads = []
    y_values_global = []

    for region in REGIONEN:
        for energy_entry in energy_entries:
            for extreme_event in EXTREME_EVENTS:
                case_label = f"{region} | {energy_entry.name} | {extreme_event}"
                print(f"\n[CASE] {case_label}")

                try:
                    masked_ds = load_masked_composites(
                        energy_name=energy_entry.name,
                        region=region,
                        extreme_event=extreme_event,
                        atm_entries=atm_entries,
                    )
                except FileNotFoundError as exc:
                    print(f"[SKIP] {case_label}: {exc}")
                    continue

                for event_date in EVENT_DATES:
                    print(f"[EVENT] {event_date}")
                    try:
                        lag_results = compute_case_study_indices(
                            daily_means_by_var=daily_means_by_var,
                            masked_ds=masked_ds,
                            event_date=event_date,
                            region=region,
                            energy_name=energy_entry.name,
                            extreme_event=extreme_event,
                            var_names=selected_variables,
                        )
                        x_time, y_prob, y_score = build_forward_predictor_series(
                            lag_results=lag_results,
                            region=region,
                            energy_name=energy_entry.name,
                            extreme_event=extreme_event,
                        )

                        for lag_id in LAGS:
                            for records in lag_results.get(lag_id, {}).values():
                                if records:
                                    y_values_global.extend([float(r["index_value"]) for r in records])
                        y_values_global.extend([float(v) for v in y_score])
                        y_values_global.extend([float(v) for v in y_prob])

                        plot_payloads.append(
                            {
                                "lag_results": lag_results,
                                "event_date": event_date,
                                "save_dir": OUTPUT_DIR / FORECAST_SOURCE / region / energy_entry.name / extreme_event,
                                "region": region,
                                "energy_name": energy_entry.name,
                                "extreme_event": extreme_event,
                                "x_time": x_time,
                                "y_prob": y_prob,
                                "y_score": y_score,
                            }
                        )
                    except (FileNotFoundError, ValueError, KeyError) as exc:
                        print(f"[SKIP] {case_label} | {event_date}: {exc}")

    if not plot_payloads:
        print("[INFO] Keine validen Fälle für Plots gefunden.")
        return

    if not y_values_global:
        print("[INFO] Keine y-Werte für globale y-Achse gefunden.")
        return

    y_min = float(np.nanmin(y_values_global))
    y_max = float(np.nanmax(y_values_global))
    padding = max((y_max - y_min) * 0.05, 0.1)
    y_limits = (y_min - padding, y_max + padding)
    print(f"[INFO] Globale feste y-Achse für alle Plots: {y_limits}")

    for payload in plot_payloads:
        plot_indices_per_lag(
            lag_results=payload["lag_results"],
            event_date=payload["event_date"],
            save_dir=payload["save_dir"],
            region=payload["region"],
            energy_name=payload["energy_name"],
            extreme_event=payload["extreme_event"],
            y_limits=y_limits,
        )
        plot_forward_predictor_model(
            x_time=payload["x_time"],
            y_prob=payload["y_prob"],
            y_score=payload["y_score"],
            event_date=payload["event_date"],
            save_dir=payload["save_dir"],
            region=payload["region"],
            energy_name=payload["energy_name"],
            extreme_event=payload["extreme_event"],
            y_limits=y_limits,
        )


if __name__ == "__main__":
    main()
