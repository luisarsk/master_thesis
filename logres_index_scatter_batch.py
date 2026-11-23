from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from coords_workflow_utils import PreparedCoordsLoader
from data_registry import Bibliography
from domino.core import IndexGenerator
from regionalmeancalculator import RegionalMeanCalculator


# ----- CONFIG -----
LAGS = [-1] #-4, -3, -2, 
REGIONEN = ["Norddeutschland"]#, "50Hertz", "Amprion", "TenneT", "TransnetBW"]#"Norddeutschland",
EXTREME_EVENTS = ["l", "h"]
REGION_PRECURSOR = "EUROPE"
CV_VERSION = ""
MASKIERUNG = "_noMask"
TRAIN_SLICE = slice("1980", "2010")
TEST_SLICE = slice("2011", "2022")


def setup_bibliography() -> tuple[list, list]:
    bib = Bibliography(Path("data"))
    bib.add("energy", "WON", "daily", "subcountry", "prepared/WON_daily_subcountry_prepared_DJF.csv")
    # bib.add("energy", "SPV", "daily", "subcountry", "prepared/SPV_daily_subcountry_prepared_DJF.csv")
    bib.add("atmosphere", "mean_sea_level_pressure", "daily", "subcountry", "ERA5/prepared/DJF_anom/mean_sea_level_pressure")
    bib.add("atmosphere", "geopotential", "daily", "subcountry", "ERA5/prepared/DJF_anom/geopotential")
    bib.add("atmosphere", "v_component_of_wind", "daily", "subcountry", "ERA5/prepared/DJF_anom/v_component_of_wind")
    # bib.add('atmosphere', '2m_temperature', 'daily', 'global', 'ERA5/prepared/DJF_anom/2m_temperature')
    # bib.add('atmosphere', 'relative_humidity', 'daily', 'global', 'ERA5/prepared/DJF_anom/relative_humidity')       
    return bib.get_by_category("energy"), bib.get_by_category("atmosphere")


def load_predictor_dataset(atm_entries: list) -> xr.Dataset:
    datasets = []
    for atm_entry in atm_entries:
        files = sorted(atm_entry.path.glob("*.nc"))
        if not files:
            raise FileNotFoundError(f"Keine .nc Dateien gefunden für {atm_entry.name} in {atm_entry.path}")

        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            chunks={"time": 50, "latitude": 50, "longitude": 50},
        ).load()
        ds = ds.sel(time=ds.time.dt.month.isin([12, 1, 2]))
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby("longitude")
        datasets.append(ds)

    return xr.merge(datasets, compat="override")


def load_masked_composites(
    energy_name: str,
    region: str,
    extreme_event: str,
    atm_entries: list,
) -> xr.Dataset:
    lags_all = [-4, -3, -2, -1, 0]
    lag_str = "_".join(map(str, lags_all))
    composites = []
    lagged_means = []
    sig_masks = []

    for atm_entry in atm_entries:
        var_name = atm_entry.name
        base = (
            f"data//domino_output_coords_weatherbench/{energy_name}/"
            f"{energy_name}-{extreme_event}_{region}_{var_name}_{REGION_PRECURSOR}_1979_lags_{lag_str}"
        )
        comp_path = Path(base + "_composites_coords_deseasonalized_DJF_anom.nc")
        lagmean_path = Path(base + "_lagged-means_coords_deseasonalized_DJF_anom.nc")
        sig_path = Path(
            f"data//domino_output_coords_weatherbench/{energy_name}/100/"
            f"{energy_name}-{extreme_event}_{region}_{var_name}_{REGION_PRECURSOR}_1979_lags_{lag_str}_"
            "bootstrap-composites_deseasonalized_DJF_anom.nc"
        )

        if not comp_path.exists() or not lagmean_path.exists() or not sig_path.exists():
            raise FileNotFoundError(
                f"Fehlende Composite-Dateien für {energy_name}/{region}/{extreme_event}/{var_name} "
                f"(comp={comp_path.exists()}, lagged={lagmean_path.exists()}, sig={sig_path.exists()})"
            )

        composites.append(xr.open_dataset(comp_path))
        lagged_means.append(xr.open_dataset(lagmean_path))
        sig_masks.append(xr.open_dataset(sig_path))

    composites_all = xr.merge(composites)
    lagged_means_all = xr.merge(lagged_means)
    sig_all = xr.merge(sig_masks)

    composites_all = composites_all.assign_coords(
        longitude=((composites_all.longitude + 180) % 360) - 180
    ).sortby("longitude")
    lagged_means_all = lagged_means_all.assign_coords(
        longitude=((lagged_means_all.longitude + 180) % 360) - 180
    ).sortby("longitude")
    sig_all = sig_all.assign_coords(longitude=((sig_all.longitude + 180) % 360) - 180).sortby("longitude")

    if MASKIERUNG == "_noMask":
        return composites_all
    return composites_all.where(sig_all == 1)


def build_index_matrices(
    masked_ds: xr.Dataset,
    training_predictor: xr.Dataset,
    test_predictor: xr.Dataset,
    region: str,
    energy_name: str,
    extreme_event: str,
    var_names: list[str],
) -> tuple[xr.DataArray, xr.DataArray]:
    index_generator = IndexGenerator()
    train_indices = []
    test_indices = []

    for var_name in var_names:
        for lag_id in LAGS:
            index_params_path = Path(
                f"data/index_parameters/{region}/{energy_name}/{extreme_event}/"
                f"{energy_name}-{extreme_event}-{var_name}-index-{lag_id}-training{CV_VERSION}{MASKIERUNG}.nc"
            )
            if not index_params_path.exists():
                print(f"[WARN] Index-Parameter fehlen: {index_params_path}")
                continue

            params = xr.open_dataset(index_params_path)
            index_name = f"{var_name}_index_val1_lag{lag_id}"
            idx_mean = params[index_name].sel(param="mean").item()
            idx_std = params[index_name].sel(param="std").item()

            if idx_mean == 0 and idx_std == 0:
                print(f"[INFO] Skip {index_name}: mean=0 und std=0")
                continue

            slices = [dict(index_val=1, lag=lag_id)]
            train_idx = index_generator.generate(
                masked_ds[[var_name]],
                training_predictor[[var_name]],
                dim="time",
                slices=slices,
                ix_means={index_name: idx_mean},
                ix_stds={index_name: idx_std},
            )
            test_idx = index_generator.generate(
                masked_ds[[var_name]],
                test_predictor[[var_name]],
                dim="time",
                slices=slices,
                ix_means={index_name: idx_mean},
                ix_stds={index_name: idx_std},
            )
            train_indices.append(train_idx)
            test_indices.append(test_idx)

    if not train_indices or not test_indices:
        raise ValueError("Keine gültigen Indizes erzeugt (train/test leer).")

    x_train = xr.concat([ds.to_array().squeeze() for ds in train_indices], dim="feature").transpose("time", "feature")
    x_test = xr.concat([ds.to_array().squeeze() for ds in test_indices], dim="feature").transpose("time", "feature")

    train_feature_names = x_train.coords["variable"].values
    test_feature_names = x_test.coords["variable"].values
    x_train = x_train.drop_vars("variable").assign_coords(feature=("feature", train_feature_names))
    x_test = x_test.drop_vars("variable").assign_coords(feature=("feature", test_feature_names))

    nan_count = x_train.isnull().sum(dim="time")
    valid_features = nan_count < x_train.sizes["time"]
    x_train = x_train.sel(feature=valid_features)
    x_test = x_test.sel(feature=valid_features)

    return x_train, x_test


def train_forward_model(
    x_train: xr.DataArray,
    y_train: xr.DataArray,
    x_test: xr.DataArray,
    y_test: xr.DataArray,
    energy_name: str,
    extreme_event: str,
) -> tuple[LogisticRegression, np.ndarray]:
    base_model = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000, random_state=42)
    selector = SequentialFeatureSelector(
        estimator=base_model,
        n_features_to_select="auto",
        direction="forward",
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )

    selector.fit(x_train.values, y_train.values.astype(int))
    mask = selector.get_support()

    x_train_sel = x_train.sel(feature=mask)
    x_test_sel = x_test.sel(feature=mask)
    if x_train_sel.sizes["feature"] == 0:
        raise ValueError("Forward-Selection hat 0 Features geliefert.")

    model = LogisticRegression(penalty="l2", class_weight="balanced", max_iter=2000, random_state=42)
    model.fit(x_train_sel.values, y_train.values.astype(int))
    y_prob = model.predict_proba(x_test_sel.values)[:, 1]
    auc = roc_auc_score(y_test.values.astype(int), y_prob)
    print(f"[RESULT] {energy_name}-{extreme_event}: Forward-Model ROC-AUC = {auc:.3f}")

    return model, x_train_sel.feature.values


def _extract_feature_meta(feature_name: str) -> tuple[str | None, int | None]:
    match = re.match(r"^(?P<variable>.*?)_index_val1_lag(?P<lag>-?\d+)$", str(feature_name))
    if not match:
        return None, None
    return match.group("variable"), int(match.group("lag"))


def plot_lag_scatter(
    abs_values: np.ndarray,
    event_flags: np.ndarray,
    lag_series: list[dict],
    save_dir: Path,
    energy_name: str,
    region: str,
    extreme_event: str,
    y_limits: tuple[float, float],
) -> None:
    if not lag_series:
        print("[INFO] Keine Einzelindex-Serien zum Plotten gefunden.")
        return

    variables = sorted({entry["variable"] for entry in lag_series})
    cmap = plt.get_cmap("tab10")
    color_map = {var: cmap(i % 10) for i, var in enumerate(variables)}

    for lag_id in sorted({entry["lag"] for entry in lag_series}):
        lag_entries = [entry for entry in lag_series if entry["lag"] == lag_id]
        if not lag_entries:
            continue

        plt.figure(figsize=(8, 6))
        for entry in lag_entries:
            plt.scatter(
                abs_values,
                entry["values"],
                alpha=0.45,
                s=18,
                color=color_map[entry["variable"]],
                label=entry["variable"],
            )

        # Horizontale Linie bei y=0
        plt.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.6)

        # Vertikale Linie bei CF an Event-Tagen (test_event == 1)
        event_mask = np.asarray(event_flags) == 1
        if np.any(event_mask):
            event_cf = abs_values[event_mask]
            if extreme_event == "l":
                target_cf = float(np.nanmax(event_cf))
                plt.axvline(
                    x=target_cf,
                    color="blue",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.6,
                    label=f"Event-CF max (l): {target_cf:.2f}",
                )
            elif extreme_event == "h":
                target_cf = float(np.nanmin(event_cf))
                plt.axvline(
                    x=target_cf,
                    color="green",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.6,
                    label=f"Event-CF min (h): {target_cf:.2f}",
                )

        handles, labels = plt.gca().get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        plt.legend(unique.values(), unique.keys(), frameon=True, fontsize=11)
        plt.xlabel(f"{energy_name} abs value", fontsize=14)
        plt.ylabel("Index amplitude", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(y_limits)
        plt.title(f"{region} | {energy_name}-{extreme_event} | Lag {lag_id}",
                  fontsize=16,
                  fontweight="bold")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        scatter_plot = save_dir / f"{energy_name}_{extreme_event}_indices_lag{lag_id}_scatterv2.png"
        plt.savefig(scatter_plot, dpi=300)
        plt.close()
        print(f"[OK] Lag-Scatter gespeichert: {scatter_plot}")


def plot_forward_scatter(
    abs_values: np.ndarray,
    event_flags: np.ndarray,
    forward_amplitude: np.ndarray,
    save_dir: Path,
    energy_name: str,
    region: str,
    extreme_event: str,
    y_limits: tuple[float, float],
) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(abs_values, forward_amplitude, alpha=0.5, s=18, color="black")
    
    # Horizontale Linie bei y=0
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1, alpha=0.6, label="CF = 0")
    
    # Vertikale Linie bei CF an Event-Tagen (test_event == 1)
    event_mask = np.asarray(event_flags) == 1
    if np.any(event_mask):
        event_cf = abs_values[event_mask]
        if extreme_event == "l":
            target_cf = float(np.nanmax(event_cf))
            plt.axvline(
                x=target_cf,
                color="blue",
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                label=f"Event-CF max (l): {target_cf:.2f}",
            )
        elif extreme_event == "h":
            target_cf = float(np.nanmin(event_cf))
            plt.axvline(
                x=target_cf,
                color="green",
                linestyle="--",
                linewidth=1.5,
                alpha=0.6,
                label=f"Event-CF min (h): {target_cf:.2f}",
            )
    
    plt.legend(frameon=True)
    plt.xlabel(f"{energy_name} abs value")
    plt.ylabel("Index amplitude")
    plt.ylim(y_limits)
    plt.title(f"{region} | {energy_name}-{extreme_event}: Forward Predictor Model")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    scatter_plot = save_dir / f"{energy_name}_{extreme_event}_forward_model_scatterv2.png"
    plt.savefig(scatter_plot, dpi=300)
    plt.close()
    print(f"[OK] Forward-Scatter gespeichert: {scatter_plot}")


def run_case(
    region: str,
    extreme_event: str,
    energy_entry,
    atm_entries: list,
    predictor_all: xr.Dataset,
) -> dict:
    quant = 0.1 if extreme_event == "l" else 0.9
    var_names = [entry.name for entry in atm_entries]

    event = PreparedCoordsLoader.load_energy_event_from_prepared(
        path=energy_entry.path,
        energy_name=energy_entry.name,
        region_ziel=region,
        quant=quant,
        extreme_event=extreme_event,
    )
    training_event = event.sel(time=TRAIN_SLICE)
    test_event = event.sel(time=TEST_SLICE)
    training_predictor = predictor_all.sel(time=TRAIN_SLICE)
    test_predictor = predictor_all.sel(time=TEST_SLICE)

    masked_ds = load_masked_composites(
        energy_name=energy_entry.name,
        region=region,
        extreme_event=extreme_event,
        atm_entries=atm_entries,
    )

    x_train, x_test = build_index_matrices(
        masked_ds=masked_ds,
        training_predictor=training_predictor,
        test_predictor=test_predictor,
        region=region,
        energy_name=energy_entry.name,
        extreme_event=extreme_event,
        var_names=var_names,
    )

    x_train, training_event = xr.align(x_train, training_event, join="inner")
    x_test, test_event = xr.align(x_test, test_event, join="inner")

    save_dir = Path(f"fig/logRes_new/prepPangu/scatterplots/{region}/{energy_entry.name}/{extreme_event}")
    save_dir.mkdir(parents=True, exist_ok=True)

    model, selected_features = train_forward_model(
        x_train=x_train,
        y_train=training_event,
        x_test=x_test,
        y_test=test_event,
        energy_name=energy_entry.name,
        extreme_event=extreme_event,
    )

    x_selected = x_test.sel(feature=selected_features)

    calculator = RegionalMeanCalculator()
    energy_abs = calculator.load_energy_values_from_prepared(
        path=energy_entry.path,
        energy_name=energy_entry.name,
        region_name=region,
    )
    energy_abs_test = energy_abs.sel(time=TEST_SLICE)

    common_time = np.intersect1d(
        x_test.time.values.astype("datetime64[ns]"),
        energy_abs_test.time.values.astype("datetime64[ns]"),
    )
    if len(common_time) == 0:
        raise ValueError("Keine gemeinsamen Zeitpunkte zwischen Indizes und Energy-Abs-Werten.")

    x_test_common = x_test.sel(time=common_time)
    x_selected_common = x_selected.sel(time=common_time)
    abs_values = energy_abs_test.sel(time=common_time).values
    event_flags = np.asarray(test_event.sel(time=common_time).values).reshape(-1)
    forward_amplitude = model.decision_function(x_selected_common.values)

    lag_series: list[dict] = []
    for feature in x_test_common.feature.values:
        variable, lag = _extract_feature_meta(str(feature))
        if variable is None or lag is None:
            continue
        lag_series.append(
            {
                "feature": str(feature),
                "variable": variable,
                "lag": lag,
                "values": x_test_common.sel(feature=feature).values,
            }
        )

    if not lag_series:
        raise ValueError("Keine Einzelindex-Features mit Lag-Information gefunden.")

    return {
        "save_dir": save_dir,
        "energy_name": energy_entry.name,
        "region": region,
        "extreme_event": extreme_event,
        "abs_values": abs_values,
        "event_flags": event_flags,
        "forward_amplitude": forward_amplitude,
        "lag_series": lag_series,
    }


def main() -> None:
    energy_entries, atm_entries = setup_bibliography()
    predictor_all = load_predictor_dataset(atm_entries)
    plot_cases = []

    for region in REGIONEN:
        for energy_entry in energy_entries:
            for extreme_event in EXTREME_EVENTS:
                case_label = f"{region} | {energy_entry.name} | {extreme_event}"
                print(f"\n[CASE] {case_label}")
                try:
                    case_payload = run_case(
                        region=region,
                        extreme_event=extreme_event,
                        energy_entry=energy_entry,
                        atm_entries=atm_entries,
                        predictor_all=predictor_all,
                    )
                    plot_cases.append(case_payload)
                except (FileNotFoundError, ValueError) as exc:
                    print(f"[SKIP] {case_label}: {exc}")

    if not plot_cases:
        print("[INFO] Keine validen Fälle für Scatter-Plots gefunden.")
        return

    all_index_values = []
    for case in plot_cases:
        all_index_values.append(case["forward_amplitude"])
        for entry in case["lag_series"]:
            all_index_values.append(entry["values"])
    all_index_values = np.concatenate(all_index_values)
    y_min = float(np.nanmin(all_index_values))
    y_max = float(np.nanmax(all_index_values))
    padding = max((y_max - y_min) * 0.05, 0.1)
    y_limits = (y_min - padding, y_max + padding)
    print(f"[INFO] Einheitliche y-Achse für alle Scatter-Plots: {y_limits}")

    for case in plot_cases:
        plot_lag_scatter(
            abs_values=case["abs_values"],
            event_flags=case["event_flags"],
            lag_series=case["lag_series"],
            save_dir=case["save_dir"],
            energy_name=case["energy_name"],
            region=case["region"],
            extreme_event=case["extreme_event"],
            y_limits=y_limits,
        )
        plot_forward_scatter(
            abs_values=case["abs_values"],
            event_flags=case["event_flags"],
            forward_amplitude=case["forward_amplitude"],
            save_dir=case["save_dir"],
            energy_name=case["energy_name"],
            region=case["region"],
            extreme_event=case["extreme_event"],
            y_limits=y_limits,
        )


if __name__ == "__main__":
    main()
