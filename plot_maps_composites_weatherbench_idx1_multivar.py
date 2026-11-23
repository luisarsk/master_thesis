## Plot WeatherBench composites for index_val=1 with 3 variables ##
####################################################################

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# -----------------------------
# CONFIGURATION
# -----------------------------
lags = [-4, -3, -2, -1, 0]
# Choose which of the available lags should be plotted.
# Set to None to plot all lags contained in the input files.
plot_lags = [-4, -2, 0]
region_ziel = "Norddeutschland"
region_prec = "EUROPE"
timeperiod = "1979"
extreme_event = "h"
energy_name = "WON"
bootstrap_dir = "100"
index_value = 1

# 'WON vars:'
variables = [
    # {
    #         "name": "10m_wind_speed",
    #         "label": "10m wind speed anomaly",
    #         "unit": r"$\mathrm{m\,s^{-1}}$,
    #         "cmap": "RdBu_r",
    #         "vmin": None, #l: 1200
    #         "vmax": None,#l
    {
        "name": "mean_sea_level_pressure",
        "label": "MSLP anomaly",
        "unit": "Pa",
        "cmap": "RdBu_r",
        "vmin": -1200.0, #l: 1200
        "vmax": 1200.0,#l
    },
    {
        "name": "geopotential",
        "label": "Z500 anomaly",
        "unit": r"$\mathrm{m^2\,s^{-2}}$",
        "cmap": "RdBu_r",
        "vmin": -1200.0, #l: 1200
        "vmax": 1200.0,
    },
    {
        "name": "v_component_of_wind",
        "label": "V wind anomaly",
        "unit": r"$\mathrm{m\,s^{-1}}$",
        "cmap": "RdBu_r",
        "vmin": -8.0,
        "vmax": 8.0,
    },
]

# # SPV Vars:
# variables = [
#     {
#         "name": "mean_sea_level_pressure",
#         "label": "MSLP anomaly",
#         "unit": "Pa",
#         "cmap": "RdBu_r",
#         "vmin": -1000.0, #l: 1200
#         "vmax": 1000.0,#l
#     },
#     {
#         "name": "2m_temperature",
#         "label": "2m temperature anomaly",
#         "unit": "K",
#         "cmap": "RdBu_r",
#         "vmin": -4.0, #l: 1200
#         "vmax": 4.0,
#     },
#     {
#         "name": "relative_humidity",
#         "label": "Relative humidity 700 hPa anomaly",
#         "unit": "10² %",
#         "cmap": "RdBu_r",
#         "vmin": -0.2,
#         "vmax": 0.2,
#     },
# ]
lags_str = "_".join(str(lag) for lag in lags)
output_path = (
    Path("fig")
    / "domino_output_coords_weatherbench"
    / energy_name
    / bootstrap_dir
    / (
        f"{energy_name}-{extreme_event}_{region_ziel}_multivar_{region_prec}_{timeperiod}"
        f"_idx{index_value}_bootstrap-composites_deseasonalized_DJF_anom.png"
    )
)


def _normalize_lon(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
    return ds.sortby("longitude").sortby("latitude")


def _build_input_path(variable_name: str) -> Path:
    return (
        Path("data")
        / "domino_output_coords_weatherbench"
        / energy_name
        # / extreme_event
        / bootstrap_dir
        / (
            f"{energy_name}-{extreme_event}_{region_ziel}_{variable_name}_{region_prec}_{timeperiod}"
            f"_lags_{lags_str}_bootstrap-composites_deseasonalized_DJF_anom.nc"
        )
    )


def _load_variable(variable_name: str) -> xr.DataArray:
    in_path = _build_input_path(variable_name)
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    ds = xr.open_dataset(in_path)
    try:
        if variable_name not in ds.data_vars:
            raise KeyError(f"Variable '{variable_name}' not found in {in_path}")

        ds = _normalize_lon(ds)
        if "index_val" not in ds.coords:
            raise KeyError(f"'index_val' coordinate missing in {in_path}")
        if index_value not in ds.index_val.values:
            raise ValueError(f"index_val={index_value} not available in {in_path}")

        return ds[variable_name].load()
    finally:
        ds.close()


var_arrays: dict[str, xr.DataArray] = {}
for var_cfg in variables:
    var_name = var_cfg["name"]
    var_arrays[var_name] = _load_variable(var_name)

available_lags = var_arrays[variables[0]["name"]].lag.values
for var_cfg in variables[1:]:
    var_name = var_cfg["name"]
    current_lags = var_arrays[var_name].lag.values
    if not np.array_equal(available_lags, current_lags):
        raise ValueError(
            f"Lag mismatch: '{variables[0]['name']}' has {available_lags}, "
            f"but '{var_name}' has {current_lags}"
        )

if plot_lags is None:
    selected_lags = [int(lag) for lag in available_lags]
else:
    available_lag_set = {int(lag) for lag in available_lags}
    selected_lags = [int(lag) for lag in plot_lags]
    missing_lags = [lag for lag in selected_lags if lag not in available_lag_set]
    if missing_lags:
        raise ValueError(
            f"Requested plot_lags not available in data: {missing_lags}. "
            f"Available lags: {sorted(available_lag_set)}"
        )

lags_str_out = "_".join(str(lag) for lag in selected_lags)
output_path = output_path.with_name(
    output_path.name.replace("_idx", f"_lags_{lags_str_out}_idx")
)

nrows = len(selected_lags)
ncols = len(variables)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(5 * ncols, 3.2 * nrows),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

if nrows == 1:
    axes = axes[np.newaxis, :]
if ncols == 1:
    axes = axes[:, np.newaxis]

ims = []
for col, var_cfg in enumerate(variables):
    var_name = var_cfg["name"]
    da = var_arrays[var_name]

    if "vmin" not in var_cfg or "vmax" not in var_cfg:
        raise KeyError(f"Please define 'vmin' and 'vmax' for variable '{var_name}' in variables.")
    cfg_vmin = var_cfg["vmin"]
    cfg_vmax = var_cfg["vmax"]

    if cfg_vmin is None or cfg_vmax is None:
        abs_max = np.abs(da.sel(index_val=index_value, lag=selected_lags)).max().item()
        vmin = -abs_max
        vmax = abs_max
    else:
        vmin = float(cfg_vmin)
        vmax = float(cfg_vmax)
    if vmin >= vmax:
        raise ValueError(f"Invalid color limits for '{var_name}': vmin ({vmin}) must be < vmax ({vmax}).")

    last_im = None
    for row, lag in enumerate(selected_lags):
        ax = axes[row, col]
        field = da.sel(lag=lag, index_val=index_value)

        last_im = field.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=var_cfg["cmap"],
            vmin=vmin,
            vmax=vmax,
            add_colorbar=False,
        )

        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)

        if row == 0:
            ax.set_title(
                f"({chr(97 + col)}) {var_cfg['label']}",
                fontsize=14,
                fontweight="bold",
                pad=12
            )
        if col == 0:
            ax.text(
                -0.1,
                0.5,
                f"Lag={lag}",
                transform=ax.transAxes,
                rotation=90,
                va="center",
                ha="right",
                fontweight="bold",
                fontsize=14,
            )

    ims.append(last_im)

fig.suptitle(
    f"{energy_name} {extreme_event} composites | "
    f"{region_ziel} | lags={lags_str_out}",
    fontsize=15,
    y=0.98,
)

fig.subplots_adjust(top=0.92, bottom=0.16, wspace=0.08, hspace=0.12)

for col, var_cfg in enumerate(variables):
    col_axes = axes[:, col]
    x0 = min(ax.get_position().x0 for ax in col_axes)
    x1 = max(ax.get_position().x1 for ax in col_axes)
    y0 = min(ax.get_position().y0 for ax in col_axes)

    cax = fig.add_axes([x0, y0 - 0.07, x1 - x0, 0.02])
    cbar = fig.colorbar(ims[col], cax=cax, orientation="horizontal")
    cbar.set_label(f"{var_cfg['label']} [{var_cfg['unit']}]", fontsize=12)

output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"[SUCCESS] Saved figure: {output_path}")
