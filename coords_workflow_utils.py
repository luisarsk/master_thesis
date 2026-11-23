from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from regionalmeancalculator import RegionalMeanCalculator


class PreparedCoordsLoader:
    """Utilities to load prepared coords CSV files into event/predictor xarray objects."""

    @staticmethod
    def load_energy_event_from_prepared(
        path: Path,
        energy_name: str,
        region_ziel: str | None,
        quant: float,
        extreme_event: str,
    ) -> xr.DataArray:
        """
        Load prepared energy data (long format) and return binary event series.

        Expected CSV format:
        Date, NUTS_ID, <energy_name>, lon, lat
        """
        df = pd.read_csv(path, parse_dates=["Date"])

        if energy_name not in df.columns:
            raise ValueError(f"Spalte '{energy_name}' fehlt in {path}")

        df_wide = (
            df.pivot_table(index="Date", columns="NUTS_ID", values=energy_name, aggfunc="mean")
            .sort_index()
        )

        if region_ziel is not None:
            calculator = RegionalMeanCalculator()
            series = calculator.calculate_mean(df=df_wide, region_name=region_ziel)
        else:
            series = df_wide.mean(axis=1)
            series.name = energy_name

        if extreme_event == "l":
            threshold = series.quantile(quant)
            binary = (series < threshold).astype(int)
        else:
            threshold = series.quantile(quant)
            binary = (series > threshold).astype(int)

        event_da = xr.DataArray(
            binary.to_numpy(),
            dims="time",
            coords={"time": binary.index.to_numpy()},
            name=energy_name,
        )
        return event_da

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

    @staticmethod
    def load_atmosphere_from_prepared(path: Path, atm_name: str) -> xr.Dataset:
        """
        Load prepared atmosphere data (long format) and return Dataset with dims time x point.

        Expected CSV format:
        Date, NUTS_ID, <atm_name>, lon, lat
        """
        df = pd.read_csv(path, parse_dates=["Date"])

        required = {"Date", "NUTS_ID", atm_name, "lon", "lat"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Fehlende Spalten in {path}: {sorted(missing)}")

        point_table = (
            df[["NUTS_ID", "lon", "lat"]]
            .drop_duplicates(subset=["NUTS_ID"])
            .sort_values("NUTS_ID")
            .reset_index(drop=True)
        )
        point_table["point"] = range(len(point_table))

        df_values = df[["Date", "NUTS_ID", atm_name]].merge(
            point_table[["NUTS_ID", "point"]], on="NUTS_ID", how="left"
        )

        matrix = (
            df_values.pivot_table(index="Date", columns="point", values=atm_name, aggfunc="mean")
            .sort_index()
            .sort_index(axis=1)
        )

        da = xr.DataArray(
            matrix.values,
            dims=("time", "point"),
            coords={
                "time": matrix.index.to_numpy(),
                "point": np.asarray(matrix.columns),
            },
            name=atm_name,
        )

        da = da.assign_coords(
            NUTS_ID=("point", point_table.set_index("point").loc[matrix.columns, "NUTS_ID"].values),
            lon=("point", point_table.set_index("point").loc[matrix.columns, "lon"].values),
            lat=("point", point_table.set_index("point").loc[matrix.columns, "lat"].values),
        )

        return da.to_dataset(name=atm_name)


class NUTSMaskTools:
    """Mask diagnostics, adjacency filtering and plotting for point/NUTS based datasets."""

    @staticmethod
    def print_mask_coverage(mask_ds: xr.Dataset, label: str) -> None:
        for var in mask_ds.data_vars:
            vals = mask_ds[var].values
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                print(f"[INFO] {label} - {var}: no finite values")
                continue
            frac = float(np.mean(vals > 0))
            print(f"[INFO] {label} - {var}: {frac:.2%} unmasked")

    @staticmethod
    def _build_nuts_adjacency(nuts_gdf: gpd.GeoDataFrame) -> dict[str, set[str]]:
        gdf = nuts_gdf[["NUTS_ID", "geometry"]].drop_duplicates("NUTS_ID").reset_index(drop=True)
        sindex = gdf.sindex

        adjacency = {str(nid): set() for nid in gdf["NUTS_ID"].astype(str)}

        for i, geom in enumerate(gdf.geometry):
            nid_i = str(gdf.loc[i, "NUTS_ID"])
            candidates = list(sindex.query(geom, predicate="intersects"))
            for j in candidates:
                if i == j:
                    continue
                nid_j = str(gdf.loc[j, "NUTS_ID"])
                adjacency[nid_i].add(nid_j)
        return adjacency

    @staticmethod
    def apply_nuts_graph_area_mask(
        mask_ds: xr.Dataset,
        min_region_size: int,
        nuts_gdf: gpd.GeoDataFrame,
    ) -> xr.Dataset:
        adjacency = NUTSMaskTools._build_nuts_adjacency(nuts_gdf)
        out = mask_ds.copy(deep=True).astype(int)

        for var in out.data_vars:
            da = out[var]
            if "point" not in da.dims or "NUTS_ID" not in da.coords:
                continue

            other_dims = [d for d in da.dims if d != "point"]
            da_t = da.transpose("point", *other_dims)
            vals = da_t.values
            vals_out = np.zeros_like(vals, dtype=int)
            point_ids = [str(x) for x in da_t["NUTS_ID"].values]

            if vals.ndim == 1:
                iter_idx = [()]
            else:
                iter_idx = list(np.ndindex(vals.shape[1:]))

            for idx in iter_idx:
                slicer = (slice(None),) + idx
                active_mask = vals[slicer].astype(bool)
                active_ids = {point_ids[k] for k, is_on in enumerate(active_mask) if is_on}

                keep_ids = set()
                visited = set()
                for node in active_ids:
                    if node in visited:
                        continue
                    stack = [node]
                    comp = set()
                    while stack:
                        cur = stack.pop()
                        if cur in visited:
                            continue
                        visited.add(cur)
                        comp.add(cur)
                        for nb in adjacency.get(cur, set()):
                            if nb in active_ids and nb not in visited:
                                stack.append(nb)

                    if len(comp) >= min_region_size:
                        keep_ids.update(comp)

                vals_out[slicer] = np.array([1 if pid in keep_ids else 0 for pid in point_ids], dtype=int)

            da_new = xr.DataArray(vals_out, coords=da_t.coords, dims=da_t.dims).transpose(*da.dims)
            out[var] = da_new

        return out

    @staticmethod
    def plot_mask_as_map(
        mask_ds: xr.Dataset,
        label: str,
        energy_name: str,
        atm_name: str,
        region_ziel: str,
        region_prec: str,
        timeperiod: str,
        nuts_shapefile: str = "data/NUTS_RG_20M_2021_3035.shp",
    ) -> None:
        data_var_name = list(mask_ds.data_vars)[0]
        da_plot = mask_ds[data_var_name].astype(float)

        nuts = gpd.read_file(nuts_shapefile)

        if "lag" in da_plot.dims:
            lags_plot = da_plot["lag"].values.tolist()
        else:
            lags_plot = [None]

        if "index_val" in da_plot.dims:
            index_vals_plot = da_plot["index_val"].values.tolist()
        else:
            index_vals_plot = [None]

        fig, axes = plt.subplots(
            nrows=len(lags_plot),
            ncols=len(index_vals_plot),
            figsize=(4.8 * len(index_vals_plot), 4.2 * len(lags_plot)),
            constrained_layout=True,
        )

        if len(lags_plot) == 1 and len(index_vals_plot) == 1:
            axes = np.array([[axes]])
        elif len(lags_plot) == 1:
            axes = np.array([axes])
        elif len(index_vals_plot) == 1:
            axes = np.array([[ax] for ax in axes])

        for i, lag in enumerate(lags_plot):
            for j, idx in enumerate(index_vals_plot):
                ax = axes[i, j]
                nuts.plot(ax=ax, color="lightgrey", edgecolor="black", linewidth=0.2)

                if lag is None and idx is None:
                    sel = da_plot
                    title = label
                elif idx is None:
                    sel = da_plot.sel(lag=lag)
                    title = f"{label} | lag={lag}"
                elif lag is None:
                    sel = da_plot.sel(index_val=idx)
                    title = f"{label} | index_val={idx}"
                else:
                    sel = da_plot.sel(lag=lag, index_val=idx)
                    title = f"{label} | lag={lag}, index_val={idx}"

                df_sel = sel.to_dataframe(name="mask").reset_index()
                if "NUTS_ID" not in df_sel.columns:
                    print(f"[WARN] NUTS_ID not found for panel: {title}")
                    ax.set_title(title)
                    ax.set_axis_off()
                    continue

                df_sel["NUTS_ID"] = df_sel["NUTS_ID"].astype(str)
                gdf_sel = nuts.merge(df_sel[["NUTS_ID", "mask"]], on="NUTS_ID", how="left")

                if gdf_sel["mask"].notna().any():
                    gdf_sel_nonan = gdf_sel[gdf_sel["mask"].notna()]
                    gdf_sel_nonan.plot(
                        column="mask",
                        cmap="viridis",
                        vmin=0,
                        vmax=1,
                        ax=ax,
                        edgecolor="black",
                        linewidth=0.2,
                        legend=False,
                    )

                ax.set_xlim(2500000, 7300000)
                ax.set_ylim(1500000, 5500000)
                ax.set_title(title)
                ax.set_axis_off()

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
        sm._A = []
        cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.02, pad=0.01)
        cbar.set_label("mask value (0/1)")
        # plt.show()
        plt.savefig(
            f"fig/index_computation_coords/mask_{label}_{energy_name}_{atm_name}_{region_ziel}_{region_prec}_{timeperiod}.png",
            dpi=300,
            bbox_inches="tight",
        )
