## run compositing with coords-event + WeatherBench dailymeans predictor ##
######################################################################

from pathlib import Path
import datetime as dt
import warnings
import numpy as np
import os

import xarray as xr
from domino import agg
from domino.core import LaggedAnalyser, PatternFilter, IndexGenerator

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from dask.distributed import Client, LocalCluster
except Exception:
    Client = None
    LocalCluster = None

from data_registry import Bibliography
from coords_workflow_utils import PreparedCoordsLoader
from itertools import product
import gc
warnings.filterwarnings("ignore")

dask_client = None
dask_cluster = None


# -----INPUT FILES----------
bib = Bibliography(Path("data"))
bib.add("energy", "SPV", "daily", "subcountry", "prepared/SPV_daily_subcountry_prepared_DJF.csv")
#bib.add("energy", "WON", "daily", "subcountry", "prepared/WON_daily_subcountry_prepared_DJF.csv")
# bib.add('atmosphere', '10m_wind_speed', 'daily', 'global', 'C:/Users/peter/Desktop/UNI/Masterarbeit/program_copy/data/ERA5/10m_wind_speed')
#bib.add('atmosphere', 'wind_speed', 'daily', 'global', 'ERA5/prepared/DJF_anom/wind_speed')
#bib.add('atmosphere', 'geopotential', 'daily', 'global', 'ERA5/prepared/DJF_anom/geopotential500')
bib.add('atmosphere', 'mean_sea_level_pressure', 'daily', 'global', 'ERA5/prepared/DJF_anom/mean_sea_level_pressure')
#bib.add('atmosphere', 'v_component_of_wind', 'daily', 'global', 'ERA5/prepared/DJF_anom/v_component_of_wind')
#bib.add('atmosphere', 'total_column_water_vapour', 'daily', 'global', 'ERA5/prepared/DJF_anom/total_column_water_vapour')
#bib.add('atmosphere', '2m_temperature', 'daily', 'global', 'ERA5/prepared/DJF_anom/2m_temperature')
#bib.add('atmosphere', 'relative_humidity', 'daily', 'global', 'ERA5/prepared/DJF_anom/relative_humidity700')

energy_entries = bib.get_by_category("energy")
atm_entries = bib.get_by_category("atmosphere")
for energy_entry, atm_entry in product(energy_entries, atm_entries):
    # Predictor dataset created by test_compute_weatherbench_dailymeans.py
    ATM_NAME = atm_entry.name # "geopotential", "mean_sea_level_pressure", "2m_temperature"
    LEVEL = '' # '' for non level data
    WEATHERBENCH_DAILY_PATH = atm_entry.path
    print(WEATHERBENCH_DAILY_PATH)
    print("[INFO] Merging all years into a single dataset...")
    files = sorted(WEATHERBENCH_DAILY_PATH.glob("*.nc")) #era5_daily_
    print(files)
    # print(files)
    ds_all = xr.open_mfdataset(
        files, combine="by_coords",
        chunks={"time": 50, "latitude":50, "longitude":50} # changed time from 365 for deseason
    )
    print("[INFO] Loaded and merged WeatherBench dailymeans dataset")

    # print(ds_all)

    ds_djf = ds_all.sel(time=ds_all.time.dt.month.isin([12, 1, 2]))#ds_all.sel(init_time=ds_all.init_time.dt.month.isin([12, 1, 2]))
    ds_all = ds_djf
    month_idx = '_DJF_anom'
    # # print(ds_all)

    # -----VARIABLES-----------
    erstelldatum = dt.date.today()
    lags = [-4,-3,-2,-1, 0]
    region_ziel = "Norddeutschland"
    region_prec = "EUROPE"
    timeperiod = "1979"
    quant = 0.1
    extreme_event = "l"  # l=low, h=high

    # -----OPTIONS-----------
    use_deseasonalization = True
    deseasonalization_agg = "dayofyear"
    deseasonalization_smooth = 30
    use_bootstrap = True
    bootstrap_num = 100
    bootstrap_p = 0.05
    bootstrap_reuse_lag0_boots = True
    bootstrap_batch_size = 16
    bootstrap_use_gpu = False
    bootstrap_show_progress = True
    use_dask_parallel = False
    dask_n_workers = max(1, (os.cpu_count() or 4) // 2)
    dask_threads_per_worker = 2
    dask_use_processes = False
    use_composites_cache = True
    force_recompute_composites = False

    if use_dask_parallel and (Client is not None) and (dask_client is None):
        print(
            f"[INFO] Starting Dask LocalCluster with {dask_n_workers} workers x "
            f"{dask_threads_per_worker} threads (processes={dask_use_processes})..."
        )
        try:
            dask_cluster = LocalCluster(
                n_workers=dask_n_workers,
                threads_per_worker=dask_threads_per_worker,
                processes=dask_use_processes,
                dashboard_address=None,
            )
            dask_client = Client(dask_cluster)
            print("[INFO] Dask parallel backend active")
        except Exception as e:
            print(f"[WARN] Dask startup failed ({e}). Continuing without Dask parallel backend.")
            dask_client = None
            dask_cluster = None

    energy_entries = bib.get_by_category("energy")

    # -----MAIN LOOP---------
    energy_iter = energy_entries if tqdm is None else tqdm(energy_entries, desc="energy", unit="series")
    for energy_entry in energy_iter:
        energy_name = energy_entry.name
        energy_grid = energy_entry.grid
        energy_timestep = energy_entry.timestep
        energy_path = energy_entry.path

        print("===================================")
        print(f"Energy: {energy_name} ({energy_grid}, {energy_timestep})")
        print(f"Atmosphere dataset: {ATM_NAME} (WeatherBench dailymeans)")
        print("Energy path:", energy_path)
        print("WB path:", WEATHERBENCH_DAILY_PATH)
        print("===================================")

        # --------EVENT: build binary event series from prepared coords energy data --------
        df_ene = PreparedCoordsLoader.load_energy_event_from_prepared(
            path=energy_path,
            energy_name=energy_name,
            region_ziel=region_ziel,
            quant=quant,
            extreme_event=extreme_event,
        )

        # align predictor to event timeline (intersection on time)
        wb_ds = ds_all
        print(wb_ds)
        df_atm_xr = xr.align(wb_ds, df_ene.to_dataset(name="event"), join="inner")[0]
        # df_atm_xr = df_atm_xr.rename({'init_time': 'time'})
        print(df_atm_xr)

        # --------LAGGED ANALYSER --------
        print("[INFO] Initializing LaggedAnalyser with event and predictor...")
        analyser = LaggedAnalyser(event=df_ene, variables=df_atm_xr, name=ATM_NAME)
        # analyser.variables.load() # zu langsam nicht genug memory
        names = list(analyser.variables.data_vars)
        print(f"[INFO] Variables in analyser: {names}")

        # --------OPTION 1: DESEASONALIZATION--------
      #  if use_deseasonalization:
    ##        print(f"[INFO] Deseasonalizing data (agg={deseasonalization_agg}, smooth={deseasonalization_smooth})...")
   #         analyser.deseasonalise_variables(variable_list=names, dim='time', agg=deseasonalization_agg, smooth=deseasonalization_smooth)
  #          mean_state = analyser.aggregate_variables()

        # --------LAG VARIABLES --------
        print(f"[INFO] Lagging variables with lags: {lags}...")
        analyser.lag_variables(lags, offset_unit="days", mode="past")

        # --------CACHE PATHS / OUTPUT PATHS--------
        output_base_dir = "data/domino_output_coords_weatherbench"
        output_dir = Path(output_base_dir) / energy_name 
        output_dir.mkdir(parents=True, exist_ok=True)

        lags_str = "_".join(str(lag) for lag in lags)
        deseasonalization_suffix = "_deseasonalized" if use_deseasonalization else ""

        output_filename_composites = (
            f"{energy_name}-{extreme_event}_{region_ziel}_{ATM_NAME}{LEVEL}_{region_prec}_{timeperiod}"
            f"_lags_{lags_str}_composites_coords{deseasonalization_suffix}{month_idx}.nc" 
        )
        output_path_composites = output_dir / output_filename_composites

        output_filename_lagged_means = (
            f"{energy_name}-{extreme_event}_{region_ziel}_{ATM_NAME}{LEVEL}_{region_prec}_{timeperiod}"
            f"_lags_{lags_str}_lagged-means_coords{deseasonalization_suffix}{month_idx}.nc"
        )
        output_path_lagged_means = output_dir / output_filename_lagged_means

        cache_available = (
            use_composites_cache
            and (not force_recompute_composites)
            and output_path_composites.exists()
            and output_path_lagged_means.exists()
        )

        # --------COMPUTE OR LOAD COMPOSITES --------
        if cache_available:
            print(f"[INFO] Loading cached composites: {output_path_composites}")
            composites = xr.open_dataset(str(output_path_composites), engine="netcdf4")
            lagged_means = xr.open_dataset(str(output_path_lagged_means), engine="netcdf4")

            # Restore analyser state needed by bootstrap_significance.
            analyser.composites = composites
            analyser.lagged_means = lagged_means
            analyser.composites_are_anomaly = True
            analyser.composite_func = (agg.mean_ds, agg.cat_occ_ds)
        else:
            print("[INFO] Computing composites...")
            composites = analyser.compute_composites(as_anomaly=True)

            # Cache both composites and lagged means so future runs can skip recomputation.
            composites.to_netcdf(str(output_path_composites), engine="netcdf4", format="NETCDF4")
            analyser.lagged_means.to_netcdf(str(output_path_lagged_means), engine="netcdf4", format="NETCDF4")
            print(f"[SUCCESS] Saved composites cache: {output_path_composites}")
            print(f"[SUCCESS] Saved lagged means cache: {output_path_lagged_means}")

        composites.attrs["Erstelldatum"] = str(erstelldatum)
        composites.attrs["Extreme event"] = extreme_event + ", where l: low (quantile), h: high (quantile)"
        composites.attrs["Quantile"] = str(quant)
        composites.attrs["Method"] = "compute composites with WeatherBench dailymeans predictor"
        composites.attrs["Deseasonalized"] = str(use_deseasonalization)
        composites.attrs["Lags"] = str(lags)
        composites.attrs["Region Ziel"] = region_ziel
        composites.attrs["Region Precursor"] = region_prec
        composites.attrs["Predictor Source"] = str(WEATHERBENCH_DAILY_PATH)
        if not cache_available:
            # File already written during cache creation above.
            print(f"[SUCCESS] Saved composites: {output_path_composites}")

        # --------OPTION 2: BOOTSTRAP SIGNIFICANCE TEST--------
        if use_bootstrap:
            print(f"[INFO] Running bootstrap significance test (bootnum={bootstrap_num}, p={bootstrap_p})...")
            sig_mask = analyser.bootstrap_significance(
                bootnum=bootstrap_num,
                p=bootstrap_p,
                hb_correction=False,
                reuse_lag0_boots=bootstrap_reuse_lag0_boots,
                batch_size=bootstrap_batch_size,
                use_gpu=bootstrap_use_gpu,
                show_progress=bootstrap_show_progress,
                progress_label=f"{energy_name}-{ATM_NAME}",
            )
            print("[INFO] Applying significance mask on CPU...")
            masked_composites = composites.where(sig_mask == 1)
            
            output_filename_masked = f"{energy_name}-{extreme_event}_{region_ziel}_{ATM_NAME}{LEVEL}_{region_prec}_{timeperiod}_lags_{lags_str}_bootstrap-composites{deseasonalization_suffix}{month_idx}.nc"
            output_path_masked = output_dir / str(bootstrap_num) / output_filename_masked
            output_path_masked.parent.mkdir(parents=True, exist_ok=True)
            
            masked_composites.attrs['Erstelldatum'] = str(erstelldatum)
            masked_composites.attrs['Extreme event'] = extreme_event + ', where l: low (quantile), h: high (quantile)'
            masked_composites.attrs['Quantile'] = str(quant)
            masked_composites.attrs['Method'] = f'compute masked composites (bootstrap: bootnum={bootstrap_num}, p={bootstrap_p})'
            masked_composites.attrs['Deseasonalized'] = str(use_deseasonalization)
            masked_composites.attrs['Lags'] = str(lags)
            masked_composites.attrs['Region Ziel'] = region_ziel
            masked_composites.attrs['Region Precursor'] = region_prec
            masked_composites.attrs['Predictor Source'] = str(WEATHERBENCH_DAILY_PATH)
            masked_composites.to_netcdf(str(output_path_masked), engine="netcdf4", format="NETCDF4")
            print(f"[SUCCESS] Saved masked composites: {output_path_masked}")
        else:
            print("[INFO] Bootstrap significance test skipped (use_bootstrap=False)")

        

        del df_ene
        del df_atm_xr
        del analyser
        del composites
        del wb_ds

        gc.collect()
        print('[INFO] Finished processing, cleared variables and collected garbage.')
        print()

if dask_client is not None:
    dask_client.close()
if dask_cluster is not None:
    dask_cluster.close()
