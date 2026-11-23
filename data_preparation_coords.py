# data prep nuts to grid

import pandas as pd
import numpy as np
from shapely import Point
import geopandas as gpd
from pathlib import Path
from data_registry import Bibliography
import matplotlib.pyplot as plt

# -----Input Files----------
bib = Bibliography(Path("data"))
bib.add("energy", "demand",'daily','subcountry', "daily/H_ERA5_ECMW_T639_GHI_0000m_daily_subcountry.csv")
# bib.add("energy", "SPV",'daily','subcountry', "daily/H_ERA5_ECMW_T639_SPV_0000m_daily_subcountry_PhM02.csv")
bib.add("energy", "WON",'daily','subcountry', "daily/H_ERA5_ECMW_T639_WON_0100m_daily_subcountry_PhM01.csv")
bib.add("atmosphere", "WS10",'daily','subcountry', "daily/H_ERA5_ECMW_T639_WS-_0100m_daily_subcountry.csv")
bib.add("atmosphere", "WS100",'daily','subcountry', "daily/H_ERA5_ECMW_T639_WS-_0010m_daily_subcountry.csv")
bib.add("atmosphere", "MSL",'daily','subcountry', "daily/H_ERA5_ECMW_T639_MSL_0000m_daily_subcountry.csv")
bib.add("atmosphere", "TA",'daily','subcountry', "daily/H_ERA5_ECMW_T639_TA-_0002m_daily_subcountry.csv")
bib.add("atmosphere", "TP",'daily','subcountry', "daily/H_ERA5_ECMW_T639_TP-_0000m_daily_subcountry.csv")

nuts = gpd.read_file('data/NUTS_RG_20M_2021_3035.shp')
# print(nuts)
def main_polygon(geom):
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    return geom
gdf = nuts.copy()
# größtes Polygon auswählen
gdf["geometry_main"] = gdf.geometry.apply(main_polygon)

# Centroid berechnen
gdf["centroid"] = gdf["geometry_main"].centroid

# neues GeoDataFrame mit Punkten
centroids = gpd.GeoDataFrame(
    gdf[["NUTS_ID"]],
    geometry=gdf["centroid"],
    crs=gdf.crs
)
print(centroids)

for entry in bib.entries:
    path = entry.path
    df = pd.read_csv(path, comment="#", parse_dates=["Date"])
    
    df_long = df.melt(
        id_vars="Date",
        var_name="NUTS_ID",
        value_name=entry.name
    )
    centroids["lon"] = centroids.geometry.x
    centroids["lat"] = centroids.geometry.y

    df_final = df_long.merge(
        centroids[["NUTS_ID", "lon", "lat"]],
        on="NUTS_ID",
        how="left"
    )
    print(df_final.head())
    df_final.to_csv(f"data/prepared/{entry.name}_{entry.timestep}_{entry.grid}_prepared.csv", index=False)