#!/usr/bin/env python3
"""Ground truth: is each grid point inside a Natural Earth 1:10m land polygon (v5.1.1, public domain)?"""
import csv
import os

import numpy as np
import shapefile
import shapely
from shapely.geometry import shape

from run_eval import LATS, LONS

HERE = os.path.dirname(os.path.abspath(__file__))
SHP = os.path.join(HERE, "../data/raw/ne_10m_land/ne_10m_land.shp")
LAKES = os.path.join(HERE, "../data/raw/ne_10m_lakes/ne_10m_lakes.shp")  # v5.0.0, reported only
OUT = os.path.join(HERE, "../data/landmask_ne10m_land_v5.1.1.csv")

geoms = [shape(s.__geo_interface__) for s in shapefile.Reader(SHP).shapes()]
land = shapely.union_all(geoms)
shapely.prepare(land)
lat = np.repeat(LATS, len(LONS)).astype(float)
lon = np.tile(LONS, len(LATS)).astype(float)
inside = shapely.contains_xy(land, lon, lat)
lakes = shapely.union_all([shape(s.__geo_interface__) for s in shapefile.Reader(LAKES).shapes()])
shapely.prepare(lakes)
in_lake = shapely.contains_xy(lakes, lon, lat)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["row", "col", "lat", "lon", "land", "in_ne10m_lake"])
    for k in range(len(lat)):
        w.writerow([k // len(LONS), k % len(LONS), int(lat[k]), int(lon[k]), int(inside[k]), int(in_lake[k])])
print(f"{OUT}: {inside.sum()} land of {len(inside)} points ({inside.mean():.4f}); "
      f"{(inside & in_lake).sum()} land points fall inside a 1:10m lake polygon (still counted as land)")
for name, la, lo in [("Caspian", 41, 51), ("Lake Superior", 47, -87), ("South Pole", -89, 1), ("Sahara", 23, 9),
                     ("Pacific", 1, -151), ("Greenland", 73, -41), ("Black Sea", 43, 35)]:
    k = LATS.index(la) * len(LONS) + LONS.index(lo)
    print(f"  {name:14s} ({la},{lo}) land={int(inside[k])}")
