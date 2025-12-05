#IMPORT LIBRARIES
import os
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
#%%
# main.py

import xarray as xr
from metpy_vertical import metpy_find_level_index     # ← IMPORT your custom function
from station_utils import load_stations, select_station
from horizontal_utils import nearest_grid_index
from file_utils import ds_T,ds_PL
def main():

    stations = load_stations("stations.txt")
    station = select_station(stations, idx=0)

    lat_s = float(station["Latitude"])
    lon_s = float(station["Longitude"])
    alt_s = float(station["Altitude"])

    ds_T = xr.open_dataset("T_file.nc4")
    ds_PL = xr.open_dataset("PL_file.nc4")

    # extract 1D profiles
    T_prof = ds_T["T"].values[0, :, i, j]
    p_prof = ds_PL["PL"].values[0, :, i, j]

    # call your vertical function
    idx, p_level, z_level = metpy_find_level_index(
        p_prof_Pa=p_prof,
        T_prof_K=T_prof,
        station_alt_m=alt_s
    )

    print("Nearest model level:", idx)
    print("Pressure (hPa):", p_level)
    print("Height (m):", z_level)


if __name__ == "__main__":
    main()

