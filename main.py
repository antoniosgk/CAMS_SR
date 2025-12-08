#IMPORT LIBRARIES
#%%
import os
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
# main.py
#%%
from vertical_indexing import metpy_find_level_index
from stations_utils import load_stations, select_station
from horizontal_indexing import nearest_grid_index
from file_utils import stations_path, species_file, T_file, pl_file
def main():
    idx=45
    name=None 
    stations = load_stations(stations_path)
    station = select_station(stations, idx)
    lat_s = float(station["Latitude"])
    lon_s = float(station["Longitude"])
    alt_s = float(station["Altitude"])
    ds_species=xr.open_dataset(species_file)
    ds_T = xr.open_dataset(T_file)
    ds_PL = xr.open_dataset(pl_file)
    print(f"\nSelected station: {name} (lat={lat_s}, lon={lon_s}, alt={alt_s} m)")
    print(ds_T)
    lats = ds_species['lat'].values
    lons = ds_species['lon'].values
    i,j= nearest_grid_index(lat_s,lon_s,lats,lons)
    print(f"\n The station falls into the grid cell with lat index= {i},lon index= {j}")
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


# %%
