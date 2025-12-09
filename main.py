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
from file_utils import stations_path, species_file, T_file, pl_file,species
#%%
def main():
    idx=45
    name=None 
    cell_nums = 8
    stations = load_stations(stations_path)
    station = select_station(stations, idx)
    lat_s = float(station["Latitude"])
    lon_s = float(station["Longitude"])
    alt_s = float(station["Altitude"])
    ds_species=xr.open_dataset(species_file)
    ds_T = xr.open_dataset(T_file)
    ds_PL = xr.open_dataset(pl_file)
    print(f"\nSelected station: {name} (lat={lat_s}, lon={lon_s}, alt={alt_s} m)")

    lats = ds_species['lat'].values
    lons = ds_species['lon'].values
    i,j= nearest_grid_index(lat_s,lon_s,lats,lons)
    print(f"\n The station falls into the grid cell with lat index= {i},lon index= {j}")
    if np.ndim(lats) == 1:
        Ny = lats.shape[0]
        Nx = lons.shape[0]
    else:
        Ny, Nx = lats.shape

    i1, i2 = max(0, i-cell_nums), min(Ny-1, i+cell_nums)
    j1, j2 = max(0, j-cell_nums), min(Nx-1, j+cell_nums)

    print(f"\nLoading domain subset: i={i1}:{i2}, j={j1}:{j2} for plotting")
    
    ds_big = ds_species

    # Coordinates
    lats_big = ds_big['lat'].values
    lons_big = ds_big['lon'].values

# --- CASE 1: 1D lat/lon ---
    if lats_big.ndim == 1 and lons_big.ndim == 1:

        ds_small = ds_big.isel({'lat': slice(i1, i2+1),
                            'lon': slice(j1, j2+1)})

        lats_small = lats_big[i1:i2+1]
        lons_small = lons_big[j1:j2+1]

# --- CASE 2: 2D lat/lon ---
    else:
       ds_small = ds_big.isel({'lat': slice(i1, i2+1),
                            'lon': slice(j1, j2+1)})

       lats_small = lats_big[i1:i2+1, j1:j2+1]
       lons_small = lons_big[i1:i2+1, j1:j2+1]

# variable extraction
    var_name = species
    #data_arr = ds_small[var_name].isel({'time': 0,
                                  #  'lev': vertical_idx}).values

# free memory
    #ds_big.close()
    #del ds_big
# local (station) indices in the small box
    ii = i - i1
    jj = j - j1
    # Extract local profiles
    T_prof = ds_T["T"].values[0, :, i, j]
    p_prof = ds_PL["PL"].values[0, :, i, j]  # Pa

    #  MetPy-based vertical level selection ---
    idx_level, p_level_hPa, z_level_m = metpy_find_level_index(
        p_prof_Pa=p_prof,
        T_prof_K=T_prof,
        station_alt_m=alt_s
    )

    print("Nearest model level:", idx_level)
    print("Pressure (hPa):", p_level_hPa)
    print("Height (m):", z_level_m)

if __name__ == "__main__":
    main()


# %%
