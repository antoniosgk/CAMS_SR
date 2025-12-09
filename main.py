#IMPORT LIBRARIES
#%%
import os
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
from metpy.constants import g 
from metpy.units import units

# main.py
#%%
from vertical_indexing import metpy_find_level_index
from stations_utils import load_stations, select_station
from horizontal_indexing import nearest_grid_index
from file_utils import stations_path, species_file, T_file, pl_file,species,orog_file
#%%
def main():
    idx=1409
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
    ds_orog = xr.open_dataset(orog_file) 
    print(f"\nSelected station: {name} (lat={lat_s}, lon={lon_s}, alt={alt_s} m)")

    print("T dims:", ds_T["T"].dims)
    print("PL dims:", ds_PL["PL"].dims)

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

    # ---- Orography: PHIS / SGH diagnostics and surface height ----
    PHIS_field = ds_orog["PHIS"]
    SGH_field = ds_orog["SGH"]

    print("PHIS dims:", PHIS_field.dims)
    print("SGH dims:", SGH_field.dims)

    # Take PHIS / SGH at the same i, j as the station grid cell
    PHIS_val = PHIS_field.isel(lat=i, lon=j).item()
    SGH_val = SGH_field.isel(lat=i, lon=j).item()

    # Basic range checks (global)
    print("PHIS range (min, max):", float(PHIS_field.min()), float(PHIS_field.max()))
    print("SGH range (min, max):", float(SGH_field.min()), float(SGH_field.max()))
    print(f"PHIS at station grid cell (i={i}, j={j}):", PHIS_val)
    print(f"SGH at station grid cell (i={i}, j={j}):", SGH_val)

    # --- Heuristic: decide if PHIS is geopotential (m^2/s^2) or already height (m)
    # If PHIS is very large (order 1e5 or higher), assume geopotential and divide by g.
    # Otherwise, treat as meters. Adjust this logic for your dataset if needed.
    if PHIS_val > 2e4:  # ~ g * 2000 m
        z_surf_model = (PHIS_val * units('m^2/s^2') / g).to('meter').magnitude
        print("Interpreting PHIS as geopotential (m^2/s^2).")
    else:
        z_surf_model = PHIS_val
        print("Interpreting PHIS as geopotential height (m).")

    print(f"Model surface height at station grid cell: {z_surf_model:.1f} m")
    # Extract local profiles
    T_prof = ds_T["T"].values[0, :, i, j]
    p_prof = ds_PL["PL"].values[0, :, i, j]  # Pa
    #T_prof = ds_T["T"].isel(time=0, lat=i, lon=j).values   #see if it is better this
    #p_prof = ds_PL["PL"].isel(time=0, lat=i, lon=j).values #or the above
    #  MetPy-based vertical level selection ---
    idx_level, p_level_hPa, z_level_m = metpy_find_level_index(
        p_prof_Pa=p_prof,
        T_prof_K=T_prof,
        station_alt_m=alt_s,
        z_surf_model=z_surf_model
    )

    print("Nearest model level:", idx_level)
    print("Pressure (hPa):", p_level_hPa)
    print("Height (m):", z_level_m)

if __name__ == "__main__":
    main()


# %%
