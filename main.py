#IMPORT LIBRARIES
#%%
import os
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from metpy.constants import g 
from metpy.units import units

# main.py
#%%
from vertical_indexing import metpy_find_level_index
from stations_utils import load_stations, select_station, all_stations
from horizontal_indexing import nearest_grid_index
from file_utils import stations_path, species_file, T_file, pl_file,species,orog_file
from calculation import compute_sector_masks, sector_table
from plots import (plot_variable_on_map, plot_rectangles,plot_profile_P_T,
                   plot_profile_P_z, plot_profile_T_z,)
"""
This comment section includes all the variables,the functions,their names
and their physical interpretation
idx:index of the station of the stations file
name:name of the station of the stations file
cell_nums:number of cells around the central grid cell(the one the station falls into)
that will be plotted
lat_s
"""
#%%
def main():
    idx=1409 #index of station of the stations_file
    name=None #name of the station
    cell_nums = 14 #numb of cells that will plotted n**2
    d_zoom=1.0
    #-----------
    stations = load_stations(stations_path)
    station = select_station(stations, idx)
    lat_s = float(station["Latitude"]) #latitude of the station
    lon_s = float(station["Longitude"]) #longitude of the station
    alt_s = float(station["Altitude"]) #altitude of the station
    name=station["Station_Name"] #name of the station
    ds_species=xr.open_dataset(species_file) #nc file with species for specific t
    ds_T = xr.open_dataset(T_file) #nc file with temperature for the same t
    ds_PL = xr.open_dataset(pl_file) #nc file with pressure levels for the same t
    ds_orog = xr.open_dataset(orog_file) #nc file with orography
    print(f"\nSelected station: {name} (lat={lat_s}, lon={lon_s}, alt={alt_s} m)")

    #print("T dims:", ds_T["T"].dims)
    #print("PL dims:", ds_PL["PL"].dims)

    lats = ds_species['lat'].values #latitudes of the model
    lons = ds_species['lon'].values #longitudes of the model

    i,j= nearest_grid_index(lat_s,lon_s,lats,lons) #func that calculates the index the station falls into horizontally
    print(f"\n The station falls into the grid cell with lat index= {i},lon index= {j}")
    if np.ndim(lats) == 1:
        Ny = lats.shape[0]
        Nx = lons.shape[0]
    else:
        Ny, Nx = lats.shape

    i1, i2 = max(0, i-cell_nums), min(Ny-1, i+cell_nums) #subsets of lats (for plotting)
    j1, j2 = max(0, j-cell_nums), min(Nx-1, j+cell_nums) #subset of lons (for plotting)

    print(f"\nLoading domain subset: i={i1}:{i2}, j={j1}:{j2} for plotting")
    
    ds_big = ds_species #a copy of ds_species,maybe not needed,to reevalutate

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
    

 # free memory
    #ds_big.close()
    #del ds_big
# local (station) indices in the small box
    ii = i - i1
    jj = j - j1

    # ---- Orography: PHIS / SGH diagnostics and surface height ----
    PHIS_field = ds_orog["PHIS"] #surface geopotential height
    SGH_field = ds_orog["SGH"]  #isotropic stdv of GWD topography

    #print("PHIS dims:", PHIS_field.dims)
    #print("SGH dims:", SGH_field.dims)

    # Take PHIS / SGH at the same i, j as the station grid cell
    PHIS_val = PHIS_field.isel(lat=i, lon=j).item() #Surf Geopotential height of the gridcell
    SGH_val = SGH_field.isel(lat=i, lon=j).item()  #isotropic stdv of GWD of the gridcell

    # Basic range checks (global)
    #print(f"Surface Geopotential range (min, max):, {float(PHIS_field.min()).1f}, {float(PHIS_field.max()).1f")
    #print(f"SGH range (min, max):", float(SGH_field.min()), float(SGH_field.max()))
    print(f"Surface Geopotential at station grid cell (i={i}, j={j}):, {PHIS_val:.1f} m2/s2")
    print(f"SGH at station grid cell (i={i}, j={j}): , {SGH_val:.1f} m")
    # --- Heuristic: decide if PHIS is geopotential (m^2/s^2) or already height (m)
    #It is already geopotential (units m2/s2)
    # If PHIS is very large (order 1e5 or higher), assume geopotential and divide by g.
    if PHIS_val > 2:  # ~ g * 2000 m #2e4
        z_surf_model = (PHIS_val * units('m^2/s^2') / g).to('meter').magnitude #from geopotential to geop.height
        print("Interpreting PHIS as geopotential (m^2/s^2).")
    else:
        z_surf_model = PHIS_val
        print("Interpreting PHIS as geopotential height (m).")

    print(f"Model surface height at station grid cell: {z_surf_model:.1f} m")
    # Extract local profiles
    T_prof = ds_T["T"].values[0, :, i, j] #T profile for the specific gridcell
    p_prof = ds_PL["PL"].values[0, :, i, j]  # Pressure profile for the specific gridcell
    #T_prof = ds_T["T"].isel(time=0, lat=i, lon=j).values   #see if it is better this
    #p_prof = ds_PL["PL"].isel(time=0, lat=i, lon=j).values #or the above
    #  MetPy-based vertical level selection ---
    idx_level, p_level_hPa, z_level_m = metpy_find_level_index(
        p_prof_Pa=p_prof,
        T_prof_K=T_prof,
        station_alt_m=alt_s,
        z_surf_model=z_surf_model
    )

    print(f"Nearest model level:", idx_level)
    print(f"Pressure (hPa):, {p_level_hPa:.2f}")
    print(f"Height (m):, {z_level_m:.2f}")
    data_arr = ds_small[var_name].isel({'time': 0,
                                   'lev': idx_level}).values
    
    '''fig1, ax1, im1 = plot_variable_on_map(
    lats_small,
    lons_small,
    data_arr,
    lon_s,
    lat_s,
    units=units,
    species_name=species,
    d=d_zoom
    )'''
    
    fig2, ax2, im2 = plot_variable_on_map(
    lats_small,
    lons_small,
    data_arr,
    lon_s,
    lat_s,
    units=units,
    species_name=species,
    d=d_zoom
    )

    plot_rectangles(
    ax2,
    lats_small,
    lons_small,
    ii,
    jj,
    im2,
    units=units,
    species_name=species,
    )

    plt.show()
    fig_PT, ax_PT = plot_profile_P_T(p_prof, T_prof)

# P–z
    #fig_Pz, ax_Pz = plot_profile_P_z(p_prof, z_prof, z_units="km")

# T–z
    #fig_Tz, ax_Tz = plot_profile_T_z(T_prof, z_prof, z_units="km")




    # --- compute and print levels for *all* stations ---
    #all_stations()
#%%

if __name__ == "__main__":
    main()


# %%
