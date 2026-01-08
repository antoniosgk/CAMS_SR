#IMPORT LIBRARIES
#%%
import os
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from metpy.constants import g 
from metpy.units import units as mp_units

# main.py
#%%
from vertical_indexing import metpy_find_level_index,metpy_compute_heights
from stations_utils import load_stations, select_station, all_stations
from horizontal_indexing import nearest_grid_index
from file_utils import stations_path, species_file, T_file, pl_file,species,orog_file,RH_file
from calculation import compute_sector_masks, sector_table
from plots import (plot_variable_on_map,plot_rectangles,
    plot_profile_P_T,
    plot_profile_T_Z,
    plot_profile_T_logP,
    plot_profile_species_logP,
    plot_profile_species_Z,save_figure)
"""
/
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
    idx=2 #index of station of the stations_file
    name=None #name of the station
    cell_nums = 4 #numb of cells that will plotted n**2
    d_zoom=0.8 #zoom of plots
    out_dir="/home/agkiokas/CAMS/plots/" #where the plots are saved
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
    ds_RH=xr.open_dataset(RH_file)
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
        z_surf_model = (PHIS_val * mp_units('m^2/s^2') / g).to('meter').magnitude #from geopotential to geop.height
        print("Interpreting PHIS as geopotential (m^2/s^2).")
    else:
        z_surf_model = PHIS_val
        print("Interpreting PHIS as geopotential height (m).")

    print(f"Model surface height at station grid cell: {z_surf_model:.1f} m")
    # Extract local profiles
    T_prof = ds_T["T"].values[0, :, i, j] #T profile for the specific gridcell
    p_prof = ds_PL["PL"].values[0, :, i, j]  # Pressure profile for the specific gridcell
    species_prof= ds_species[species].values[0,:,i,j] #here i must put species or var!!!
    RH_prof = ds_RH['RH'].values[0,:,i,j]
    #T_prof = ds_T["T"].isel(time=0, lat=i, lon=j).values   #see if it is better this
    #p_prof = ds_PL["PL"].isel(time=0, lat=i, lon=j).values #or the above
    #  MetPy-based vertical level selection --- metpy_find_level_index
    idx_level, p_level_hPa, z_level_m = metpy_find_level_index(
        p_prof_Pa=p_prof,
        T_prof_K=T_prof,
        RH=RH_prof,
        station_alt_m=alt_s,
        z_surf_model=z_surf_model
    )
    
    print(f"Nearest model level:", idx_level)
    print(f"Pressure (hPa):, {p_level_hPa:.2f}")
    print(f"Height (m):, {z_level_m:.2f}")

    z_prof = metpy_compute_heights(
    p_prof_Pa=p_prof,
    T_prof_K=T_prof,
    RH=RH_prof,
    z0=z_surf_model,
)
    
    ''' #uncomment if we have 2d data
    # T_box, P_box, RH_box: shape (lev, Ny_box, Nx_box)
    T_box = ds_T["T"].isel(time=0, lat=slice(i1, i2+1), lon=slice(j1, j2+1)).values
    P_box = ds_PL["PL"].isel(time=0, lat=slice(i1, i2+1), lon=slice(j1, j2+1)).values
    RH_box=ds_RH["RH"].isel(time=0, lat=slice(i1, i2+1), lon=slice(j1, j2+1)).values
    nlev, Ny_box, Nx_box = T_box.shape
    ncol = Ny_box * Nx_box

    T_flat = T_box.reshape(nlev, ncol)
    P_flat = P_box.reshape(nlev, ncol)
    RH_flat= RH_box.reshape(nlev,ncol)
# --- PHIS for the same box (Ny_box, Nx_box) ---
    PHIS_box = ds_orog["PHIS"].isel(time=0, lat=slice(i1, i2+1), lon=slice(j1, j2+1)).values

# Convert PHIS -> surface height per cell (ASL), then flatten
# (Use your preferred heuristic; below matches your earlier intention.)
    if np.nanmax(PHIS_box) > 2:
        z_surf_box = (PHIS_box * mp_units("m^2/s^2") / g).to("meter").magnitude
    else:
        z_surf_box = PHIS_box

    z_surf_flat = z_surf_box.reshape(ncol)   # shape (ncol,)

# Now run vertical indexing for every column
    idx_levels, p_levels_hPa, z_levels_m = metpy_find_level_index(
        p_prof_Pa=P_flat,          # (nlev, ncol)
        T_prof_K=T_flat,
        RH=RH_flat,                      # (nlev, ncol)
        station_alt_m=alt_s,       # scalar station altitude ASL
        z_surf_model=z_surf_flat,  # (ncol,) surface height ASL per column
)

# Reshape back to (Ny_box, Nx_box)
    idx_levels_2d = idx_levels.reshape(Ny_box, Nx_box)
    p_levels_hPa_2d = p_levels_hPa.reshape(Ny_box, Nx_box)
    z_levels_m_2d = z_levels_m.reshape(Ny_box, Nx_box)

    '''

    data_var = ds_small[species]          # e.g. species = "O3"
    units = data_var.attrs.get("units", "")
    

    # choose time index
    tidx = 0
    time_val = data_var["time"].values[tidx]
    # quick, generic string:
    time_str = pd.to_datetime(time_val).strftime("%Y-%m-%d %H:%M")
    data_arr = ds_small[var_name].isel({'time': 0,
                                   'lev': idx_level}).values
    
    meta = {
    "station_name": name,
    "station_lat": lat_s,
    "station_lon": lon_s,
    "station_alt": alt_s,
    "model_lat": float(lats[i]) if np.ndim(lats) == 1 else float(lats[i, j]),
    "model_lon": float(lons[j]) if np.ndim(lons) == 1 else float(lons[i, j]),
    "model_level": int(idx_level),
    "model_p_hPa": float(p_level_hPa),
    "z_level_m":float(z_level_m),
    "time_str": time_str,
    "species": species,
    "units": units,
    }
    #next 4 lines regard the O3 and its conversion to ppb
    MW_O3 = 48.0
    MW_air = 28.9647
    data_arr_ppb = data_arr * (MW_air / MW_O3) * 1e9
    species_prof_ppb = species_prof * (MW_air / MW_O3) * 1e9
    units_ppb = "ppb"
    meta["units"] = units_ppb
    #--------------
    fig1, ax1, im1 = plot_variable_on_map(
    lats_small,
    lons_small,
    data_arr_ppb,
    lon_s,
    lat_s,
    units=units_ppb,
    species_name=species,
    d=d_zoom,
    time_str=time_str,
    meta=meta
    )
    
    fig2, ax2, im2 = plot_variable_on_map(
    lats_small,
    lons_small,
    data_arr,
    lon_s,
    lat_s,
    units=units,
    species_name=species,
    d=d_zoom,
    time_str=time_str,
    meta=meta
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
    time_str=time_str,
    meta=meta
    )

    # P–T
    fig_PT, ax_PT = plot_profile_P_T(p_prof, T_prof, idx_level, time_str=time_str,meta=meta)
    plt.show()
    # T–Z
    fig_TZ, ax_TZ = plot_profile_T_Z(T_prof, z_prof, idx_level,
                                 time_str=time_str, z_units="km",meta=meta)

    # T–logP
    fig_TlogP, ax_TlogP = plot_profile_T_logP(p_prof, T_prof, idx_level,
                                          time_str=time_str,meta=meta)

# species–logP
    fig_SlogP, ax_SlogP = plot_profile_species_logP(
     p_prof,
     species_prof_ppb,
     idx_level,
     species_name=species,
     species_units=units,
     time_str=time_str,
     meta=meta
    )
    

# species–Z
    fig_SZ, ax_SZ = plot_profile_species_Z(
    z_prof,
    species_prof,
    idx_level,
    species_name=species,
    species_units=units,
    time_str=time_str,
    z_units="km",
    meta=meta
)

    plt.show()
    
    save_figure(fig1, out_dir, f"map_{species}_{name}_{time_str}")
    save_figure(fig2, out_dir, f"map_with sectors_{species}_{name}_{time_str}")
    save_figure(fig_PT, out_dir, f"map_P_T_{time_str}")
    save_figure(fig_TlogP, out_dir, f"map_T_lnP_{name}_{time_str}")
    save_figure(fig_SlogP, out_dir, f"map_S_lnP_{species}_{name}_{time_str}")
    save_figure(fig_SZ, out_dir, f"map_S-Z{species}_{name}_{time_str}")
    save_figure(fig_TZ, out_dir, f"map_T-Z{species}_{name}_{time_str}")
    








if __name__ == "__main__":
    main()


# %%
