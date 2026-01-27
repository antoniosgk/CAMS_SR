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
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# main.py
#%%
from vertical_indexing import metpy_find_level_index,metpy_compute_heights
from stations_utils import load_stations, select_station, all_stations, map_stations_to_model_levels
from horizontal_indexing import nearest_grid_index
from file_utils import stations_path, species_file, T_file, pl_file,species,orog_file,RH_file
from calculation import (sector_table,compute_sector_tables_generic,
                        sector_stats,compute_ring_sector_masks,compute_cumulative_sector_tables,
                        weighted_quantile,sector_stats_unweighted,sector_stats_weighted
                        )
from plots import (plot_variable_on_map,plot_rectangles,plot_sector_boxplots,
    plot_profile_P_T,
    plot_profile_T_Z,
    plot_profile_T_logP,
    plot_profile_species_logP,
    plot_profile_species_Z,save_figure, plot_sector_boxplots,box_stats_from_df,plot_sector_boxplots_weighted)
"""
/
This comment section includes all the variables,the functions,their names
and their physical interpretation
idx:index of the station of the stations file
name:name of the station of the stations file
cell_nums:number of cells around the central grid cell(the one the station falls into)
that will be plotted lll
lat_s
"""
#%%
def main():
    idx=23 #index of station of the stations_file
    name=None #name of the station
    cell_nums = 50 #numb of cells that will get plotted n**2
    d_zoom_species=5.0 #zoom of plots
    d_zoom_topo=5.0  #zoom of topo in fig3
    zoom_map= 45.0   #extent of map in fig4
    radii = list(range(1, 4)) #(range(1,cell_nums+1))
    out_dir="/home/agkiokas/CAMS/plots/" #where the plots are saved
    fig4_with_topo = False
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

    #i1, i2 = max(0, i-cell_nums), min(Ny-1, i+cell_nums) #subsets of lats (for plotting)
    #j1, j2 = max(0, j-cell_nums), min(Nx-1, j+cell_nums) #subset of lons (for plotting)

    #print(f"\nLoading domain subset: i={i1}:{i2}, j={j1}:{j2} for plotting")
    # --- SMALL box (species) ---
    i1_s, i2_s = max(0, i - cell_nums), min(Ny - 1, i + cell_nums)
    j1_s, j2_s = max(0, j - cell_nums), min(Nx - 1, j + cell_nums)

    print(f"\nLoading domain subset: i={i1_s}:{i2_s}, j={j1_s}:{j2_s} for plotting")
    ds_small = ds_species.isel(lat=slice(i1_s, i2_s + 1), lon=slice(j1_s, j2_s + 1))
    lats_small = lats[i1_s:i2_s + 1]
    lons_small = lons[j1_s:j2_s + 1]

    ii = i - i1_s
    jj = j - j1_s

    ds_big = ds_species #a copy of ds_species,maybe not needed,to reevalutate

    # Coordinates
    lats_big = ds_big['lat'].values
    lons_big = ds_big['lon'].values



# variable extraction
    var_name = species
    

 # free memory
    #ds_big.close()
    #del ds_big
# local (station) indices in the small box
    ii = i - i1_s
    jj = j - j1_s

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
    # PHIS_small aligned to ds_small domain
    # --- LARGE box (terrain background) ---
    dlat = float(np.abs(lats[1] - lats[0]))
    dlon = float(np.abs(lons[1] - lons[0]))

    cell_nums_lat = int(np.ceil(d_zoom_topo / dlat))
    cell_nums_lon = int(np.ceil(d_zoom_topo / dlon))
    cell_nums_bg = max(cell_nums_lat, cell_nums_lon)

    i1_bg, i2_bg = max(0, i - cell_nums_bg), min(Ny - 1, i + cell_nums_bg)
    j1_bg, j2_bg = max(0, j - cell_nums_bg), min(Nx - 1, j + cell_nums_bg)

    PHIS_bg = ds_orog["PHIS"].isel(time=0, lat=slice(i1_bg, i2_bg + 1), lon=slice(j1_bg, j2_bg + 1)).values
    z_orog_bg = PHIS_bg / 9.80665

    lats_bg = lats[i1_bg:i2_bg + 1]
    lons_bg = lons[j1_bg:j2_bg + 1]


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
    lats_small, lons_small, data_arr_ppb,
    lon_s, lat_s,
    units=units_ppb,
    species_name=species,
    d=d_zoom_species,
    time_str=time_str,
    meta=meta,
    z_orog_m=z_orog_bg,          
    lats_terrain=lats_bg,
    lons_terrain=lons_bg,
    add_orog_contours=True,
    plot_orography=False
)

    

    fig2, ax2, im2 = plot_variable_on_map(
    lats_small, lons_small, data_arr_ppb,
    lon_s, lat_s,
    units=units_ppb,
    species_name=species,
    d=d_zoom_species,
    time_str=time_str,
    meta=meta,
    z_orog_m=z_orog_bg,
    lats_terrain=lats_bg,
    lons_terrain=lons_bg,plot_orography=False
)
    plot_rectangles(ax2, lats_small, lons_small, ii, jj, im2, meta=meta,radii=radii)
    plt.show()

    fig3, ax3, _ = plot_variable_on_map(
    lats_small,
    lons_small,
    data_arr=None,          # ignored
    lon_s=lon_s,
    lat_s=lat_s,
    units="",
    species_name="",
    d=d_zoom_topo,
    meta=meta,
    lats_terrain=lats_bg,
    lons_terrain=lons_bg,
    plot_orography=True,
    z_orog_m=z_orog_bg,
    terrain_alpha=0.6,
    add_orog_contours=True,
    plot_species=False,
)

    # --- FIG4: stations context map (optionally with topography) ---
    if fig4_with_topo:
    # background terrain + selected station cross (from your topo function)
        fig4, ax4, _ = plot_variable_on_map(
        lats_small,
        lons_small,
        data_arr=None,              # ignored
        lon_s=lon_s,
        lat_s=lat_s,
        d=d_zoom_topo,
        meta=meta,
        lats_terrain=lats_bg,
        lons_terrain=lons_bg,
        plot_orography=True,
        z_orog_m=z_orog_bg,
        add_orog_contours=True,
        plot_species=False,         # topo-only base
    )
    else:
    # stations-only map (no terrain, no terrain colorbar)
        proj = ccrs.PlateCarree()
        fig4, ax4 = plt.subplots(subplot_kw={"projection": proj})

        lon_min, lon_max = lon_s - zoom_map, lon_s + zoom_map
        lat_min, lat_max = lat_s - zoom_map, lat_s + zoom_map
        ax4.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

        ax4.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        ax4.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=0)
        ax4.coastlines(resolution="10m", linewidth=0.8)
        #ax4.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
        ax4.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=1)
        ax4.gridlines(crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.6,
        alpha=0.95,
        linestyle="--",
        zorder=1)
# --- overlay stations (black) + selected station (red) ---
    st = stations.copy()
    st["Latitude"] = pd.to_numeric(st["Latitude"], errors="coerce")
    st["Longitude"] = pd.to_numeric(st["Longitude"], errors="coerce")
    st = st.dropna(subset=["Latitude", "Longitude"])
    # plot only stations within the fig4 extent (cleaner + faster)
    lon_min, lon_max = lon_s - zoom_map, lon_s + zoom_map
    lat_min, lat_max = lat_s - zoom_map, lat_s + zoom_map
    st = st[st["Longitude"].between(lon_min, lon_max) & st["Latitude"].between(lat_min, lat_max)]

    ax4.scatter(
    st["Longitude"].values,
    st["Latitude"].values,
    s=6,
    c="k",
    alpha=0.7,
    transform=ccrs.PlateCarree(),
    zorder=6,
    label="Stations",
)

    ax4.scatter(
    [lon_s],
    [lat_s],
    s=45,
    c="red",
    edgecolors="k",
    linewidths=0.6,
    transform=ccrs.PlateCarree(),
    zorder=7,
    label=f"Selected: {name}",
)

    ax4.legend(loc="upper right")
    ax4.set_title(f"Stations in China", pad=18)


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
     species_units=units_ppb,
     time_str=time_str,
     meta=meta
    )
    

# species–Z
    fig_SZ, ax_SZ = plot_profile_species_Z(
    z_prof,
    species_prof_ppb,
    idx_level,
    species_name=species,
    species_units=units_ppb,
    time_str=time_str,
    z_units="km",
    meta=meta
)

    plt.show()
    

    sector_dfs, sector_masks = compute_sector_tables_generic(
    ii, jj, lats_small, lons_small, data_arr_ppb, species, radii=radii)

    sector_names = [f"S{k+1}" for k in range(len(sector_dfs))]
    
    print("\nSector statistics:")
    for nm, df in zip(sector_names, sector_dfs):
     st = sector_stats(df, species)
     print(
        f"{nm}: n={st['n']}, mean={st['mean']:.3e}, CV={st['cv']:.4f}, "
        f"median={st['median']:.3e}, IQR={st['iqr']:.3e}"
    )

    cum_dfs, cum_masks = compute_cumulative_sector_tables(
    sector_masks,
    lats_small,
    lons_small,
    data_arr_ppb,
    species
)
    def add_area_weights(df, lat_col="lat", w_col="w_area"):
     lat_rad = np.deg2rad(pd.to_numeric(df[lat_col], errors="coerce").to_numpy(dtype=float))
     w = np.cos(lat_rad)
     w = np.clip(w, 0.0, None)
     df[w_col] = w
     return df

    sector_dfs = [add_area_weights(df.copy()) for df in sector_dfs]
    cum_dfs    = [add_area_weights(df.copy()) for df in cum_dfs]

    print("\nRING SECTORS (UNWEIGHTED vs AREA-WEIGHTED)")
    for k, df in enumerate(sector_dfs, start=1):
     su = sector_stats_unweighted(df, species)
     sw = sector_stats_weighted(df, species, w_col="w_area")
     print(
        f"S{k}: mean={su['mean']:.3f}, std={su['std']:.3f}, cv={su['cv']:.3f}, "
        f"med={su['median']:.3f}, IQR={su['iqr']:.3f}  ||  "
        f"mean_w={sw['mean_w']:.3f}, std_w={sw['std_w']:.3f}, cv_w={sw['cv_w']:.3f}, "
        f"med_w={sw['median_w']:.3f}, IQR_w={sw['iqr_w']:.3f}"
    )
    print("\nCUMULATIVE SECTORS (UNWEIGHTED vs AREA-WEIGHTED)")

    for k, df in enumerate(cum_dfs, start=1):
     su = sector_stats_unweighted(df, species)
     sw = sector_stats_weighted(df, species, w_col="w_area")
     print(
        f"C{k}: mean={su['mean']:.3f}, std={su['std']:.3f}, cv={su['cv']:.3f}, "
        f"med={su['median']:.3f}, IQR={su['iqr']:.3f}  ||  "
        f"mean_w={sw['mean_w']:.3f}, std_w={sw['std_w']:.3f}, cv_w={sw['cv_w']:.3f}, "
        f"med_w={sw['median_w']:.3f}, IQR_w={sw['iqr_w']:.3f}"
    )



    # Unweighted (your existing)
    fig_u, ax_u = plot_sector_boxplots(sector_dfs, species,
                                   sector_names=[f"S{k}" for k in range(1, len(sector_dfs)+1)],
                                   title=f"{species} ring sectors (UNWEIGHTED)")

# Weighted
    fig_w, ax_w = plot_sector_boxplots_weighted(sector_dfs, species, w_col="w_area",
                                            sector_names=[f"S{k}" for k in range(1, len(sector_dfs)+1)],
                                            title=f"{species} ring sectors (AREA-WEIGHTED)")
    # Unweighted (your existing)
    fig_u, ax_u = plot_sector_boxplots(cum_dfs, species,
                                   sector_names=[f"S{k}" for k in range(1, len(sector_dfs)+1)],
                                   title=f"{species} cumulative sectors (UNWEIGHTED)")

# Weighted
    fig_w, ax_w = plot_sector_boxplots_weighted(cum_dfs, species, w_col="w_area",
                                            sector_names=[f"S{k}" for k in range(1, len(sector_dfs)+1)],
                                            title=f"{species} cumulative sectors (AREA-WEIGHTED)")
    

    """
    save_figure(fig1, out_dir, f"map_{species}_{name}_{time_str}")
    save_figure(fig2, out_dir, f"map_with sectors_{species}_{name}_{time_str}")
    save_figure(fig3,out_dir, f"topo_map_{name}")
    save_figure(fig_PT, out_dir, f"map_P_T_{time_str}")
    save_figure(fig_TlogP, out_dir, f"map_T_lnP_{name}_{time_str}")
    save_figure(fig_SlogP, out_dir, f"map_S_lnP_{species}_{name}_{time_str}")
    save_figure(fig_SZ, out_dir, f"map_S-Z{species}_{name}_{time_str}")
    save_figure(fig_TZ, out_dir, f"map_T-Z{species}_{name}_{time_str}")
    
    
    

    df_levels, n_not_surface, df_non_surface = map_stations_to_model_levels(
    stations,
    ds_T,
    ds_PL,
    ds_orog,
    lats,
    lons,
    RH_ds=ds_RH,
    )

    #print(df_levels.head())
    print(f"Stations NOT mapped to surface level: {n_not_surface} / {len(df_levels)}")
    df_levels.to_csv("station_model_level_mapping.csv", index=False)
    print(df_non_surface.head(26))
    
    """




if __name__ == "__main__":
    main()


# %%
