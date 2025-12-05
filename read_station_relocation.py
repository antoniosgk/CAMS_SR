#%%
#IMPORT LIBRARIES
import os
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
#%%
# USER INPUT (edit these)
# --------------------------
stations_file = "/home/agkiokas/CAMS/CHINESE_STATIONS_INFO_2015_2023.txt"  # your stations table
base_path = "/mnt/store01/agkiokas/CAMS/inst"

# species and dataset naming convention used previously
product = "inst3d"     # filename prefix,depends on the name of the file saved
species = "O3"          # e.g. "O3", "CO2", ...
date = "20050524"       # YYYYMMDD
time = "0200"           # HHMM

# Path construction for species file (same pattern as your original code)
species_file = pathlib.Path(f"{base_path}/{species}/{product}_{date}_{time}.nc4")

# Path construction for PL file (often in same product1 but under variable 'PL')
# Adjust pl_file if your repository stores PL elsewhere
pl_file = pathlib.Path(f"{base_path}/PL/{product}_{date}_{time}.nc4")

# Station selection: Select by index OR by name of the station
selected_index = 678
selected_name = None

# How many horizontal cells to include around central cell
cell_nums = 4

# If True: perform linear interpolation vertically instead of selecting nearest level
DO_VERTICAL_INTERPOLATION = False #For identifying the vertical level of the station 
#%%
#F U N C T I O N S
def load_stations(path):
    """Load station table. Expects tab-separated with Station_Name, Latitude, Longitude, Altitude.
       Renames index to 'idx'."""
    df = pd.read_csv(path, sep="\t")
    df = df.reset_index().rename(columns={"index": "idx"})
    expected = ["idx", "Station_Name", "Latitude", "Longitude", "Altitude"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Stations file missing expected columns: {missing}") #raises error if the col names are not those expected
    return df[expected]

def select_station(df, idx=None, name=None):
    """Return station row based on index or name 
     This function can be removed as the user puts the name or the index of the station he wants"""
    if idx is None and name is None:
        print(df[["idx", "Station_Name"]])
        s = input("Select station by index or name: ").strip()
    else:
        s = str(idx if idx is not None else name)

    if s.isdigit():
        row = df[df["idx"] == int(s)]
        if not row.empty:
            return row.iloc[0]

    row = df[df["Station_Name"] == s]
    if not row.empty:
        return row.iloc[0]

    raise ValueError(f"Station '{s}' not found.")

def nearest_grid_index(st_lat, st_lon, lats, lons):
    """
    Return nearest grid indices (i,j).
    Works for 1D lats/lons arrays:
    - If lats, lons are 1D: find argmin per axis.
    """
    # Make arrays numpy
    lats = np.array(lats)
    lons = np.array(lons)

    # 1D case
    if lats.ndim == 1 and lons.ndim == 1:
        i = np.abs(lats - st_lat).argmin()
        j = np.abs(lons - st_lon).argmin()
        return int(i), int(j)

def altitude_to_pressure(z_m):
    """Convert altitude (m) to pressure (Pa) using standard barometric formula (ISA troposphere).
       Good approximation for typical surface altitudes."""
    # Constants
    p0 = 101325.0       # Pa
    T0 = 288.15         # K
    g = 9.80665         # m/s2
    L = 0.0065          # K/m
    R = 287.05          # J/(kg K)
    # Avoid negative base when very high z: clip to realistic domain
    term = 1.0 - L * z_m / T0
    if term <= 0:
        return 0.0
    exponent = g / (R * L)
    return p0 * (term ** exponent)

 #%%
 #----------- MAIN SCRIPT --------------
print("Loading station list...")
df_st = load_stations(stations_file)
print(df_st.head())

station = select_station(df_st, selected_index, selected_name)
name = station["Station_Name"]
lat_s = float(station["Latitude"])
lon_s = float(station["Longitude"])
alt_s = float(station["Altitude"])
print(f"\nSelected station: {name} (lat={lat_s}, lon={lon_s}, alt={alt_s} m)")

    # Check files exist
if not species_file.exists():
        raise FileNotFoundError(f"Species file not found: {species_file}")
if not pl_file.exists():
        raise FileNotFoundError(f"PL (pressure) file not found: {pl_file}")

print(f"\nOpening species file: {species_file}")
ds_species = xr.open_dataset(species_file)
print(ds_species)
#%%
print(f"Opening PL file: {pl_file}")
ds_pl = xr.open_dataset(pl_file)
print(ds_pl) 

# %%
lats = ds_species['lat'].values
lons = ds_species['lon'].values

# %%
 # Find nearest indices on the full grid
i, j = nearest_grid_index(lat_s, lon_s, lats, lons)
print(f"\nStation '{name}' assigned to model grid cell (i={i}, j={j})")
print(f"Model lat={lats[i]:.6f}, lon={lons[j]:.6f}")
print(f"Station coordinates Latitude: {lat_s},Longitude: {lon_s}")
# %%
if np.ndim(lats) == 1:
        Ny = lats.shape[0]
        Nx = lons.shape[0]

i1, i2 = max(0, i-cell_nums), min(Ny-1, i+cell_nums)
j1, j2 = max(0, j-cell_nums), min(Nx-1, j+cell_nums)

print(f"\nLoading domain subset: i={i1}:{i2}, j={j1}:{j2}")


# %%

# %%
