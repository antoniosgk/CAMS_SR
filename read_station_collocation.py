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

# species and dataset naming convention
product = "inst3d"     # filename prefix,depends on the name of the file saved
species = "O3"          # e.g. "O3", "CO2", ...
date = "20050524"       # YYYYMMDD
time = "0200"           # HHMM

# Path construction for species file 
species_file = pathlib.Path(f"{base_path}/{species}/{product}_{date}_{time}.nc4")

# Path construction for Pressure Level file 
pl_file = pathlib.Path(f"{base_path}/PL/{product}_{date}_{time}.nc4")

# Path construction for Temperature file
T_file = pathlib.Path(f"{base_path}/T/{product}_{date}_{time}.nc4")

# Station selection: set one or leave None to be prompted
selected_index = 1409 # range from 0 to 2027
selected_name = None  #See from the list

# How many horizontal cells to include around central cell for slicing/plotting
cell_nums = 8

# If True: perform linear interpolation vertically instead of selecting nearest level
DO_VERTICAL_INTERPOLATION = False #For identifying the vertical level of the station 
#%%
def load_stations(path):
    """Load station table. Expects tab-separated with Station_Name, Latitude, Longitude, Altitude.
       Renames index to 'idx'."""
    df = pd.read_csv(path, sep="\t")
    df = df.reset_index().rename(columns={"index": "idx"})
    # Attempt normalization of column names: Cases if the names are not those expected,not needed as
    #we know the names of the columns a priori
    for col in df.columns:
        if col.lower().startswith("station"):
            df = df.rename(columns={col: "Station_Name"})
        if col.lower().startswith("lat"):
            df = df.rename(columns={col: "Latitude"})
        if col.lower().startswith("lon"):
            df = df.rename(columns={col: "Longitude"})
        if col.lower().startswith("alt"):
            df = df.rename(columns={col: "Altitude"})
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


def find_coord_names(ds):
    """Return coordinate names for lat, lon, time, lev if present.
    Also this function may be unneeded as we already know the names of the coords of the ds"""
    lat_names = ["lat", "latitude", "nav_lat"]
    lon_names = ["lon", "longitude", "nav_lon"]
    lev_names = ["lev", "level", "lev_m", "model_level", "levb"]  # try common
    time_names = ["time"]

    def pick(names):
        for n in names:
            if n in ds.coords or n in ds:
                return n
        for n in names:
            if n in ds.variables:
                return n
        return None

    return {
        "lat": pick(lat_names),
        "lon": pick(lon_names),
        "lev": pick(lev_names),
        "time": pick(time_names)
    }


def nearest_grid_index(st_lat, st_lon, lats, lons):
    """
    Return nearest grid indices (i,j).
    Works for 1D or 2D lats/lons arrays:
    - If lats, lons are 1D: find argmin per axis.
    - If 2D: compute haversine-ish distance (approx) and find argmin of flattened index.
    In our case lats/lons are 1D so the 2D case scenario may be removed
    """
    # Make arrays numpy
    lats = np.array(lats)
    lons = np.array(lons)

    # 1D case
    if lats.ndim == 1 and lons.ndim == 1:
        i = np.abs(lats - st_lat).argmin()
        j = np.abs(lons - st_lon).argmin()
        return int(i), int(j)

    # If they are 2D and same shape
    if lats.ndim == 2 and lons.ndim == 2 and lats.shape == lons.shape:
        # compute squared distance on lat/lon (approx, OK for nearest)
        # convert degrees to radians for small-angle approx, but simple squared differences suffice too
        dlat = (lats - st_lat)
        dlon = (lons - st_lon)
        dist2 = dlat**2 + dlon**2
        idx_flat = np.argmin(dist2)
        i, j = np.unravel_index(idx_flat, lats.shape)
        return int(i), int(j)

    raise ValueError("Unsupported lat/lon array shapes for nearest-grid lookup.")


def find_variable_name(ds, candidates):
    """Return the first variable name from candidates that exists in ds."""
    for c in candidates:
        if c in ds.variables:
            return c
    return None


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

Rd = 287.05
g  = 9.80665

def pressure_to_height(p_hPa, T_K):
    """
    Compute geometric height using the hypsometric equation.

    Parameters
    ----------
    p_hPa : 1D array
        Pressure profile in hPa.
    T_K : 1D array
        Temperature profile in K.

    Returns
    -------
    z : 1D array
        Heights in meters of each model level.
    """
    p_hPa = np.asarray(p_hPa)
    T_K   = np.asarray(T_K)

    # Avoid divide-by-zero / negative pressures
    p_hPa = np.clip(p_hPa, 1e-6, None)

    # Hypsometric equation (relative to 1000 hPa)
    # z = (Rd * T / g) * ln(p0 / p)
    return (Rd * T_K / g) * np.log(1000.0 / p_hPa)


def geos_find_level_index(p_prof_Pa, T_prof_K, station_alt):
    """
    Find the model level closest to station altitude using PL and T profiles.

    Parameters
    ----------
    p_prof_Pa : 1D array
        Pressure profile in Pa at model level midpoints.
    T_prof_K : 1D array
        Temperature profile in K at the same levels.
    station_alt : float
        Station altitude in meters.

    Returns
    -------
    idx : int
        Index of the closest model level.
    p_level_hPa : float
        Pressure of that level (hPa).
    z_level_m : float
        Height of that level (m).
    """
    p_prof_Pa = np.asarray(p_prof_Pa).squeeze()
    T_prof_K  = np.asarray(T_prof_K).squeeze()

    if p_prof_Pa.ndim != 1 or T_prof_K.ndim != 1:
        raise ValueError(f"Expected 1D profiles, got shapes {p_prof_Pa.shape} and {T_prof_K.shape}")

    # Pa -> hPa
    p_prof_hPa = p_prof_Pa / 100.0

    # Height of each model level from hypsometric equation
    z_prof = pressure_to_height(p_prof_hPa, T_prof_K)

    # Level closest in height to station altitude
    idx = int(np.argmin(np.abs(z_prof - station_alt)))

    return idx, float(p_prof_hPa[idx]), float(z_prof[idx])
#%%
 #----------- MAIN SCRIPT --------------
if __name__ == "__main__":
    print("Loading station list...")
    df_st = load_stations(stations_file)
    print(df_st.head())
    df_sorted = df_st.sort_values(by="Altitude", ascending=False)

# --- Select the top 10 ---
    #top10 = df_sorted.head(10)

# --- Print results ---
    #print("\nTop 10 highest-altitude stations:")
    #print(top10[["idx", "Station_Name", "Altitude"]].to_string(index=False))
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

    #print(f"\nOpening species file: {species_file}")
    ds_species = xr.open_dataset(species_file)
    #print(ds_species)
#%%
    print(f"Opening PL file: {pl_file}")
    ds_pl = xr.open_dataset(pl_file)
    print(ds_pl)

#%%
    print(f"Opening T file: {T_file}")
    ds_T = xr.open_dataset(T_file)
    print(ds_T)

#%%
    # Determine coordinate name mapping (try to prefer the species DS coords)
    coords = find_coord_names(ds_species)
    if coords["lat"] is None or coords["lon"] is None:
        # fallback to PL dataset
        coords = find_coord_names(ds_pl)
    lat_name = coords["lat"]
    lon_name = coords["lon"]
    lev_name = coords["lev"]
    time_name = coords["time"] or "time"

    if lat_name is None or lon_name is None:
        raise ValueError("Could not find lat/lon coordinate names in datasets.")

    # Extract lat/lon arrays (convert to 1D if needed)
    lats = ds_species[lat_name].values
    lons = ds_species[lon_name].values

    # Find nearest indices on the full grid
    i, j = nearest_grid_index(lat_s, lon_s, lats, lons)
    print(f"\nStation '{name}' assigned to model grid cell (i={i}, j={j})")
    # print model coords if 1D
    if np.ndim(lats) == 1 and np.ndim(lons) == 1:
        print(f"Model lat={lats[i]:.6f}, lon={lons[j]:.6f}")
    else:
        print(f"Model lat={lats[i,j]:.6f}, lon={lons[i,j]:.6f}")
    print(f'\nStation {name} has lat={lat_s} and lon={lon_s}')
    # --- compute slicing window ---
    # If lats/lons were 2D, we need grid dimensions — assume first two dims are (y,x)
    if np.ndim(lats) == 1:
        Ny = lats.shape[0]
        Nx = lons.shape[0]
    else:
        Ny, Nx = lats.shape

    i1, i2 = max(0, i-cell_nums), min(Ny-1, i+cell_nums)
    j1, j2 = max(0, j-cell_nums), min(Nx-1, j+cell_nums)

    print(f"\nLoading domain subset: i={i1}:{i2}, j={j1}:{j2}")
   
#%%
    # Read PL and T profiles at the nearest grid point
# ---------------------------
# Find PL variable name in pl dataset
    pl_var_name = find_variable_name(ds_pl, ["PL", "pl", "pressure", "P", "plev"])
    if pl_var_name is None:
       raise ValueError("Could not find variable 'PL' (or alternatives) in PL file.")

    print(f"Using PL variable: {pl_var_name}")

    pl = ds_pl[pl_var_name]

# Build indexer dict for PL / T: time + horizontal point
    indexer = {}
# time (assume first time step)
    if time_name in pl.dims:
      indexer[time_name] = 0

    pl_dims = list(pl.dims)

# try to map lat/lon dims
    lat_dim = lat_name if lat_name in pl_dims else None
    lon_dim = lon_name if lon_name in pl_dims else None

# If their PL lat/lon dims are named differently (e.g., 'y','x'), try detect by size
    if lat_dim is None or lon_dim is None:
      for d in pl_dims:
          if lat_dim is None and pl.sizes[d] == Ny:
             lat_dim = d
          elif lon_dim is None and pl.sizes[d] == Nx:
            lon_dim = d

# Fallback: assume last two dims are lat, lon
    if lat_dim is None or lon_dim is None:
       if len(pl_dims) >= 2:
         lat_dim = pl_dims[-2]
         lon_dim = pl_dims[-1]

# Now build indexer with horizontal indices
    if lat_dim is not None and lon_dim is not None:
     indexer[lat_dim] = i
     indexer[lon_dim] = j

# ---- Extract PL profile (Pa) ----
    try:
      pl_profile_xr = pl.isel(indexer)
    except Exception as e:
    # fallback: select via nearest lat/lon coordinates
        try:
         sel_kwargs = {}
         if lat_name in ds_pl.coords:
            sel_kwargs[lat_name] = ds_pl[lat_name].sel({lat_name: lat_s}, method="nearest")
         if lon_name in ds_pl.coords:
            sel_kwargs[lon_name] = ds_pl[lon_name].sel({lon_name: lon_s}, method="nearest")
         pl_profile_xr = pl.sel(sel_kwargs, method="nearest")
        except Exception as e2:
         raise RuntimeError("Failed to index PL profile at station grid point") from e

    pl_profile = np.asarray(pl_profile_xr).squeeze()
    if pl_profile.ndim != 1:
     raise ValueError(f"PL profile after selection is not 1D: shape {pl_profile.shape}")

# Ensure PL units are Pa; if values look like hPa (<2000), convert to Pa
    median_pl = np.nanmedian(pl_profile)
    if median_pl < 2000:  # likely in hPa
     print("Detected PL likely in hPa; converting to Pa.")
     pl_profile = pl_profile * 100.0

# ---- Extract T profile (K) at the same point ----
    t_var_name = find_variable_name(ds_T, ["T", "t", "TMPU", "temperature"])
    if t_var_name is None:
     raise ValueError("Could not find temperature variable (T) in PL file.")

    T = ds_T[t_var_name]

    try:
        T_profile_xr = T.isel(indexer)
    except Exception as e:
        try:
          sel_kwargs = {}
          if lat_name in ds_T.coords:
            sel_kwargs[lat_name] = ds_pl[lat_name].sel({lat_name: lat_s}, method="nearest")
          if lon_name in ds_T.coords:
            sel_kwargs[lon_name] = ds_pl[lon_name].sel({lon_name: lon_s}, method="nearest")
          T_profile_xr = T.sel(sel_kwargs, method="nearest")
        except Exception as e2:
            raise RuntimeError("Failed to index T profile at station grid point") from e

    T_profile = np.asarray(T_profile_xr).squeeze()
    if T_profile.ndim != 1:
        raise ValueError(f"T profile after selection is not 1D: shape {T_profile.shape}")

    if T_profile.shape != pl_profile.shape:
        raise ValueError(f"PL and T profiles have different shapes: {pl_profile.shape} vs {T_profile.shape}")

# ---- Use hypsometric equation to find closest level in height ----
    print(f"\nFinding closest model level to station altitude {alt_s} m...")
    model_level_index, p_level_hPa, z_level_m = geos_find_level_index(pl_profile, T_profile, alt_s)

    print(f"Nearest model level index (0-based) = {model_level_index}")
    print(f"  Pressure at level = {p_level_hPa:.1f} hPa")
    print(f"  Height at level   = {z_level_m:.1f} m")

# Optional: show a few nearby levels
    low = max(0, model_level_index - 2)
    high = min(len(pl_profile) - 1, model_level_index + 2)
    print("Nearby levels:")
    for lev_idx in range(low, high + 1):
    # convert to hPa for printing
      p_hPa = pl_profile[lev_idx] / 100.0
    # recompute z for the printout
      z_lev = pressure_to_height(p_hPa, T_profile[lev_idx])
      print(f"  lev {lev_idx:2d}: p={p_hPa:7.1f} hPa, z≈{z_lev:8.1f} m")

    

# %%
ds_big = ds_species

# Coordinates
lats_big = ds_big[lat_name].values
lons_big = ds_big[lon_name].values

# --- CASE 1: 1D lat/lon ---
if lats_big.ndim == 1 and lons_big.ndim == 1:

    ds_small = ds_big.isel({lat_name: slice(i1, i2+1),
                            lon_name: slice(j1, j2+1)})

    lats_small = lats_big[i1:i2+1]
    lons_small = lons_big[j1:j2+1]

# --- CASE 2: 2D lat/lon ---
else:
    ds_small = ds_big.isel({lat_name: slice(i1, i2+1),
                            lon_name: slice(j1, j2+1)})

    lats_small = lats_big[i1:i2+1, j1:j2+1]
    lons_small = lons_big[i1:i2+1, j1:j2+1]

# variable extraction
var_name = species
data_arr = ds_small[var_name].isel({time_name: 0,
                                    lev_name: model_level_index}).values

# free memory
ds_big.close()
del ds_big
#%%
# local (station) indices in the small box
ii = i - i1
jj = j - j1

# %%
Ny, Nx = data_arr.shape

def safe_slice(low, high, maxN):
    return slice(max(low, 0), min(high, maxN))

# Sector 1
S1 = np.zeros((Ny, Nx), dtype=bool)
S1[safe_slice(ii-1, ii+2, Ny), safe_slice(jj-1, jj+2, Nx)] = True
S1[ii, jj] = True

# Sector 2
S2 = np.zeros((Ny, Nx), dtype=bool)
S2[safe_slice(ii-2, ii+3, Ny), safe_slice(jj-2, jj+3, Nx)] = True
S2[S1] = False

# Sector 3
S3 = np.zeros((Ny, Nx), dtype=bool)
S3[safe_slice(ii-3, ii+4, Ny), safe_slice(jj-3, jj+4, Nx)] = True
S3[S1] = False
S3[S2] = False
# %%
# Extract the part of the masks corresponding to the PLOTTED domain
# (which is exactly ds_small)

S1_small = S1
S2_small = S2
S3_small = S3
# (above is enough because S1,S2,S3 already match ds_small dimensions)

# -------------------------------------------------------------------
#                        TABLES FOR EACH SECTOR
# -------------------------------------------------------------------
def sector_table(mask):
    iy, ix = np.where(mask)
    return pd.DataFrame({
        "lat_idx": iy,
        "lon_idx": ix,
        "lat": lats_small[iy],
        "lon": lons_small[ix],
        species: data_arr[iy, ix]
    })

df_S1 = sector_table(S1)
df_S2 = sector_table(S2)
df_S3 = sector_table(S3)

print(df_S1)
print(df_S2)
print(df_S3)
# %%
# %%
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
fig = plt.figure(figsize=(8, 8))
proj = ccrs.PlateCarree()
ax = plt.axes(projection=proj)

data_var = ds_small[species]
units = data_var.attrs.get("units", species)

d = 0.8
lon_min, lon_max = lon_s - d, lon_s + d
lat_min, lat_max = lat_s - d, lat_s + d
ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

# ticks
xticks = np.arange(lon_min, lon_max + 0.1, 0.5)
yticks = np.arange(lat_min, lat_max + 0.1, 0.5)
ax.set_xticks(xticks, crs=proj)
ax.set_yticks(yticks, crs=proj)
ax.set_xticklabels(xticks, rotation=45, ha='right')
ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:.3f}'))

# --- 2D lon/lat grids ---
LON2D, LAT2D = np.meshgrid(lons_small, lats_small)

# --- color limits ---
vmin = np.nanmin(data_arr)
vmax = np.nanmax(data_arr)
norm = Normalize(vmin=vmin, vmax=vmax)

# --- main pcolormesh plot ---
im = ax.pcolormesh(LON2D, LAT2D, data_arr,
                   cmap="turbo", shading="auto", norm=norm)

# station marker
ax.plot(lon_s, lat_s, 'kx', markersize=12)

ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=0)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)
#                RECTANGLES FOR SECTORS S1, S2, S3
# ------------------------------------------------------------
# dx, dy in DEGREES (size of one grid cell)
dlon = np.abs(lons_small[1] - lons_small[0])
dlat = np.abs(lats_small[1] - lats_small[0])

# Sector sizes (in grid cells)
sizes = {
    "S1": 1,   # 3x3  → radius = 1 cell
    "S2": 2,   # 5x5  → radius = 2 cells
    "S3": 3,   # 7x7  → radius = 3 cells
}

# Colors and linewidths
rect_styles = {
    "S1": {"edgecolor": "black", "linewidth": 2},
    "S2": {"edgecolor": "red",   "linewidth": 2},
    "S3": {"edgecolor": "blue",  "linewidth": 2},
}

# Center of rectangles = central grid cell center position
cx = lons_small[jj]
cy = lats_small[ii]

for name, r in sizes.items():
    width = (2*r + 1) * dlon
    height = (2*r + 1) * dlat
    left = cx - width / 2
    bottom = cy - height / 2

    rect = Rectangle(
        (left, bottom),
        width,
        height,
        facecolor="none",
        transform=ccrs.PlateCarree(),
        **rect_styles[name]
    )
    ax.add_patch(rect)

plt.colorbar(im, ax=ax, pad=0.02, label=units)
plt.title(f"{species} around station")
plt.tight_layout()
plt.show()

# %%
