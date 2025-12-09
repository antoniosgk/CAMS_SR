#%%
import pandas as pd
from file_utils import base_path,stations_path
#%%
def load_stations(stations_path):
    """Load station table. Expects tab-separated with Station_Name, Latitude, Longitude, Altitude.
       Renames index to 'idx'."""
    df = pd.read_csv(stations_path, sep="\t")
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

def all_stations():
    """
    For all stations in stations_path:
      - find nearest model grid point
      - compute nearest model level to station altitude
      - print: Station_Name, Altitude, Nearest_Level (row by row)

    At the beginning, the user is asked how many stations they want printed.
    In the end, it prints how many stations (out of the total) do NOT have
    level 71 as the closest level.
    """
    import xarray as xr
    import numpy as np
    from metpy.units import units
    from metpy.constants import g
    from vertical_indexing import metpy_find_level_index
    from horizontal_indexing import nearest_grid_index
    from file_utils import (
        stations_path,
        species_file,
        T_file,
        pl_file,
        orog_file,
    )

    # Load all stations
    stations = load_stations(stations_path)
    total_stations = len(stations)

    # Ask user how many stations to print

    t = 10  # or set this from outside / function argument

# t is an integer here; just clamp it to [1, total_stations]
    n_to_print = max(1, min(int(t), total_stations))
    # Open datasets once
    ds_species = xr.open_dataset(species_file)
    ds_T = xr.open_dataset(T_file)
    ds_PL = xr.open_dataset(pl_file)
    ds_orog = xr.open_dataset(orog_file)

    # Horizontal grid
    lats = ds_species["lat"].values
    lons = ds_species["lon"].values

    # Orography field
    PHIS_field = ds_orog["PHIS"]

    printed = 0
    non71_count = 0
    for _, station in stations.iterrows():
        lat_raw = station["Latitude"]
        lon_raw = station["Longitude"]
        alt_raw = station["Altitude"]

    # Skip stations with missing / placeholder values
        if lat_raw in ("-", "", None) or lon_raw in ("-", "", None) or alt_raw in ("-", "", None):
        # optional: print a message once or count skipped stations
           continue

        try:
          lat_s = float(lat_raw)
          lon_s = float(lon_raw)
          alt_s = float(alt_raw)
        except ValueError:
        # Any other non-numeric junk → skip
          continue

        name = station["Station_Name"]

    # Nearest model grid cell
        i, j = nearest_grid_index(lat_s, lon_s, lats, lons)

    # PHIS at that grid point (assume time dimension exists → time=0)
        PHIS_val = PHIS_field.isel(time=0, lat=i, lon=j).item()

    # Interpret PHIS as geopotential (m^2/s^2) or height (m)
        if PHIS_val > 2e4:  # heuristic threshold; adjust if needed
          z_surf_model = (PHIS_val * units("m^2/s^2") / g).to("meter").magnitude
        else:
          z_surf_model = PHIS_val

    # Vertical profiles at that grid cell
        T_prof = ds_T["T"].isel(time=0, lat=i, lon=j).values  # (lev,)
        p_prof = ds_PL["PL"].isel(time=0, lat=i, lon=j).values  # (lev,)

        try:
            level_idx, p_level_hPa, z_level_m = metpy_find_level_index(
            p_prof_Pa=p_prof,
            T_prof_K=T_prof,
            station_alt_m=alt_s,
            z_surf_model=z_surf_model,
                )
        except Exception as e:
        # If something goes wrong, skip this station but report it if you want
        # print(f"{name}: error computing level -> {e}")
          continue

    # Count stations whose nearest level is not 71
        if level_idx != 71:
          non71_count += 1

    # Print only up to n_to_print stations
        if printed < n_to_print:
          print(f"{name}\t{alt_s:.1f}\t{level_idx}")
          printed += 1



    # Summary for all stations
    print(
        f"\nOut of {total_stations} stations, "
        f"{non71_count} do not have level 71 as the closest model level."
    )
