#%%
import pandas as pd
import numpy as np
from file_utils import base_path,stations_path
from vertical_indexing import metpy_find_level_index
from horizontal_indexing import nearest_grid_index
#%%
def load_stations(stations_path):
    """Load station table. Keeps all rows; marks invalid rows (NaNs) for later filtering."""
    df = pd.read_csv(stations_path, sep="\t")
    df = df.reset_index().rename(columns={"index": "idx"})

    # normalize column names
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
        raise ValueError(f"Stations file missing expected columns: {missing}")

    # coerce numeric; DO NOT drop rows here
    for col in ["Latitude", "Longitude", "Altitude"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # mark validity (so bulk loops can filter)
    df["is_valid"] = df[["Latitude", "Longitude", "Altitude"]].notna().all(axis=1)

    return df[expected + ["is_valid"]]



def select_station(df, idx=None, name=None):
    """Return station row by idx or name. If station exists but has NaNs, raise a clear error."""
    if idx is None and name is None:
        print(df[["idx", "Station_Name", "is_valid"]])
        s = input("Select station by index or name: ").strip()
    else:
        s = str(idx if idx is not None else name)

    row = None

    if s.isdigit():
        r = df[df["idx"] == int(s)]
        if not r.empty:
            row = r.iloc[0]
    else:
        r = df[df["Station_Name"] == s]
        if not r.empty:
            row = r.iloc[0]

    if row is None:
        raise ValueError(f"Station '{s}' not found.")

    if not bool(row.get("is_valid", True)):
        raise ValueError(
            f"Station '{row['Station_Name']}' (idx={row['idx']}) has missing "
            f"Latitude/Longitude/Altitude in the stations file; cannot run calculations."
        )

    return row


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

    t = 1  # or set this from outside / function argument

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
    stations_valid = stations[stations["is_valid"]].copy()
    for _, station in stations_valid.iterrows():
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


        z_surf_model = (PHIS_val * units("m^2/s^2") / g).to("meter").magnitude
        

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
        '''if printed < n_to_print:
          print(f"{name}\t{alt_s:.1f}\t{level_idx}")
          printed += 1'''



    # Summary for all stations
    print(
        f"\nOut of {total_stations} stations, "
        f"{non71_count} do not have level 71 as the closest model level."
    )
def map_stations_to_model_levels(
    stations,
    ds_T,
    ds_PL,
    ds_orog,
    lats,
    lons,
    RH_ds=None,
):
    # --- Drop stations with invalid lat/lon/alt ---
    stations_clean = stations.copy()

    for col in ["Latitude", "Longitude", "Altitude"]:
     stations_clean[col] = pd.to_numeric(stations_clean[col], errors="coerce")

    stations_clean = stations_clean.dropna(
    subset=["Latitude", "Longitude", "Altitude"]
)

    stations_clean = stations_clean.reset_index(drop=True)

    records = []
    non_surface_records=[]
    not_surface_count = 0
    surface_level = 22

    stations_valid = stations[stations["is_valid"]].copy()
    for _, station in stations_valid.iterrows():
        lat_s = float(station["Latitude"])
        lon_s = float(station["Longitude"])
        alt_s = float(station["Altitude"])
        name = station["Station_Name"]

        # nearest grid cell
        i, j = nearest_grid_index(lat_s, lon_s, lats, lons)

        # surface height
        PHIS_val = ds_orog["PHIS"].isel(time=0, lat=i, lon=j).item()
        z_surf_model = PHIS_val / 9.80665  # geopotential → height (m)

        # profiles
        T_prof = ds_T["T"].values[0, :, i, j]
        p_prof = ds_PL["PL"].values[0, :, i, j]

        RH_prof = None
        if RH_ds is not None:
            RH_prof = RH_ds["RH"].values[0, :, i, j]

        # vertical indexing
        idx_level, p_hPa, z_level = metpy_find_level_index(
            p_prof_Pa=p_prof,
            T_prof_K=T_prof,
            RH=RH_prof,
            station_alt_m=alt_s,
            z_surf_model=z_surf_model,
        )

        if idx_level != np.argmax(p_prof):
            not_surface_count += 1
            non_surface_records.append({
            "station_name": name,
            "lat": lat_s,
            "lon": lon_s,
            "alt_m": alt_s,
            "model_level": int(idx_level),
            "model_pressure_hPa": float(p_hPa),
            "model_height_m": int(idx_level),
    })
        records.append({
            "station": name,
            "lat": lat_s,
            "lon": lon_s,
            "altitude_m": alt_s,
            "model_lat": float(lats[i]),
            "model_lon": float(lons[j]),
            "model_level": int(idx_level),
            "model_pressure_hPa": float(p_hPa),
            "model_height_m": float(z_level),
        })
        

    df = pd.DataFrame(records)
    n_non_surface = len(non_surface_records)

    print(f"\nStations NOT mapped to surface model level ({surface_level}): "
      f"{n_non_surface}")

    if n_non_surface > 0:
     
     df_non_surface = pd.DataFrame(non_surface_records)
     df_non_surface.to_csv(
        "stations_not_on_surface_level.csv",
        index=False
    )    

    return df, not_surface_count,df_non_surface

