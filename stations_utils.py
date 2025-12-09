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

def all_stations(ds_species, ds_T, ds_PL, ds_orog):
    """
    Loop over all stations, compute the nearest model level for each station,
    and print the result.

    Parameters
    ----------
    ds_species : xarray.Dataset
        Dataset that contains 'lat' and 'lon' (for horizontal grid).
    ds_T : xarray.Dataset
        Dataset that contains temperature field 'T' with dims ('time','lev','lat','lon').
    ds_PL : xarray.Dataset
        Dataset that contains pressure field 'PL' with dims ('time','lev','lat','lon').
    ds_orog : xarray.Dataset
        Dataset that contains orography fields 'PHIS' (and optionally 'SGH').
    """
    import numpy as np
    from metpy.units import units
    from metpy.constants import g
    from vertical_indexing import metpy_find_level_index
    from horizontal_indexing import nearest_grid_index
    from file_utils import stations_path

    # Load stations table
    stations = load_stations(stations_path)

    # Horizontal grid
    lats = ds_species["lat"].values
    lons = ds_species["lon"].values

    # Orography fields
    PHIS_field = ds_orog["PHIS"]

    for _, station in stations.iterrows():
        lat_s = float(station["Latitude"])
        lon_s = float(station["Longitude"])
        alt_s = float(station["Altitude"])
        name = station["Station_Name"]
        idx = int(station["idx"])

        # Nearest model grid cell for this station
        i, j = nearest_grid_index(lat_s, lon_s, lats, lons)

        # PHIS at that grid point (time dimension assumed to exist → take time=0)
        PHIS_val = PHIS_field.isel(time=0, lat=i, lon=j).item()

        # Interpret PHIS as geopotential (m^2/s^2) or height (m)
        if PHIS_val > 2e4:  # heuristic threshold; adjust if needed
            z_surf_model = (PHIS_val * units("m^2/s^2") / g).to("meter").magnitude
        else:
            z_surf_model = PHIS_val

        # Extract vertical profiles at that grid cell
        T_prof = ds_T["T"].isel(time=0, lat=i, lon=j).values  # (lev,)
        p_prof = ds_PL["PL"].isel(time=0, lat=i, lon=j).values  # (lev,)

        try:
            level_idx, p_level_hPa, z_level_m = metpy_find_level_index(
                p_prof_Pa=p_prof,
                T_prof_K=T_prof,
                station_alt_m=alt_s,
                z_surf_model=z_surf_model,
            )

            print(
                f"Station {name} (idx={idx}, alt={alt_s:.1f} m): "
                f"nearest level={level_idx}, p={p_level_hPa:.1f} hPa, z={z_level_m:.1f} m"
            )
        except Exception as e:
            print(
                f"Station {name} (idx={idx}): error computing level -> {e}"
            )
