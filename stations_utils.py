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
# %%
