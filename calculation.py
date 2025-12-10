# calculation.py

import numpy as np
import pandas as pd


def safe_slice(low, high, maxN):
    """
    Safe slice helper:
    - low, high are like Python slice bounds (high is exclusive)
    - maxN is the size of the axis
    """
    return slice(max(low, 0), min(high, maxN))


def compute_sector_masks(ii, jj, Ny, Nx):
    """
    Compute boolean masks S1, S2, S3 around central grid cell (ii, jj).

    S1: 3x3 box around (ii, jj)
    S2: 5x5 box minus S1
    S3: 7x7 box minus S1 and S2
    """
    # Sector 1
    S1 = np.zeros((Ny, Nx), dtype=bool)
    S1[safe_slice(ii - 1, ii + 2, Ny),
       safe_slice(jj - 1, jj + 2, Nx)] = True
    S1[ii, jj] = True

    # Sector 2
    S2 = np.zeros((Ny, Nx), dtype=bool)
    S2[safe_slice(ii - 2, ii + 3, Ny),
       safe_slice(jj - 2, jj + 3, Nx)] = True
    S2[S1] = False

    # Sector 3
    S3 = np.zeros((Ny, Nx), dtype=bool)
    S3[safe_slice(ii - 3, ii + 4, Ny),
       safe_slice(jj - 3, jj + 4, Nx)] = True
    S3[S1] = False
    S3[S2] = False

    return S1, S2, S3


def sector_table(mask, lats_small, lons_small, data_arr, var_name):
    """
    Build a DataFrame with indices and values for a given sector mask.

    mask      : boolean 2D array (Ny, Nx)
    lats_small: 1D or 2D array of latitudes matching mask shape
    lons_small: 1D or 2D array of longitudes matching mask shape
    data_arr  : 2D array (Ny, Nx) with variable values
    var_name  : column name for the variable
    """
    iy, ix = np.where(mask)

    # Handle 1D vs 2D lat/lon
    if lats_small.ndim == 1 and lons_small.ndim == 1:
        lat_vals = lats_small[iy]
        lon_vals = lons_small[ix]
    else:
        lat_vals = lats_small[iy, ix]
        lon_vals = lons_small[iy, ix]

    vals = data_arr[iy, ix]

    return pd.DataFrame({
        "lat_idx": iy,
        "lon_idx": ix,
        "lat": lat_vals,
        "lon": lon_vals,
        var_name: vals,
    })


def compute_sector_tables(ii, jj, lats_small, lons_small, data_arr, var_name):
    """
    Convenience helper: compute S1/S2/S3 masks and return their tables.
    """
    Ny, Nx = data_arr.shape
    S1, S2, S3 = compute_sector_masks(ii, jj, Ny, Nx)

    df_S1 = sector_table(S1, lats_small, lons_small, data_arr, var_name)
    df_S2 = sector_table(S2, lats_small, lons_small, data_arr, var_name)
    df_S3 = sector_table(S3, lats_small, lons_small, data_arr, var_name)
    print(df_S1)
    return df_S1, df_S2, df_S3, S1, S2, S3
