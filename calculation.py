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



def compute_ring_sector_masks(ii, jj, Ny, Nx, radii):
    """
    Build ring sectors around (ii, jj) for arbitrary radii.

    radii: list/tuple of integers, e.g. [1,2,3,4]
           Sector k is the square (radius=radii[k]) minus square (radius=radii[k-1]).
           The first sector is the (2*r+1)x(2*r+1) square including the center.

    Returns: list of masks [S1, S2, ...]
    """
    masks = []
    prev = np.zeros((Ny, Nx), dtype=bool)

    for r in radii:
        box = np.zeros((Ny, Nx), dtype=bool)
        box[safe_slice(ii - r, ii + r + 1, Ny),
            safe_slice(jj - r, jj + r + 1, Nx)] = True  #True 

        ring = box & (~prev)
        masks.append(ring)
        prev = box

    return masks

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




def sector_stats(df, var_name):
    """
    Compute summary stats for a sector:
    mean, std, CV, median, IQR (Q3-Q1), n
    """
    vals = pd.to_numeric(df[var_name], errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "cv": np.nan,
                "median": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan}

    mean = float(np.mean(vals))
    std = float(np.std(vals))
    cv = float(std / mean) if mean != 0 else np.nan

    q1 = float(np.percentile(vals, 25))
    median = float(np.percentile(vals, 50))
    q3 = float(np.percentile(vals, 75))
    iqr = float(q3 - q1)

    return {"n": int(vals.size), "mean": mean, "std": std, "cv": cv,
            "median": median, "q1": q1, "q3": q3, "iqr": iqr}

def compute_sector_tables_generic(ii, jj, lats_small, lons_small, data_arr, var_name, radii):
    Ny, Nx = data_arr.shape
    masks = compute_ring_sector_masks(ii, jj, Ny, Nx, radii)

    dfs = [sector_table(m, lats_small, lons_small, data_arr, var_name) for m in masks]
    return dfs, masks

def cumulative_sector_masks(sector_masks):
    """
    Build cumulative sector masks.

    Input
    -----
    sector_masks : list of boolean masks
        [S1, S2, S3, ...] where each is a disjoint ring

    Returns
    -------
    cumulative_masks : list of boolean masks
        [C1, C2, C3, ...]
        Ck = union of S1 ... Sk
    """
    cumulative_masks = []
    running = np.zeros_like(sector_masks[0], dtype=bool)

    for S in sector_masks:
        running = running | S
        cumulative_masks.append(running.copy())

    return cumulative_masks

def compute_cumulative_sector_tables(
    sector_masks,
    lats_small,
    lons_small,
    data_arr,
    var_name,
):
    """
    Build DataFrames for cumulative sectors.
    """
    cumulative_masks = cumulative_sector_masks(sector_masks)

    dfs = []
    for k, mask in enumerate(cumulative_masks, start=1):
        df = sector_table(mask, lats_small, lons_small, data_arr, var_name)
        dfs.append(df)

    return dfs, cumulative_masks

