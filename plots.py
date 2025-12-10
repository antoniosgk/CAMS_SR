# plots.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def plot_variable_on_map(
    lats_small,
    lons_small,
    data_arr,
    lon_s,
    lat_s,
    units="",
    species_name="var",
    proj=None,
    ax=None,
    d=0.8,
):
    """
    Plot a 2D variable (data_arr) on a map around a station.

    lats_small, lons_small: 1D or 2D lat/lon arrays matching data_arr
    data_arr              : 2D array (Ny, Nx)
    lon_s, lat_s          : station longitude/latitude
    units                 : string for colorbar label
    species_name          : used in title
    proj                  : cartopy CRS (defaults to PlateCarree)
    ax                    : existing GeoAxes (optional)
    d                     : half-width of domain in degrees around station
    """
    if proj is None:
        proj = ccrs.PlateCarree()

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": proj})
    else:
        fig = ax.figure

    # Map extent around station
    lon_min, lon_max = lon_s - d, lon_s + d
    lat_min, lat_max = lat_s - d, lat_s + d
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # Ticks
    xticks = np.arange(lon_min, lon_max + 0.1, 0.5)
    yticks = np.arange(lat_min, lat_max + 0.1, 0.5)
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    ax.set_xticklabels(xticks, rotation=45, ha="right")
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.3f}"))

    # Lon/lat grids
    if lats_small.ndim == 1 and lons_small.ndim == 1:
        LON2D, LAT2D = np.meshgrid(lons_small, lats_small)
    else:
        LAT2D = lats_small
        LON2D = lons_small

    # Color limits
    vmin = np.nanmin(data_arr)
    vmax = np.nanmax(data_arr)
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Main pcolormesh
    im = ax.pcolormesh(
        LON2D,
        LAT2D,
        data_arr,
        cmap="turbo",
        shading="auto",
        norm=norm,
        transform=ccrs.PlateCarree(),
    )

    # Station marker
    ax.plot(
        lon_s,
        lat_s,
        "kx",
        markersize=12,
        transform=ccrs.PlateCarree(),
    )

    # Features
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=1)

    return fig, ax, im


def plot_rectangles(
    ax,
    lats_small,
    lons_small,
    ii,
    jj,
    im,
    units="",
    species_name="var",
):
    """
    Plot the 3 sector rectangles (S1, S2, S3) around the central grid cell (ii, jj)
    on an existing map, and add colorbar + title.

    ax         : GeoAxes used in plot_variable_on_map
    lats_small : 1D or 2D lat array
    lons_small : 1D or 2D lon array
    ii, jj     : indices of the central grid cell in the small domain
    im         : mappable from pcolormesh (for colorbar)
    units      : colorbar label
    species_name: used in plot title
    """
    # Grid spacing in degrees
    if lons_small.ndim == 1:
        dlon = float(np.abs(lons_small[1] - lons_small[0]))
    else:
        dlon = float(np.abs(lons_small[0, 1] - lons_small[0, 0]))

    if lats_small.ndim == 1:
        dlat = float(np.abs(lats_small[1] - lats_small[0]))
    else:
        dlat = float(np.abs(lats_small[1, 0] - lats_small[0, 0]))

    # Sector sizes (radius in grid cells)
    sizes = {
        "S1": 1,  # 3x3
        "S2": 2,  # 5x5
        "S3": 3,  # 7x7
    }

    # Colors and linewidths
    rect_styles = {
        "S1": {"edgecolor": "black", "linewidth": 2},
        "S2": {"edgecolor": "red", "linewidth": 2},
        "S3": {"edgecolor": "blue", "linewidth": 2},
    }

    # Center of rectangles
    if lons_small.ndim == 1 and lats_small.ndim == 1:
        cx = float(lons_small[jj])
        cy = float(lats_small[ii])
    else:
        cx = float(lons_small[ii, jj])
        cy = float(lats_small[ii, jj])

    for name, r in sizes.items():
        width = (2 * r + 1) * dlon
        height = (2 * r + 1) * dlat
        left = cx - width / 2
        bottom = cy - height / 2

        rect = Rectangle(
            (left, bottom),
            width,
            height,
            facecolor="none",
            transform=ccrs.PlateCarree(),
            **rect_styles[name],
        )
        ax.add_patch(rect)

    # Colorbar and title
    plt.colorbar(im, ax=ax, pad=0.02, label=units)
    ax.set_title(f"{species_name} around station")



def _sort_by_pressure(p_hPa, *arrays):
    """
    Helper: sort profiles from surface (max p) to top (min p).
    Returns sorted p_hPa and all input arrays in the same order.
    """
    p_hPa = np.asarray(p_hPa)
    order = np.argsort(p_hPa)[::-1]  # descending: surface → top
    sorted_arrays = [p_hPa[order]]
    for arr in arrays:
        if arr is None:
            sorted_arrays.append(None)
        else:
            arr = np.asarray(arr)
            sorted_arrays.append(arr[order])
    return sorted_arrays


def plot_profile_P_T(p_prof_Pa, T_prof_K, ax=None):
    """
    Plot vertical profile: Pressure (hPa) vs Temperature (°C)
    for a single grid cell.

    p_prof_Pa : 1D array, pressure in Pa
    T_prof_K  : 1D array, temperature in K (same levels as p_prof_Pa)
    ax        : optional matplotlib Axes. If None, a new figure/axes is created.

    Returns (fig, ax).
    """
    p_hPa = np.asarray(p_prof_Pa) / 100.0
    T_C = np.asarray(T_prof_K) - 273.15

    # Sort from surface (max p) to top (min p)
    p_hPa, T_C = _sort_by_pressure(p_hPa, T_C)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(T_C, p_hPa, marker="o")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title("Vertical profile: P–T")
    ax.invert_yaxis()  # pressure decreases upward

    ax.grid(True, linestyle="--", alpha=0.5)
    return fig, ax


def plot_profile_P_z(p_prof_Pa, z_prof_m, ax=None, z_units="km"):
    """
    Plot vertical profile: Pressure (hPa) vs Height (z).

    p_prof_Pa : 1D array, pressure in Pa
    z_prof_m  : 1D array, height in meters (ASL) for the same levels
    ax        : optional matplotlib Axes
    z_units   : "km" or "m" for x-axis units

    Returns (fig, ax).
    """
    p_hPa = np.asarray(p_prof_Pa) / 100.0
    z_m = np.asarray(z_prof_m)

    # Sort from surface (max p) to top (min p)
    p_hPa, z_m = _sort_by_pressure(p_hPa, z_m)

    if z_units == "km":
        z_vals = z_m / 1000.0
        xlabel = "Height (km)"
    else:
        z_vals = z_m
        xlabel = "Height (m)"

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(z_vals, p_hPa, marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pressure (hPa)")
    ax.set_title("Vertical profile: P–z")
    ax.invert_yaxis()

    ax.grid(True, linestyle="--", alpha=0.5)
    return fig, ax


def plot_profile_T_z(T_prof_K, z_prof_m, ax=None, z_units="km"):
    """
    Plot vertical profile: Temperature (°C) vs Height (z).

    T_prof_K  : 1D array, temperature in K
    z_prof_m  : 1D array, height in meters (ASL) for the same levels
    ax        : optional matplotlib Axes
    z_units   : "km" or "m" for y-axis units

    Returns (fig, ax).
    """
    T_C = np.asarray(T_prof_K) - 273.15
    z_m = np.asarray(z_prof_m)

    # Sort by height ascending (ground → top)
    order = np.argsort(z_m)
    z_m = z_m[order]
    T_C = T_C[order]

    if z_units == "km":
        z_vals = z_m / 1000.0
        ylabel = "Height (km)"
    else:
        z_vals = z_m
        ylabel = "Height (m)"

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(T_C, z_vals, marker="o")
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(ylabel)
    ax.set_title("Vertical profile: T–z")

    ax.grid(True, linestyle="--", alpha=0.5)
    return fig, ax
