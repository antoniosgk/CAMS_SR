# plots.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import os
import re

def _sanitize_filename(s: str) -> str:
    # keep it filesystem-safe
    s = re.sub(r"\s+", "_", str(s).strip())
    s = re.sub(r"[^A-Za-z0-9_.-]+", "", s)
    return s


def save_figure(fig, out_dir, filename_base, dpi=200):
    """
    Save a matplotlib figure as PNG into out_dir with a safe filename.
    """
    os.makedirs(out_dir, exist_ok=True)
    fname = _sanitize_filename(filename_base) + ".png"
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path

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
    d=5,
    time_str=None
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
    time_str              : optional time string for time
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
    #ax.add_feature(cfeature.ShadedRelief(), zorder=0)
    #ax.stock_img() # background topography-like
    #ax.set_xticks(xticks, crs=proj)
    #ax.set_yticks(yticks, crs=proj)

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
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
        cmap="viridis",
        shading="auto",
        norm=norm,
        transform=ccrs.PlateCarree(),alpha=0.6,zorder=2
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
    plt.colorbar(im, ax=ax, pad=0.02, label=units)
    step = 0.5  # degrees

    # Build tick values aligned to step
    lon0 = np.floor(lon_min / step) * step
    lon1 = np.ceil(lon_max / step) * step
    lat0 = np.floor(lat_min / step) * step
    lat1 = np.ceil(lat_max / step) * step

    xticks = np.round(np.arange(lon0, lon1 + 0.5 * step, step), 6)
    yticks = np.round(np.arange(lat0, lat1 + 0.5 * step, step), 6)

    gl = ax.gridlines(
     crs=ccrs.PlateCarree(),
     draw_labels=True,
     linewidth=0,
     alpha=0.01,          # set this >0 so gridlines are visible; labels still show even if tiny
     linestyle="--",
     zorder=5,
    )

    gl.top_labels = False
    gl.right_labels = False

    gl.xlocator = ticker.FixedLocator(xticks)
    gl.ylocator = ticker.FixedLocator(yticks)

    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

# Optional: nicer label styling
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    if units:
        base_title = f"{species_name} ({units})"
    else:
        base_title = species_name

    if time_str is not None:
        title = f"{base_title} at {time_str}.6f"
    else:
        title = base_title

    ax.set_title(title)


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
    time_str= None
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
    #plt.colorbar(im, ax=ax, pad=0.02, label=units)
    # Title consistent with map
    if units:
        base_title = f"{species_name} ({units})"
    else:
        base_title = species_name

    if time_str is not None:
        title = f"{base_title} at {time_str}"
    else:
        title = base_title

    ax.set_title(f"{title} around station")
    
    return ax,im


def _sort_by_pressure_with_index(p_hPa, idx_level, *arrays):
    """
    Sort profiles from surface (max p) to top (min p),
    and return the new index of the selected level.

    Returns
    -------
    p_sorted, arrays_sorted..., idx_sorted
    """
    p_hPa = np.asarray(p_hPa)
    order = np.argsort(p_hPa)[::-1]  # descending: surface → top

    p_sorted = p_hPa[order]
    sorted_arrays = []
    for arr in arrays:
        if arr is None:
            sorted_arrays.append(None)
        else:
            arr = np.asarray(arr)
            sorted_arrays.append(arr[order])

    # find where the original idx_level moved to
    idx_sorted = int(np.where(order == idx_level)[0][0])

    return (p_sorted, *sorted_arrays, idx_sorted)

def plot_profile_P_T(p_prof_Pa, T_prof_K, idx_level,
                     time_str=None, ax=None):
    """
    Plot vertical profile: Pressure (hPa) vs Temperature (°C)
    for a single grid cell, with a red dot at idx_level.

    Parameters
    ----------
    p_prof_Pa : 1D array, pressure in Pa
    T_prof_K  : 1D array, temperature in K
    idx_level : int, selected model level index (0-based)
    time_str  : optional, string to show in title (e.g. '2025-12-15 00:00 UTC')
    ax        : optional matplotlib Axes

    Returns (fig, ax)
    """
    p_hPa = np.asarray(p_prof_Pa) / 100.0
    T_C = np.asarray(T_prof_K) - 273.15

    # Sort from surface (max p) to top (min p), track level index
    p_sorted, T_sorted, idx_sorted = _sort_by_pressure_with_index(
        p_hPa, idx_level, T_C
    )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # main profile
    ax.plot(T_sorted, p_sorted, "-o")

    # red dot at selected level
    ax.scatter(T_sorted[idx_sorted], p_sorted[idx_sorted],
               color="red", zorder=3, label="Selected level")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pressure (hPa)")
    title = "Vertical profile: P–T"
    if time_str is not None:
        title += f" at {time_str}"
    ax.set_title(title)

    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    return fig, ax

def plot_profile_T_Z(T_prof_K, z_prof_m, idx_level,
                     time_str=None, z_units="km", ax=None):
    """
    Plot vertical profile: Temperature (°C) vs Height (Z),
    with red dot at idx_level.

    Parameters
    ----------
    T_prof_K  : 1D array, temperature in K
    z_prof_m  : 1D array, height in m (ASL)
    idx_level : int, selected model level index
    time_str  : optional, string for title
    z_units   : 'km' or 'm'
    ax        : optional Axes

    Returns (fig, ax)
    """
    T_C = np.asarray(T_prof_K) - 273.15
    z_m = np.asarray(z_prof_m)

    # sort by height ascending (surface → top)
    order = np.argsort(z_m)
    z_sorted = z_m[order]
    T_sorted = T_C[order]
    idx_sorted = int(np.where(order == idx_level)[0][0])

    if z_units == "km":
        z_vals = z_sorted / 1000.0
        ylabel = "Height (km)"
    else:
        z_vals = z_sorted
        ylabel = "Height (m)"

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(T_sorted, z_vals, "-o")
    ax.scatter(T_sorted[idx_sorted], z_vals[idx_sorted],
               color="red", zorder=3, label="Selected level")

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel(ylabel)
    title = "Vertical profile: T–Z"
    if time_str is not None:
        title += f" at {time_str}"
    ax.set_title(title)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    return fig, ax

def plot_profile_T_logP(p_prof_Pa, T_prof_K, idx_level, time_str=None, ax=None):
    """
    Temperature vs Pressure with log-pressure vertical axis (ticks labeled in hPa).
    Red dot indicates the selected model level.
    """
    p_hPa = np.asarray(p_prof_Pa) / 100.0
    T_C = np.asarray(T_prof_K) - 273.15

    # Sort surface→top and track selected index
    p_sorted, T_sorted, idx_sorted = _sort_by_pressure_with_index(p_hPa, idx_level, T_C)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(T_sorted, p_sorted, "-o")
    ax.scatter(T_sorted[idx_sorted], p_sorted[idx_sorted],
               color="red", zorder=3, label="Selected level")

    # Log pressure axis with nice pressure ticks
    ax.set_yscale("log")          # base-10 log axis (standard)
    ax.invert_yaxis()             # pressure decreases upward

    p_ticks = [1000, 700, 500, 300, 200, 100, 70, 50, 30, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1,0.01]
    p_ticks = [t for t in p_ticks if p_sorted.min() <= t <= p_sorted.max()]
    if p_ticks:
        ax.yaxis.set_major_locator(ticker.FixedLocator(p_ticks))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pressure (hPa)")

    title = "Vertical profile: T–log(P)"
    if time_str is not None:
        title += f" at {time_str}"
    ax.set_title(title)

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    return fig, ax

def plot_profile_species_logP(p_prof_Pa, species_prof, idx_level,
                              species_name="species", species_units="",
                              time_str=None, ax=None):
    """
    Species vs Pressure with log-pressure vertical axis (ticks labeled in hPa).
    Red dot indicates the selected model level.
    """
    p_hPa = np.asarray(p_prof_Pa) / 100.0
    sp = np.asarray(species_prof)

    p_sorted, sp_sorted, idx_sorted = _sort_by_pressure_with_index(p_hPa, idx_level, sp)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(sp_sorted, p_sorted, "-o")
    ax.scatter(sp_sorted[idx_sorted], p_sorted[idx_sorted],
               color="red", zorder=3, label="Selected level")

    # Log pressure axis with nice pressure ticks
    ax.set_yscale("log")
    ax.invert_yaxis()

    p_ticks = [1000, 700, 500, 300, 200, 100, 70, 50, 30, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1,0.01]
    p_ticks = [t for t in p_ticks if p_sorted.min() <= t <= p_sorted.max()]
    if p_ticks:
        ax.yaxis.set_major_locator(ticker.FixedLocator(p_ticks))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))

    xlabel = species_name
    if species_units:
        xlabel += f" ({species_units})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pressure (hPa)")

    title = f"{species_name}–log(P)"
    if time_str is not None:
        title += f" at {time_str}"
    ax.set_title(title)

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    return fig, ax

def plot_profile_species_Z(z_prof_m, species_prof, idx_level,
                           species_name="species", species_units="",
                           time_str=None, z_units="km", ax=None):
    """
    Plot vertical profile: species vs Height (Z),
    with red dot at idx_level.

    Parameters
    ----------
    z_prof_m     : 1D array, height in m (ASL)
    species_prof : 1D array, species values
    idx_level    : int, selected model level index
    species_name : name of species
    species_units: units of species
    time_str     : optional string for title
    z_units      : 'km' or 'm'
    ax           : optional Axes

    Returns (fig, ax)
    """
    z_m = np.asarray(z_prof_m)
    sp = np.asarray(species_prof)

    # sort by height ascending
    order = np.argsort(z_m)
    z_sorted = z_m[order]
    sp_sorted = sp[order]
    idx_sorted = int(np.where(order == idx_level)[0][0])

    if z_units == "km":
        z_vals = z_sorted / 1000.0
        ylabel = "Height (km)"
    else:
        z_vals = z_sorted
        ylabel = "Height (m)"

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(sp_sorted, z_vals, "-o")
    ax.scatter(sp_sorted[idx_sorted], z_vals[idx_sorted],
               color="red", zorder=3, label="Selected level")

    xlabel = species_name
    if species_units:
        xlabel += f" ({species_units})"

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    title = f"{species_name}–Z"
    if time_str is not None:
        title += f" at {time_str}"
    ax.set_title(title)

    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    return fig, ax
