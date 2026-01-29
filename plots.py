# plots.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from calculation import weighted_quantile, haversine_km
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import os
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd



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

def build_meta_title(meta, kind=""):
    """
    kind: optional short label like 'Map', 'Profile P–T', etc.
    """
    if meta is None:
        return kind

    header = f"{meta['species']} ({meta['units']}) at {meta['time_str']}"
    if kind:
        header = f"{kind} | " + header

    line2 = (
        f"Station {meta['station_name']}: "
        f"({meta['station_lat']:.2f}, {meta['station_lon']:.2f}), alt={meta['station_alt']:.1f} m"
    )
    line3 = (
        f"Model: ({meta['model_lat']:.2f}, {meta['model_lon']:.2f}), "
        f"lev={meta['model_level']}, alt={meta['z_level_m']:.1f} m"
    )
    return header + "\n" + line2 + "\n" + line3


def plot_variable_on_map(
    lats_small,
    lons_small,
    data_arr,
    # station
    lon_s,
    lat_s,
    # labels/meta
    units="",
    species_name="var",
    time_str=None,
    meta=None,
    # map control
    d=0.4,
    proj=None,
    ax=None,
    # terrain background on a DIFFERENT grid (always 1D)
    lats_terrain=None,
    lons_terrain=None,
    z_orog_m=None,
    terrain_alpha=0.5,
    field_alpha=0.8,
    add_orog_contours=True,
    plot_species=True,plot_orography=False
    ):
    

    # --- Projection / axes (must be GeoAxes) ---
    if proj is None:
        proj = ccrs.PlateCarree()

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": proj})
    else:
        fig = ax.figure

    # --- extent around station (lon/lat degrees => PlateCarree) ---
    lon_min, lon_max = lon_s - d, lon_s + d
    lat_min, lat_max = lat_s - d, lat_s + d
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # --- ocean background first (so masked sea shows this color) ---
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=-2)

    # --- TERRAIN underlay (optional) ---
    terrain_im = None
    if plot_orography and (z_orog_m is not None) and (lats_terrain is not None) and (lons_terrain is not None):
        lats_terrain = np.asarray(lats_terrain, dtype=float)
        lons_terrain = np.asarray(lons_terrain, dtype=float)
        z_orog_m = np.asarray(z_orog_m, dtype=float)

        LON_T, LAT_T = np.meshgrid(lons_terrain, lats_terrain)

        # Mask sea so it doesn't use the terrain colormap
        z_plot = np.ma.masked_where(~np.isfinite(z_orog_m) | (z_orog_m <= 0.0), z_orog_m)

        # Use land-only min/max for better contrast (avoid sea pulling vmin down)
        if np.ma.is_masked(z_plot):
            zmin = float(z_plot.min())
            zmax = float(z_plot.max())
        else:
            zmin = float(np.nanmin(z_orog_m))
            zmax = float(np.nanmax(z_orog_m))

        terrain_im = ax.pcolormesh(
            LON_T, LAT_T, z_plot,
            cmap="terrain",
            shading="auto",
            vmin=zmin, vmax=zmax,
            alpha=terrain_alpha,
            transform=ccrs.PlateCarree(),
            zorder=-1,
        )

        if add_orog_contours:
            step_m = 200.0
            levels = np.arange(np.floor(zmin / step_m) * step_m,
                               np.ceil(zmax / step_m) * step_m + step_m,
                               step_m)
            ax.contour(
                LON_T, LAT_T, z_orog_m,
                levels=levels,
                colors="k",
                linewidths=0.4,
                alpha=0.30,
                transform=ccrs.PlateCarree(),
                zorder=0,
            )

    # --- Species overlay (small domain) ---
    lats_small = np.asarray(lats_small, dtype=float)
    lons_small = np.asarray(lons_small, dtype=float)
    data_arr = np.asarray(data_arr, dtype=float)

    

    vmin = float(np.nanmin(data_arr))
    vmax = float(np.nanmax(data_arr))
    norm = Normalize(vmin=vmin, vmax=vmax)

    im = None
    if plot_species:
      LON_S, LAT_S = np.meshgrid(lons_small, lats_small)
      im = ax.pcolormesh(
        LON_S, LAT_S, data_arr,
        cmap="viridis",
        shading="auto",
        norm=norm,
        transform=ccrs.PlateCarree(),
        alpha=field_alpha,
        zorder=2,
    )


    # station marker
    ax.plot(
        lon_s, lat_s, "kx",
        markersize=12,
        transform=ccrs.PlateCarree(),
        zorder=4,
    )

    # coast/borders on top
    ax.coastlines(resolution="10m", linewidth=0.8)
    #ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=5)

    # --- Two colorbars (terrain left, species right) ---
    cb_w=0.02
    cb_h=0.60
    cb_y=0.18
    #left colorbar(terrain)
    if terrain_im is not None:
        cax_terr=fig.add_axes([0.01,cb_y,cb_w,cb_h])
        cb_terr=fig.colorbar(terrain_im,cax=cax_terr)
        cb_terr.set_label("Elevation (m)")

    #right colorbar (species)
    if plot_species:
      cax_sp=fig.add_axes([0.8,cb_y,cb_w,cb_h])
      cb_sp=fig.colorbar(im,cax=cax_sp)
      cb_sp.set_label(units)    
    
    if not plot_species and meta is not None:
      ax.set_title(
        f"Station {meta['station_name']} "
        f"({meta['station_lat']:.3f}°, {meta['station_lon']:.3f}°), "
        f"{meta['station_alt']:.0f} m ASL\n"
        "Topography map",
        pad=18,
    )


    # -----------------------------
    # Gridlines with DMS labels
    # -----------------------------
    step = 0.4  # degrees
    lon0 = np.floor(lon_min / step) * step
    lon1 = np.ceil(lon_max / step) * step
    lat0 = np.floor(lat_min / step) * step
    lat1 = np.ceil(lat_max / step) * step

    xticks = np.round(np.arange(lon0, lon1 + 0.5 * step, step), 6)
    yticks = np.round(np.arange(lat0, lat1 + 0.5 * step, step), 6)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.6,
        alpha=0.35,
        linestyle="--",
        zorder=5,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = ticker.FixedLocator(xticks)
    gl.ylocator = ticker.FixedLocator(yticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}

    ax.set_title(build_meta_title(meta, kind="Map with Station"), pad=18)
    fig.subplots_adjust(left=0.1,right=0.9,top=0.82,bottom=0.1)  # keep title readable

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
    time_str=None,
    meta=None,
    radii=None
):
    import numpy as np
    import cartopy.crs as ccrs
    from matplotlib.patches import Rectangle

    # ---------- SAFETY ----------
    if radii is None:
        raise ValueError("plot_rectangles: 'radii' must be provided (e.g. [1,2,3]).")

    lats_small = np.asarray(lats_small, dtype=float)
    lons_small = np.asarray(lons_small, dtype=float)

    if not (0 <= ii < len(lats_small) and 0 <= jj < len(lons_small)):
        raise IndexError(f"(ii,jj)=({ii},{jj}) out of bounds.")

    # ---------- GRID SPACING ----------
    dlon = float(np.abs(lons_small[1] - lons_small[0]))
    dlat = float(np.abs(lats_small[1] - lats_small[0]))

    cx = float(lons_small[jj])
    cy = float(lats_small[ii])

    colors = ["black", "red", "blue", "orange", "purple", "brown"]

    # ---------- DRAW RECTANGLES ----------
    for k, r in enumerate(radii, start=1):
        color = colors[(k - 1) % len(colors)]

        width = (2 * r + 1) * dlon
        height = (2 * r + 1) * dlat
        left = cx - width / 2
        bottom = cy - height / 2

        rect = Rectangle(
            (left, bottom),
            width,
            height,
            facecolor="none",
            edgecolor=color,       
            linewidth=2.5,         
            transform=ccrs.PlateCarree(),
            zorder=20              # always above fields & terrain
        )
        ax.add_patch(rect)

    # ---------- TITLE ----------
    if meta is not None:
        ax.set_title(build_meta_title(meta, kind="Map with Sectors"), pad=22)
        ax.figure.subplots_adjust(top=0.84)

    return ax, im



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
                     time_str=None, ax=None,meta=None):
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
    ax.set_title(build_meta_title(meta, kind="Profile T-P"))
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    return fig, ax

def plot_profile_T_Z(T_prof_K, z_prof_m, idx_level,
                     time_str=None, z_units="km", ax=None,meta=None):
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
    ax.set_title(build_meta_title(meta, kind="Profile T–Z"))
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    return fig, ax

def plot_profile_T_logP(p_prof_Pa, T_prof_K, idx_level, time_str=None, ax=None,meta=None):
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

    p_ticks = [1000, 700, 500, 300, 200, 100, 70, 50, 30, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1,0.05]
    p_ticks = [t for t in p_ticks if p_sorted.min() <= t <= p_sorted.max()]
    if p_ticks:
        ax.yaxis.set_major_locator(ticker.FixedLocator(p_ticks))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Pressure (hPa)")

    ax.set_title(build_meta_title(meta, kind="Profile T–logP"))

    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    return fig, ax

def plot_profile_species_logP(p_prof_Pa, species_prof, idx_level,
                              species_name="species", species_units="",
                              time_str=None, ax=None,meta=None):
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

    p_ticks = [1000, 700, 500, 300, 200, 100, 70, 50, 30, 20, 10, 5, 2, 1, 0.5, 0.2, 0.1,0.05]
    p_ticks = [t for t in p_ticks if p_sorted.min() <= t <= p_sorted.max()]
    if p_ticks:
        ax.yaxis.set_major_locator(ticker.FixedLocator(p_ticks))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g}"))

    xlabel = species_name
    if species_units:
        xlabel += f" ({species_units})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pressure (hPa)")
    
    ax.set_title(build_meta_title(meta, kind=f"Profile{species_name} -Pressure"))


    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    return fig, ax

def plot_profile_species_Z(z_prof_m, species_prof, idx_level,
                           species_name="species", species_units="",
                           time_str=None, z_units="km", ax=None,meta=None):
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

    ax.set_title(build_meta_title(meta, kind="Profile concentration-Height"))


    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best")

    return fig, ax



def plot_sector_boxplots(sector_dfs, var_name, sector_names=None, title=None, ax=None):
    """
    Boxplots of var_name for multiple sectors.
    sector_dfs: list of DataFrames [df_S1, df_S2, ...]
    """
    if sector_names is None:
        sector_names = [f"S{k+1}" for k in range(len(sector_dfs))]

    data = []
    for df in sector_dfs:
        vals = pd.to_numeric(df[var_name], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        data.append(vals)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.boxplot(data, labels=sector_names, showfliers=False)
    ax.set_ylabel(var_name)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig, ax

def plot_sector_boxplots_weighted(sector_dfs, var_name, w_col="w_area",
                                  sector_names=None, title=None, ax=None):
    """
    Weighted boxplots using bxp() and weighted quantiles.
    Shows Q1/median/Q3 based on weights; whiskers use min/max of values.
    """
    if sector_names is None:
        sector_names = [f"S{k+1}" for k in range(len(sector_dfs))]

    bxp_stats = []
    for name, df in zip(sector_names, sector_dfs):
        vals = pd.to_numeric(df[var_name], errors="coerce").to_numpy(dtype=float)
        w = pd.to_numeric(df[w_col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(vals) & np.isfinite(w) & (w > 0)
        vals = vals[m]
        w = w[m]

        if vals.size == 0:
            q1 = med = q3 = lo = hi = np.nan
        else:
            q1 = weighted_quantile(vals, w, 0.25)
            med = weighted_quantile(vals, w, 0.50)
            q3 = weighted_quantile(vals, w, 0.75)
            lo = float(np.nanmin(vals))
            hi = float(np.nanmax(vals))

        bxp_stats.append(dict(label=name, q1=q1, med=med, q3=q3, whislo=lo, whishi=hi))

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.bxp(bxp_stats, showfliers=False)
    ax.set_ylabel(var_name)
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    return fig, ax


def box_stats_from_df(df, var_col, w_col=None, label=""):
    import numpy as np
    x = df[var_col].to_numpy(dtype=float)

    if w_col is None:
        x = x[np.isfinite(x)]
        q25, med, q75 = np.quantile(x, [0.25, 0.5, 0.75]) if x.size else (np.nan, np.nan, np.nan)
        lo = np.nanmin(x) if x.size else np.nan
        hi = np.nanmax(x) if x.size else np.nan
    else:
        w = df[w_col].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(w) & (w > 0)
        x = x[m]; w = w[m]
        if x.size:
            q25 = weighted_quantile(x, w, 0.25)
            med = weighted_quantile(x, w, 0.50)
            q75 = weighted_quantile(x, w, 0.75)
            lo = np.nanmin(x)
            hi = np.nanmax(x)
        else:
            q25 = med = q75 = lo = hi = np.nan

    return dict(label=label, q1=q25, med=med, q3=q75, whislo=lo, whishi=hi)

def plot_cv_ring_sectors(stats_unw, stats_w, title=None, ax=None):
    """
    Line plot of CV (unweighted vs weighted) for ring sectors.

    stats_unw : list of dicts from sector_stats_unweighted (S1, S2, ...)
    stats_w   : list of dicts from sector_stats_weighted   (S1, S2, ...)
    """

    cv_unw = [d["cv"] for d in stats_unw]
    cv_w   = [d["cv_w"] for d in stats_w]
    x = np.arange(1, len(cv_unw) + 1)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(x, cv_unw, marker="o", label="CV (unweighted)")
    ax.plot(x, cv_w, marker="s", label="CV (area-weighted)")

    ax.set_xlabel("Ring sector")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{k}" for k in x])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    if title:
        ax.set_title(title)

    return fig, ax
def plot_cv_ring_sectors(stats_unw, stats_w, title=None, ax=None):
    """
    Line plot of CV (unweighted vs area-weighted) for ring sectors S1, S2, ...
    """

    cv_unw = [d["cv"] for d in stats_unw]
    cv_w   = [d["cv_w"] for d in stats_w]
    x = np.arange(1, len(cv_unw) + 1)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(x, cv_unw, marker="o", linewidth=2,
            label="CV (unweighted)")
    ax.plot(x, cv_w, marker="s", linewidth=2,
            label="CV (area-weighted)")

    ax.set_xlabel("Ring sector")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{k}" for k in x])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    if title:
        ax.set_title(title)

    return fig, ax
def plot_cv_cumulative_sectors(stats_unw, stats_w, title=None, ax=None):
    """
    Line plot of CV (unweighted vs area-weighted)
    for cumulative sectors C1, C2, ...
    """

    cv_unw = [d["cv"] for d in stats_unw]
    cv_w   = [d["cv_w"] for d in stats_w]
    x = np.arange(1, len(cv_unw) + 1)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(x, cv_unw, marker="o", linewidth=2,
            label="CV (unweighted)")
    ax.plot(x, cv_w, marker="s", linewidth=2,
            label="CV (area-weighted)")

    ax.set_xlabel("Cumulative sector")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{k}" for k in x])
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim(0.066,0.069)
    ax.legend()

    if title:
        ax.set_title(title)

    return fig, ax
def plot_selected_stations_map(
    stations_df,
    station_names,
    *,
    d_zoom=10.0,
    with_topography=True,
    # topo inputs (optional)
    lats_terrain=None,
    lons_terrain=None,
    z_orog_m=None,
    terrain_alpha=0.6,
    proj=None,
    ax=None,
    title=None,
):
    """
    Plot ONLY selected stations as red dots, optionally over topography.

    Parameters
    ----------
    stations_df : DataFrame
        Must contain columns: Station_Name, Latitude, Longitude, Altitude
    station_names : list[str]
        Names of stations to plot (others are excluded)
    d_zoom : float
        Half-width of map in degrees
    with_topography : bool
        Whether to draw terrain background
    lats_terrain, lons_terrain, z_orog_m :
        Terrain grid (1D lat, 1D lon, 2D elevation in meters)
    """

    import numpy as np
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if proj is None:
        proj = ccrs.PlateCarree()

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": proj})
    else:
        fig = ax.figure

    # --------------------------------------------------
    # Filter stations (EXCLUDE ALL OTHERS)
    # --------------------------------------------------
    df = stations_df[stations_df["Station_Name"].isin(station_names)].copy()
    if df.empty:
        raise ValueError("No matching stations found.")

    lats = df["Latitude"].to_numpy(float)
    lons = df["Longitude"].to_numpy(float)

    # Map extent centered on mean location
    lon_c = np.mean(lons)
    lat_c = np.mean(lats)
    ax.set_extent(
        [68, 140,
         10 , 60],
        crs=proj,
    )
        # --------------------------------------------------
    # Gridlines + degree labels
    # --------------------------------------------------
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=0.6,
        linestyle="--",
        alpha=0.5,
        zorder=6,
    )

    gl.top_labels = False
    gl.right_labels = False

    # Degree formatting (°)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    step = 10  # degrees

    lon_min_i = int(np.floor((lon_c - d_zoom) / step) * step)
    lon_max_i = int(np.ceil((lon_c + d_zoom) / step) * step)
    lat_min_i = int(np.floor((lat_c - d_zoom) / step) * step)
    lat_max_i = int(np.ceil((lat_c + d_zoom) / step) * step)

    lon_ticks = np.arange(lon_min_i, lon_max_i + step, step)
    lat_ticks = np.arange(lat_min_i, lat_max_i + step, step)

    gl.xlocator = ticker.FixedLocator(lon_ticks)
    gl.ylocator = ticker.FixedLocator(lat_ticks)

    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}


    # --------------------------------------------------
    # Optional TOPOGRAPHY
    # --------------------------------------------------
    terrain_im = None
    if with_topography:
        if lats_terrain is None or lons_terrain is None or z_orog_m is None:
            raise ValueError("Topography requested but terrain arrays not provided.")

        LON_T, LAT_T = np.meshgrid(lons_terrain, lats_terrain)
        z = np.asarray(z_orog_m, float)

        # Mask sea (<= 0 m)
        z = np.ma.masked_where(z <= 0.0, z)

        terrain_im = ax.pcolormesh(
            LON_T, LAT_T, z,
            cmap="terrain",
            shading="auto",
            alpha=terrain_alpha,
            transform=ccrs.PlateCarree(),
            zorder=0,
        )

    # --------------------------------------------------
    # Map features
    # --------------------------------------------------
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue", zorder=-1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)

    # --------------------------------------------------
    # Plot selected stations ONLY
    # --------------------------------------------------
    ax.scatter(
        lons, lats,
        s=35,
        c="red",
        edgecolor="black",
        transform=ccrs.PlateCarree(),
        zorder=5,
        label="Selected stations",
    )
    '''
    # Optional labels
    for _, r in df.iterrows():
        ax.text(
            float(r["Longitude"]), float(r["Latitude"]),str(r["Station_Name"]),
            fontsize=8,
            transform=ccrs.PlateCarree(),
            ha="left", va="bottom",
            zorder=6
)   '''
    
    ax.legend(loc="upper right")

    # --------------------------------------------------
    # Colorbar for terrain ONLY
    # --------------------------------------------------
    # --------------------------------------------------
# Title (make it visible even with gridliner labels)
# --------------------------------------------------
    if title is None:
     title = "Selected stations" + (" over topography" if with_topography else "")

# Use a larger pad + explicit y to avoid gridliner label overlap
    ax.set_title(title, fontsize=12, pad=22, y=1.02)

# --------------------------------------------------
# Terrain colorbar on the LEFT (Cartopy-safe: manual axes)
# --------------------------------------------------
    if terrain_im is not None:
    # Reserve space for the left colorbar and top title
     fig.subplots_adjust(left=0.12, right=0.92, top=0.90, bottom=0.08)

    # Create a colorbar axes manually (no projection issues)
     bbox = ax.get_position()
     cax = fig.add_axes([bbox.x0 - 0.2, bbox.y0, 0.02, bbox.height])  # [left, bottom, width, height]
     cb = fig.colorbar(terrain_im, cax=cax)
     cb.set_label("Elevation (m)")
    else:
     fig.subplots_adjust(top=0.90)


    return fig, ax
def plot_cv_vs_distance(df_unw, df_w=None, ax=None, title=None):
    """
    Line plot of CV vs distance.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax.plot(df_unw["dmax_km"], df_unw["cv"], marker="o", label="Unweighted")
    
    if df_w is not None:
        ax.plot(df_w["dmax_km"], df_w["cv_w"], marker="s", label="Area-weighted")

    ax.set_xlabel("Distance from station (km)")
    ax.set_ylabel("Coefficient of Variation")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()

    if title:
        ax.set_title(title)

    return fig, ax
def plot_cv_map(
    lats_small,
    lons_small,
    data_arr,
    lat_s,
    lon_s,
    window_km,
    proj=None,
    ax=None,
):
    """
    Map showing local CV computed in a moving window.
    """
    if proj is None:
        proj = ccrs.PlateCarree()

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": proj})
    else:
        fig = ax.figure

    # compute distance field
    LON2D, LAT2D = np.meshgrid(lons_small, lats_small)
    dist = haversine_km(lat_s, lon_s, LAT2D, LON2D)

    mask = dist <= window_km
    vals = np.where(mask, data_arr, np.nan)

    # local CV (global within window)
    mean = np.nanmean(vals)
    std = np.nanstd(vals)
    cv = std / mean if mean != 0 else np.nan

    im = ax.pcolormesh(
        LON2D, LAT2D, vals,
        cmap="viridis",
        shading="auto",
        transform=ccrs.PlateCarree(),
    )

    ax.plot(lon_s, lat_s, "rx", markersize=10, transform=ccrs.PlateCarree())
    ax.set_title(f"Species field within {window_km} km (CV={cv:.3f})")

    plt.colorbar(im, ax=ax, label="Species")

    return fig, ax, cv

