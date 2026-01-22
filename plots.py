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
    # species grid (always 1D)
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
    plot_species=True
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
    if (z_orog_m is not None) and (lats_terrain is not None) and (lons_terrain is not None):
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
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=5)
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
    radii=3
     ):
    import numpy as np
    import cartopy.crs as ccrs
    from matplotlib.patches import Rectangle

    lats_small = np.asarray(lats_small, dtype=float)
    lons_small = np.asarray(lons_small, dtype=float)

    # safety (optional but helpful)
    if not (0 <= ii < len(lats_small) and 0 <= jj < len(lons_small)):
        raise IndexError(f"(ii,jj)=({ii},{jj}) out of bounds for lats/lons small.")

    dlon = float(np.abs(lons_small[1] - lons_small[0]))
    dlat = float(np.abs(lats_small[1] - lats_small[0]))

    cx = float(lons_small[jj])
    cy = float(lats_small[ii])

    sizes = {"S1": 1, "S2": 2, "S3": 3}
    rect_styles = {
        "S1": {"edgecolor": "black", "linewidth": 2},
        "S2": {"edgecolor": "red",   "linewidth": 2},
        "S3": {"edgecolor": "blue",  "linewidth": 2},
    }

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
            zorder=10,                 # <-- key: always on top
            **rect_styles[name],
        )
        ax.add_patch(rect)

    if meta is not None:
        ax.set_title(build_meta_title(meta, kind="Map with Sectors"), pad=18)
        ax.figure.subplots_adjust(top=0.82)

    return ax, im


'''


def plot_variable_on_map(
    lats_small,           # 1D (Ny,)
    lons_small,           # 1D (Nx,)
    data_arr,             # 2D (Ny, Nx)
    lon_s, lat_s,
    units="",
    species_name="var",
    time_str=None,
    meta=None,
    # Option A inputs
    z_orog_m=None,        # 2D (Ny, Nx) orography height (m) for same domain
    add_orog_contours=True,
    # Choose backend
    backend="cartopy",    # "cartopy" (A) or "folium" (C)
    # Window around station
    d=0.4,
    # Visual tuning
    field_alpha=0.80,
    hillshade_alpha=0.35,
    # Folium tuning
    folium_tiles="Stamen Terrain",
    folium_zoom=9,
):
    """
    Returns:
      - cartopy: (fig, ax, im)
      - folium : (m, None, None)
    """
    lats_small = np.asarray(lats_small, dtype=float)
    lons_small = np.asarray(lons_small, dtype=float)
    arr = np.asarray(data_arr, dtype=float)

    # grid
    LON2D, LAT2D = np.meshgrid(lons_small, lats_small)

    # extent
    lon_min, lon_max = lon_s - d, lon_s + d
    lat_min, lat_max = lat_s - d, lat_s + d

    # -------------------------
    # OPTION C (Folium)
    # -------------------------
    if backend.lower() == "folium":
        # If you don't want Folium, comment out this whole block.
        import folium
        import io, base64
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        import matplotlib.cm as cm

        vmin = np.nanmin(arr)
        vmax = np.nanmax(arr)
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap("turbo")

        rgba = cmap(norm(arr))
        rgba[..., -1] = np.where(np.isfinite(arr), field_alpha, 0.0)

        fig_tmp, ax_tmp = plt.subplots(figsize=(6, 6), dpi=220)
        ax_tmp.axis("off")
        ax_tmp.imshow(rgba, origin="lower")
        buf = io.BytesIO()
        fig_tmp.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, transparent=True)
        plt.close(fig_tmp)
        png_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # bounds from the grid (more correct than lon_s±d)
        bounds = [[float(lats_small.min()), float(lons_small.min())],
                  [float(lats_small.max()), float(lons_small.max())]]

        m = folium.Map(location=[lat_s, lon_s], zoom_start=folium_zoom, tiles=None)
        folium.TileLayer(folium_tiles, name="Terrain").add_to(m)

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{png_b64}",
            bounds=bounds,
            opacity=1.0,
            interactive=True,
            cross_origin=False,
            zindex=10,
        ).add_to(m)

        popup = None
        if meta is not None:
            popup = (
                f"{meta.get('species', species_name)} ({meta.get('units', units)})<br>"
                f"time: {meta.get('time_str', time_str)}<br>"
                f"station: {meta.get('station_name','')} "
                f"({meta.get('station_lat',lat_s):.4f}, {meta.get('station_lon',lon_s):.4f}), "
                f"alt={meta.get('station_alt',np.nan):.1f} m<br>"
                f"model: ({meta.get('model_lat',lat_s):.4f}, {meta.get('model_lon',lon_s):.4f}), "
                f"lev={meta.get('model_level','?')}, p={meta.get('model_p_hPa',np.nan):.2f} hPa"
            )
        folium.Marker([lat_s, lon_s], popup=popup).add_to(m)
        folium.LayerControl().add_to(m)

        return m, None, None

    # -------------------------
    # OPTION A (Cartopy + PHIS hillshade)
    # -------------------------
    # If you don't want Cartopy, comment out this whole block.
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Terrain underlay from your PHIS-derived orography
    if z_orog_m is not None:
        z_orog_m = np.asarray(z_orog_m, dtype=float)
        gy, gx = np.gradient(z_orog_m)
        shade = 1.0 / np.sqrt(1.0 + gx**2 + gy**2)

        ax.pcolormesh(
            LON2D, LAT2D, shade,
            cmap="Greys",
            shading="auto",
            alpha=hillshade_alpha,
            transform=ccrs.PlateCarree(),
            zorder=0
        )

        if add_orog_contours:
            levels = np.arange(0, 7000, 500)
            cs = ax.contour(
                LON2D, LAT2D, z_orog_m,
                levels=levels,
                linewidths=0.5,
                alpha=0.5,
                colors="k",
                transform=ccrs.PlateCarree(),
                zorder=1
            )
            ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f m")

    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)

    norm = Normalize(vmin=np.nanmin(arr), vmax=np.nanmax(arr))
    im = ax.pcolormesh(
        LON2D, LAT2D, arr,
        cmap="turbo",
        shading="auto",
        norm=norm,
        transform=ccrs.PlateCarree(),
        alpha=field_alpha,
        zorder=3
    )

    ax.plot(lon_s, lat_s, "kx", markersize=10, transform=ccrs.PlateCarree(), zorder=4)

    if meta is not None:
        title = (
            f"{meta.get('species', species_name)} ({meta.get('units', units)}) at {meta.get('time_str', time_str)}\n"
            f"Station {meta.get('station_name','')}: ({meta.get('station_lat',lat_s):.4f}, {meta.get('station_lon',lon_s):.4f}), "
            f"alt={meta.get('station_alt',np.nan):.1f} m | "
            f"Model: ({meta.get('model_lat',lat_s):.4f}, {meta.get('model_lon',lon_s):.4f}), "
            f"lev={meta.get('model_level','?')}, p={meta.get('model_p_hPa',np.nan):.2f} hPa"
        )
    else:
        title = f"{species_name} ({units})" + (f" at {time_str}" if time_str else "")

    ax.set_title(title, pad=14)
    fig.subplots_adjust(top=0.82)

    return fig, ax, im


def plot_rectangles(
    ax_or_map,
    lats_small,    # 1D
    lons_small,    # 1D
    ii, jj,        # center indices in the small domain
    im=None,       # only used in cartopy
    units="",
    species_name="var",
    time_str=None,
    meta=None,
    backend="cartopy",
):
    """
    Returns:
      - cartopy: (ax, im)
      - folium : (m, None)
    """
    lats_small = np.asarray(lats_small, dtype=float)
    lons_small = np.asarray(lons_small, dtype=float)

    # center
    cx = float(lons_small[jj])
    cy = float(lats_small[ii])

    # grid spacing
    dlon = float(abs(lons_small[1] - lons_small[0]))
    dlat = float(abs(lats_small[1] - lats_small[0]))

    sizes = {"S1": 1, "S2": 2, "S3": 3}

    # -------------------------
    # OPTION C (Folium rectangles)
    # -------------------------
    if backend.lower() == "folium":
        # If you don't want Folium rectangles, comment out this block.
        import folium
        m = ax_or_map

        styles = {
            "S1": dict(color="black", weight=2, fill=False),
            "S2": dict(color="red", weight=2, fill=False),
            "S3": dict(color="blue", weight=2, fill=False),
        }

        for name, r in sizes.items():
            width = (2 * r + 1) * dlon
            height = (2 * r + 1) * dlat
            left = cx - width / 2
            right = cx + width / 2
            bottom = cy - height / 2
            top = cy + height / 2

            folium.Rectangle(
                bounds=[[bottom, left], [top, right]],
                tooltip=name,
                **styles[name]
            ).add_to(m)

        return m, None

    # -------------------------
    # OPTION A (Cartopy rectangles)
    # -------------------------
    # If you don't want Cartopy rectangles, comment out this block.
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.patches import Rectangle

    ax = ax_or_map

    rect_styles = {
        "S1": {"edgecolor": "black", "linewidth": 2},
        "S2": {"edgecolor": "red",   "linewidth": 2},
        "S3": {"edgecolor": "blue",  "linewidth": 2},
    }

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

    if im is not None:
        cb = plt.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(meta.get("units", units) if meta else units)

    if meta is not None:
        title = (
            f"{meta.get('species', species_name)} ({meta.get('units', units)}) at {meta.get('time_str', time_str)}\n"
            f"Station {meta.get('station_name','')}: ({meta.get('station_lat',np.nan):.4f}, {meta.get('station_lon',np.nan):.4f}), "
            f"alt={meta.get('station_alt',np.nan):.1f} m | "
            f"Model: ({meta.get('model_lat',np.nan):.4f}, {meta.get('model_lon',np.nan):.4f}), "
            f"lev={meta.get('model_level','?')}, p={meta.get('model_p_hPa',np.nan):.2f} hPa"
        )
        ax.set_title(title, pad=14)
        ax.figure.subplots_adjust(top=0.82)
    else:
        ax.set_title(f"{species_name} with sectors")

    return ax, im
    '''
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
