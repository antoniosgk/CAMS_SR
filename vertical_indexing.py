'''
This file contains the functions that can be used
in order to retrieve the vertical indexing of the stations
'''

import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from metpy.constants import Rd, g  
'''
def metpy_compute_heights(p_prof_Pa, T_prof_K, qv=None, RH=None, z0=0.0):
    """
    Compute geometric heights from pressure + temperature using the hypsometric equation
    (MetPy version). Works with model-level midpoint values.

    Parameters
    ----------
    p_prof_Pa : array-like
        Pressure profile in Pa (no units attached yet).
    T_prof_K : array-like
        Temperature profile in K (no units attached yet).
    qv : array-like, optional
        Specific humidity [kg/kg]. If provided, used to compute virtual temperature.
    RH : array-like, optional
        Relative humidity (dimensionless, 0–1 or %, depending on how you handle it).
    z0 : float, optional
        Starting height [m].

    Returns
    -------
    z : ndarray
        Height profile in meters (plain numpy array, no units).
    """

    # Attach units
    p = np.asarray(p_prof_Pa) * units.pascal
    T = np.asarray(T_prof_K) * units.kelvin

    # Compute virtual temperature
    if qv is not None:
        # assume qv is kg/kg without units
        qv_q = np.asarray(qv) * units('kg/kg')
        Tv = mpcalc.virtual_temperature(T, qv_q)

    elif RH is not None:
        RH_arr = np.asarray(RH)

        # If your RH is in %, convert to fraction here:
        # RH_q = (RH_arr / 100.0) * units.dimensionless
        # If it's already 0–1, use:
        RH_q = RH_arr * units.dimensionless

        mixing_ratio = mpcalc.mixing_ratio_from_relative_humidity(p, T, RH_q)
        Tv = mpcalc.virtual_temperature(T, mixing_ratio)

    else:
        Tv = T

    nlev = len(p)
    z = np.empty(nlev) * units.meter

    # Find index of maximum pressure (assume this is "surface")
    k_surf = int(np.argmax(p.magnitude))
    z[k_surf] = z0 * units.meter  # anchor at model surface

    # Integrate upward (towards lower pressure)
    for k in range(k_surf - 1, -1, -1):
        Tv_layer = 0.5 * (Tv[k + 1] + Tv[k])
        dz = (Rd * Tv_layer / g) * np.log(p[k + 1] / p[k])
        z[k] = z[k + 1] + dz

    # Integrate downward (if any levels have p > surface)
    for k in range(k_surf + 1, nlev):
        Tv_layer = 0.5 * (Tv[k - 1] + Tv[k])
        dz = (Rd * Tv_layer / g) * np.log(p[k - 1] / p[k])
        z[k] = z[k - 1] + dz

    return z.magnitude
'''
def metpy_compute_heights(p_prof_Pa, T_prof_K, qv=None, RH=None, z0=0.0):
    """
    Compute geometric heights from pressure + temperature using the hypsometric equation
    (MetPy version). Works with model-level midpoint values.

    Supports:
      - 1D profiles: p_prof_Pa.shape == (nlev,)
      - 2D profiles: p_prof_Pa.shape == (nlev, ncol), broadcasting over columns.

    Parameters
    ----------
    p_prof_Pa : array-like
        Pressure profile in Pa.
    T_prof_K : array-like
        Temperature profile in K (same shape as p_prof_Pa).
    qv : array-like, optional
        Specific humidity [kg/kg]. If provided, used to compute virtual temperature.
    RH : array-like, optional
        Relative humidity (dimensionless, 0–1 or %, depending on how you handle it).
    z0 : float or array-like, optional
        Surface height ASL [m]. For 1D p/T, z0 is scalar.
        For 2D p/T, z0 can be scalar (same for all columns) or 1D (ncol) matching columns.

    Returns
    -------
    z : ndarray
        Height(s) in meters (same shape as p_prof_Pa, i.e. (nlev,) or (nlev, ncol)).
    """
    p_arr = np.asarray(p_prof_Pa)
    T_arr = np.asarray(T_prof_K)

    if p_arr.shape != T_arr.shape:
        raise ValueError("p_prof_Pa and T_prof_K must have the same shape.")

    # Attach units
    p = p_arr * units.pascal
    T = T_arr * units.kelvin

    # Decide which humidity input is used
    if qv is not None and RH is not None:
        print("INFO: Both qv and RH provided; using qv for virtual temperature.")
    if qv is not None:
        print("INFO: Using qv (specific humidity) for virtual temperature.")
        qv_q = np.asarray(qv) * units("kg/kg")
        Tv = mpcalc.virtual_temperature(T, qv_q)
    elif RH is not None:
        print("INFO: Using RH (relative humidity) for virtual temperature.")
        RH_arr = np.asarray(RH)
        # If RH is in %, uncomment the division by 100:
        # RH_q = (RH_arr / 100.0) * units.dimensionless
        RH_q = RH_arr * units.dimensionless
        mixing_ratio = mpcalc.mixing_ratio_from_relative_humidity(p, T, RH_q)
        Tv = mpcalc.virtual_temperature(T, mixing_ratio)
    else:
        print("INFO: No humidity provided; using dry temperature T for virtual temperature.")
        Tv = T

    # Shape handling: 1D or 2D
    if p.ndim == 1:
        # ----- 1D CASE -----
        nlev = p.shape[0]
        z = np.empty(nlev) * units.meter

        # Surface index (max pressure)
        k_surf = int(np.argmax(p.magnitude))

        # Surface height ASL
        z_surf = z0 * units.meter
        z[k_surf] = z_surf

        # Integrate upward (towards lower pressure)
        for k in range(k_surf - 1, -1, -1):
            Tv_layer = 0.5 * (Tv[k + 1] + Tv[k])
            dz = (Rd * Tv_layer / g) * np.log(p[k + 1] / p[k])
            z[k] = z[k + 1] + dz

        # Integrate downward (if any levels have p > surface)
        for k in range(k_surf + 1, nlev):
            Tv_layer = 0.5 * (Tv[k - 1] + Tv[k])
            dz = (Rd * Tv_layer / g) * np.log(p[k - 1] / p[k])
            z[k] = z[k - 1] + dz

        return z.magnitude

    elif p.ndim == 2:
        # ----- 2D CASE: (nlev, ncol) -----
        nlev, ncol = p.shape
        z = np.empty_like(p.magnitude) * units.meter

        # Surface index for each column
        k_surf = np.argmax(p.magnitude, axis=0)  # shape (ncol,)

        # Handle z0: scalar or 1D per-column
        z0_arr = np.asarray(z0)
        if z0_arr.ndim == 0:
            z_surf = np.full(ncol, z0_arr) * units.meter
        elif z0_arr.shape == (ncol,):
            z_surf = z0_arr * units.meter
        else:
            raise ValueError("For 2D p/T, z0 must be scalar or shape (ncol,)")

        # Set surface heights
        for j in range(ncol):
            z[k_surf[j], j] = z_surf[j]

        # Integrate upward
        for k in range(nlev - 2, -1, -1):
            # where k < k_surf: we are above surface in that column
            mask = k < k_surf
            if not np.any(mask):
                continue
            Tv_layer = 0.5 * (Tv[k + 1, mask] + Tv[k, mask])
            dz = (Rd * Tv_layer / g) * np.log(p[k + 1, mask] / p[k, mask])
            z[k, mask] = z[k + 1, mask] + dz

        # Integrate downward
        for k in range(1, nlev):
            mask = k > k_surf
            if not np.any(mask):
                continue
            Tv_layer = 0.5 * (Tv[k - 1, mask] + Tv[k, mask])
            dz = (Rd * Tv_layer / g) * np.log(p[k - 1, mask] / p[k, mask])
            z[k, mask] = z[k - 1, mask] + dz

        return z.magnitude

    else:
        raise ValueError("metpy_compute_heights currently supports only 1D or 2D p/T profiles.")

'''
def metpy_find_level_index(p_prof_Pa, T_prof_K, station_alt_m,
                           qv=None, RH=None, z_surf_model=0.0):
    """
    Return nearest model level to station altitude using MetPy heights.

    z_surf_model : float
        Model surface height (m) at the grid cell, derived from PHIS.
    """

    # Compute height profile anchored at model surface
    z_prof = metpy_compute_heights(
        p_prof_Pa=p_prof_Pa,
        T_prof_K=T_prof_K,
        qv=qv,
        RH=RH,
        z0=z_surf_model,
    )  # plain meters (ndarray)

    # Basic vertical diagnostics
    p_hPa_prof = p_prof_Pa / 100.0
    print(f"DEBUG: p_prof range (hPa):, {float(p_hPa_prof.min()):.1f}, →, {float(p_hPa_prof.max()):.1f}")
    print("DEBUG: z_prof range (m):", float(z_prof.min()), "→", float(z_prof.max()))
    print("DEBUG: few levels near surface (by max pressure index):")

    k_surf = int(np.argmax(p_prof_Pa))
    for k in range(max(0, k_surf - 2), min(len(z_prof), k_surf + 3)):
        print(f"  k={k:2d}: p={p_hPa_prof[k]:7.2f} hPa, z={z_prof[k]:8.1f} m")

    # station_alt_m should be in meters above sea level
    vertical_idx = int(np.argmin(np.abs(z_prof - station_alt_m)))

    # Distance between station altitude and nearest model level
    diff = abs(z_prof[vertical_idx] - station_alt_m)
    if diff > 1500.0:
        print(f"WARNING: nearest model level is {diff:.0f} m away from station altitude")

    # Standard atmosphere check at station height (sanity only)
    p_std = mpcalc.height_to_pressure_std(station_alt_m * units.meter)
    print("DEBUG: Standard atmosphere pressure at station height:",
          p_std.to('hectopascal'))

    p_hPa = p_prof_Pa[vertical_idx] / 100.0  # Pa -> hPa

    return vertical_idx, p_hPa, z_prof[vertical_idx]
'''
def metpy_find_level_index(p_prof_Pa, T_prof_K, station_alt_m,
                           qv=None, RH=None, z_surf_model=0.0):
    """
    Return nearest model level(s) to station altitude using MetPy heights.

    Supports:
      - 1D p/T: single column → scalar outputs.
      - 2D p/T: shape (nlev, ncol) → 1D arrays of length ncol.

    Parameters
    ----------
    p_prof_Pa     : array-like, pressure profile(s) [Pa]
    T_prof_K      : array-like, temperature profile(s) [K]
    station_alt_m : float, station altitude ASL [m]
    qv            : optional specific humidity profile(s) [kg/kg]
    RH            : optional relative humidity profile(s) (0–1 or %)
    z_surf_model  : surface height ASL [m]; for 2D p/T can be scalar or (ncol,)

    Returns
    -------
    vertical_idx  : int (1D case) or ndarray of ints (2D case)
    p_hPa         : float (1D) or ndarray of floats (2D)
    z_level_m     : float (1D) or ndarray of floats (2D)
    """
    p_arr = np.asarray(p_prof_Pa)
    T_arr = np.asarray(T_prof_K)

    if p_arr.shape != T_arr.shape:
        raise ValueError("p_prof_Pa and T_prof_K must have the same shape.")

    # Height profile(s) ASL
    z_prof = metpy_compute_heights(
        p_prof_Pa=p_arr,
        T_prof_K=T_arr,
        qv=qv,
        RH=RH,
        z0=z_surf_model,
    )

    p_hPa_prof = p_arr / 100.0

    # ----- 1D CASE -----
    if p_arr.ndim == 1:
        print(f"DEBUG: p_prof range (hPa): {p_hPa_prof.min():.1f} → {p_hPa_prof.max():.1f}")
        print("DEBUG: z_prof range (m):", float(z_prof.min()), "→", float(z_prof.max()))
        print("DEBUG: few levels near surface (by max pressure index):")

        k_surf = int(np.argmax(p_arr))
        for k in range(max(0, k_surf - 2), min(len(z_prof), k_surf + 3)):
            print(f"  k={k:2d}: p={p_hPa_prof[k]:7.2f} hPa, z={z_prof[k]:8.1f} m")

        vertical_idx = int(np.argmin(np.abs(z_prof - station_alt_m)))

        diff = abs(z_prof[vertical_idx] - station_alt_m)
        if diff > 1500.0:
            print(f"WARNING: nearest model level is {diff:.0f} m away from station altitude")

        p_std = mpcalc.height_to_pressure_std(station_alt_m * units.meter)
        print("DEBUG: Standard atmosphere pressure at station height:",
              p_std.to("hectopascal"))

        p_hPa = p_arr[vertical_idx] / 100.0
        return vertical_idx, p_hPa, z_prof[vertical_idx]

    # ----- 2D CASE: (nlev, ncol) -----
    elif p_arr.ndim == 2:
        nlev, ncol = p_arr.shape

        # Broadcast station altitude to (nlev, ncol) for comparison
        z_target = station_alt_m  # scalar
        diff = np.abs(z_prof - z_target)

        # argmin along vertical for each column
        vertical_idx = np.argmin(diff, axis=0)  # shape (ncol,)

        # Gather p and z at those indices
        cols = np.arange(ncol)
        p_sel_hPa = p_hPa_prof[vertical_idx, cols]
        z_sel_m = z_prof[vertical_idx, cols]

        # Diagnostics: global ranges + a sample column (0)
        print(f"DEBUG (2D): p_prof range (hPa): {p_hPa_prof.min():.1f} → {p_hPa_prof.max():.1f}")
        print(f"DEBUG (2D): z_prof range (m): {z_prof.min():.1f} → {z_prof.max():.1f}")
        sample_col = 0
        k_surf_sample = int(np.argmax(p_arr[:, sample_col]))
        print(f"DEBUG (2D): sample column {sample_col}, few levels near surface:")
        for k in range(max(0, k_surf_sample - 2), min(nlev, k_surf_sample + 3)):
            print(f"  k={k:2d}: p={p_hPa_prof[k, sample_col]:7.2f} hPa, z={z_prof[k, sample_col]:8.1f} m")

        # Optional: warn if any column is far from station altitude
        diff_min = np.min(diff, axis=0)  # per-column min difference
        n_far = np.sum(diff_min > 1500.0)
        if n_far > 0:
            print(f"WARNING: {n_far} column(s) have nearest level > 1500 m away from station altitude.")

        p_std = mpcalc.height_to_pressure_std(station_alt_m * units.meter)
        print("DEBUG: Standard atmosphere pressure at station height:",
              p_std.to("hectopascal"))

        return vertical_idx, p_sel_hPa, z_sel_m

    else:
        raise ValueError("metpy_find_level_index currently supports only 1D or 2D p/T profiles.")

    
def altitude_to_pressure_ISA(z_m):
    """Convert altitude (m) to pressure (Pa) using standard barometric formula (ISA troposphere).
       Good approximation for typical surface altitudes."""
    # Constants
    p0 = 101325.0       # Pa
    T0 = 288.15         # K
    g = 9.80665         # m/s2
    L = 0.0065          # K/m
    R = 287.05          # J/(kg K)
    # Avoid negative base when very high z: clip to realistic domain
    term = 1.0 - L * z_m / T0
    if term <= 0:
        return 0.0
    exponent = g / (R * L)
    return p0 * (term ** exponent)

def pressure_to_height(p_hPa, T_K):
    """
    Compute geometric height using the hypsometric equation.

    Parameters
    ----------
    p_hPa : 1D array
        Pressure profile in hPa.
    T_K : 1D array
        Temperature profile in K.

    Returns
    -------
    z : 1D array
        Heights in meters of each model level.
    """
    p_hPa = np.asarray(p_hPa)
    T_K   = np.asarray(T_K)

    # Avoid divide-by-zero / negative pressures
    p_hPa = np.clip(p_hPa, 1e-6, None)

    # Hypsometric equation (relative to 1000 hPa)
    # z = (Rd * T / g) * ln(p0 / p)
    return (Rd * T_K / g) * np.log(1000.0 / p_hPa)


def geos_find_level_index(p_prof_Pa, T_prof_K, station_alt):
    """
    Find the model level closest to station altitude using PL and T profiles.

    Parameters
    ----------
    p_prof_Pa : 1D array
        Pressure profile in Pa at model level midpoints.
    T_prof_K : 1D array
        Temperature profile in K at the same levels.
    station_alt : float
        Station altitude in meters.

    Returns
    -------
    idx : int
        Index of the closest model level.
    p_level_hPa : float
        Pressure of that level (hPa).
    z_level_m : float
        Height of that level (m).
    """
    p_prof_Pa = np.asarray(p_prof_Pa).squeeze()
    T_prof_K  = np.asarray(T_prof_K).squeeze()

    if p_prof_Pa.ndim != 1 or T_prof_K.ndim != 1:
        raise ValueError(f"Expected 1D profiles, got shapes {p_prof_Pa.shape} and {T_prof_K.shape}")

    # Pa -> hPa
    p_prof_hPa = p_prof_Pa / 100.0

    # Height of each model level from hypsometric equation
    z_prof = pressure_to_height(p_prof_hPa, T_prof_K)

    # Level closest in height to station altitude
    idx = int(np.argmin(np.abs(z_prof - station_alt)))

    return idx, float(p_prof_hPa[idx]), float(z_prof[idx])

    