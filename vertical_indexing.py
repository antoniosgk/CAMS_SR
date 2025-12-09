'''
This file contains the functions that can be used
in order to retrieve the vertical indexing of the stations
'''

import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
from metpy.constants import Rd, g  
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

    # Allocate height array with meter units
    z = np.zeros_like(Tv.magnitude) * units.meter
    z[-1] = z0 * units.meter

    # Hypsometric integration for top-to-bottom pressure indexing
    for k in range(len(p) - 2, -1, -1):
        Tv_layer = 0.5 * (Tv[k + 1] + Tv[k])
        dz = (Rd * Tv_layer / g) * np.log(p[k + 1] / p[k])
        z[k] = z[k + 1] + dz

    return z.magnitude


def metpy_find_level_index(p_prof_Pa, T_prof_K, station_alt_m,
                           qv=None, RH=None):
    """Return nearest model level to station altitude using MetPy heights."""

    z_prof = metpy_compute_heights(
        p_prof_Pa=p_prof_Pa,
        T_prof_K=T_prof_K,
        qv=qv,
        RH=RH,
        z0=0.0,
    )  # plain meters (ndarray)

    # station_alt_m should be in meters as a plain float
    vertical_idx = int(np.argmin(np.abs(z_prof - station_alt_m)))

    p_hPa = p_prof_Pa[vertical_idx] / 100.0  # Pa -> hPa

    return vertical_idx, p_hPa, z_prof[vertical_idx]
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

    