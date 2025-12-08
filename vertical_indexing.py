'''
This file contains the functions that can be used
in order to retrieve the vertical indexing of the stations
'''

import numpy as np
from metpy.units import units
import metpy.calc as mpcalc
Rd = 287.05
g  = 9.80665


import numpy as np
import metpy.calc as mpcalc
from metpy.units import units

def metpy_compute_heights(p_prof_Pa, T_prof_K, qv=None, RH=None, z0=0.0):
    """
    Compute geometric heights from pressure + temperature using the hypsometric equation
    (MetPy version). Works with model-level midpoint values.
    """

    p = (p_prof_Pa * units.pascal)
    T = (T_prof_K * units.kelvin)

    # Virtual temperature
    if qv is not None:
        Tv = mpcalc.virtual_temperature(T, qv * units('kg/kg'))
    elif RH is not None:
        RH = RH / 100.0
        qv = mpcalc.mixing_ratio_from_relative_humidity(RH, T, p)
        Tv = mpcalc.virtual_temperature(T, qv)
    else:
        Tv = T

    # Convert to hPa for stability in the ln(p)
    p_hPa = p.to('hPa')

    # Output height array
    z = np.zeros(len(p)) * units.meter
    z[0] = z0 * units.meter

    # Hypsometric integration from level 1..N
    for k in range(1, len(p)):
        dz = (Rd / g) * Tv[k] * \
             np.log(p_hPa[k-1] / p_hPa[k])
        z[k] = z[k-1] + dz

    return z


def metpy_find_level_index(p_prof_Pa, T_prof_K, station_alt_m,
                           qv=None, RH=None):
    """Return nearest model level to station altitude using MetPy heights."""
    z_prof = metpy_compute_heights(
        p_prof_Pa=p_prof_Pa,
        T_prof_K=T_prof_K,
        qv=qv,
        RH=RH,
        z0=0.0,
    ).m  # convert to plain meters

    vertical_idx = int(np.argmin(np.abs(z_prof - station_alt_m)))
    p_hPa = p_prof_Pa[vertical_idx] / 100.0
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

    