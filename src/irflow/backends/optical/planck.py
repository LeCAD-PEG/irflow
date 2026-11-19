"""Planck's law: spectral radiance of a black body, band radiance and apparent temperature.

Kept apart from the raysect backend so that other optical backends can use it without raysect.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, Planck, speed_of_light

# Radiation constants for wavelengths in nm and spectral radiance in W/(m^2 sr nm)
FIRST_RADIATION_CONSTANT = 2.0 * Planck * speed_of_light**2 * 1e36  # 2 h c^2 [W nm^4/(m^2 sr)]
SECOND_RADIATION_CONSTANT = Planck * speed_of_light / Boltzmann * 1e9  # h c / k [nm K]

# Temperature range [K] of the table that converts band radiance to apparent temperature
LOWEST_APPARENT_TEMPERATURE = 50.0
HIGHEST_APPARENT_TEMPERATURE = 1.0e5
TABLE_POINTS = 2000  # gives a relative error of the conversion below 1e-5


def planck_radiance(wavelength: np.ndarray, temperature: float | np.ndarray) -> np.ndarray:
    """Black body spectral radiance [W/(m^2 sr nm)] at ``wavelength`` [nm], ``temperature`` [K].

    ``wavelength`` and ``temperature`` may be arrays that broadcast against each other.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    # For very cold surfaces the exponential overflows to inf, which correctly gives radiance 0.
    with np.errstate(over="ignore"):
        exponent = np.expm1(SECOND_RADIATION_CONSTANT / (wavelength * temperature))
        return np.asarray(FIRST_RADIATION_CONSTANT / wavelength**5 / exponent)


def band_radiance(
    temperature: float | np.ndarray, wavelengths: np.ndarray, delta_wavelength: float
) -> np.ndarray:
    """Black body radiance [W/(m^2 sr)] summed over wavelength bins, one value per temperature.

    ``wavelengths`` [nm] are the bin centres and ``delta_wavelength`` [nm] the bin width. This is
    the midpoint rule, which is how raysect integrates a spectrum over the camera band.
    """
    temperatures = np.atleast_1d(np.asarray(temperature, dtype=float))
    spectral = planck_radiance(np.asarray(wavelengths)[None, :], temperatures[:, None])
    return np.asarray(spectral.sum(axis=1) * delta_wavelength)


def apparent_temperature(
    radiance: np.ndarray, wavelengths: np.ndarray, delta_wavelength: float, emissivity: float
) -> np.ndarray:
    """Temperature [K] at which a grey body of ``emissivity`` has band radiance ``radiance``.

    ``radiance`` [W/(m^2 sr)] must have been computed with the wavelength bins ``wavelengths``
    (centres [nm]) of width ``delta_wavelength`` [nm], see :func:`band_radiance`. With the same
    bins, a grey body of this emissivity at temperature T gives back T.

    The result is NaN where the radiance is below that of a body at
    :data:`LOWEST_APPARENT_TEMPERATURE` (e.g. zero, for pixels that see nothing) or above that of
    a body at :data:`HIGHEST_APPARENT_TEMPERATURE`.
    """
    table_temperature = np.geomspace(
        LOWEST_APPARENT_TEMPERATURE, HIGHEST_APPARENT_TEMPERATURE, TABLE_POINTS
    )
    table_radiance = emissivity * band_radiance(table_temperature, wavelengths, delta_wavelength)
    usable = table_radiance > 0.0  # at short wavelengths, cold bodies underflow to 0

    # log(radiance) is nearly linear in 1/T (Wien's approximation), so linear interpolation of
    # 1/T over log(radiance) is accurate with a moderate table.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_radiance = np.log(np.asarray(radiance, dtype=float))
    inverse_temperature = np.interp(
        log_radiance,
        np.log(table_radiance[usable]),
        1.0 / table_temperature[usable],
        left=np.nan,
        right=np.nan,
    )
    return np.asarray(1.0 / inverse_temperature)
