"""Planck's law: spectral radiance of a black body.

Kept apart from the raysect backend so that other optical backends can use it without raysect.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import Boltzmann, Planck, speed_of_light

# Radiation constants for wavelengths in nm and spectral radiance in W/(m^2 sr nm)
FIRST_RADIATION_CONSTANT = 2.0 * Planck * speed_of_light**2 * 1e36  # 2 h c^2 [W nm^4/(m^2 sr)]
SECOND_RADIATION_CONSTANT = Planck * speed_of_light / Boltzmann * 1e9  # h c / k [nm K]


def planck_radiance(wavelength: np.ndarray, temperature: float) -> np.ndarray:
    """Black body spectral radiance [W/(m^2 sr nm)] at ``wavelength`` [nm], ``temperature`` [K]."""
    wavelength = np.asarray(wavelength, dtype=float)
    # For very cold surfaces the exponential overflows to inf, which correctly gives radiance 0.
    with np.errstate(over="ignore"):
        exponent = np.expm1(SECOND_RADIATION_CONSTANT / (wavelength * temperature))
        return FIRST_RADIATION_CONSTANT / wavelength**5 / exponent
