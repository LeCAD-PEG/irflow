"""Interfaces that every backend of a simulation step implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from irflow.data import HeatFluxField, TemperatureField


class FieldLineTracingBackend(ABC):
    """A field line tracing code: computes heat flux on the target mesh."""

    @abstractmethod
    def run(self, step_dir: Path) -> None:
        """Run the code; all files go into ``step_dir``."""

    @abstractmethod
    def get_result(self) -> HeatFluxField:
        """Heat flux on the target mesh after :meth:`run`."""


class ThermalModelBackend(ABC):
    """A thermal code: computes the plasma-facing surface temperature from a heat flux."""

    @abstractmethod
    def run(
        self, step_dir: Path, heat_flux: HeatFluxField, mapping_tolerance: float | None = None
    ) -> None:
        """Run the code with ``heat_flux`` as plasma boundary condition.

        Before solving, check with :func:`irflow.mapping.check_overlap` that the plasma surface
        lies within ``mapping_tolerance`` [m] of the heat flux points (None: automatic).
        """

    @abstractmethod
    def get_result(self) -> TemperatureField:
        """Plasma-facing surface temperature after :meth:`run`."""
