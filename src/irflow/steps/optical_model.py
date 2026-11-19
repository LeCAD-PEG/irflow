"""Simulation step 3: optical model, giving synthetic IR camera images of the surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from irflow.backends.base import OpticalModelBackend
from irflow.config import check_config
from irflow.data import CameraImages, HeatFluxField, TemperatureField
from irflow.io import load_temperature


class OpticalModel:
    """Optical model step. The backend code is chosen with ``backend_code``.

    It computes one camera image per time of the surface temperature (one image for a steady
    thermal model). By default the temperature comes from the previous (thermal model) step.
    To run the optical model on its own, give ``input_data``: a :class:`TemperatureField` or
    the path of a temperature file saved by irflow (e.g. ``ThermalModel/temperature.pvd``).

    The config keys are the keyword arguments of the backend class, e.g.
    :class:`irflow.backends.optical.raysect.BackendRaysect`.
    """

    dir_name = "OpticalModel"

    def __init__(
        self,
        name: str = "optical_case",
        enabled: bool = True,
        backend_code: str = "raysect",
        config: dict[str, Any] | None = None,
        resources: Any = None,
        input_data: TemperatureField | str | Path | None = None,
    ) -> None:
        self.name = name
        self.enabled = enabled
        self.backend_code = backend_code
        self.config = config or {}
        self.resources = resources
        self.result: CameraImages | None = None

        if isinstance(input_data, (str, Path)):
            input_data = load_temperature(input_data)
        self.temperature: TemperatureField | None = input_data
        self.has_own_input = input_data is not None

        self.backend = self.get_backend(backend_code)

    def get_backend(self, backend_code: str) -> OpticalModelBackend:
        """Create the backend for ``backend_code`` from the step config."""
        # Backends are imported here so that only the chosen code has to be installed.
        if backend_code.lower() == "raysect":
            from irflow.backends.optical.raysect import BackendRaysect

            check_config(
                self.config,
                required=["camera_position", "look_at", "fov", "pixels", "emissivity"],
                optional=[
                    "up",
                    "wavelength_range",
                    "pixel_samples",
                    "spectral_bins",
                    "processes",
                ],
                where=f"OpticalModel config (backend '{backend_code}')",
            )
            return BackendRaysect(**self.config)

        raise ValueError(f"Unknown optical model backend '{backend_code}'. Supported: 'raysect'.")

    def set_irflow_pass_data(self, data: HeatFluxField | TemperatureField | CameraImages) -> None:
        """Receive the surface temperature from the previous step."""
        if not isinstance(data, TemperatureField):
            raise TypeError(f"OpticalModel needs a TemperatureField, got {type(data).__name__}")
        self.temperature = data

    def run(self, step_dir: str | Path) -> None:
        """Run the backend in ``step_dir`` (must exist) and store the camera images."""
        step_dir = Path(step_dir)
        if not step_dir.is_dir():
            raise FileNotFoundError(f"Directory does not exist: {step_dir}")
        if self.temperature is None:
            raise RuntimeError(
                "OpticalModel has no surface temperature: run a ThermalModel step before it "
                "or give input_data."
            )

        self.backend.run(step_dir, self.temperature)
        self.result = self.backend.get_result()

    def get_irflow_pass_data(self) -> CameraImages:
        """Synthetic camera images, one per time."""
        if self.result is None:
            raise RuntimeError("OpticalModel has no result yet: call run() first.")
        return self.result
