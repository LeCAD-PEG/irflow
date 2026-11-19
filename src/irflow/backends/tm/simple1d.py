"""Thermal model backend 'simple1d': every heat flux triangle is an independent 1D slab.

Each triangle of the heat flux mesh is the front surface (x = 0) of a slab of thickness L,
normal to the surface. The triangle's heat flux enters at the front from t = 0 on and stays
constant; the back (x = L) has a fixed temperature, a heat transfer coefficient or is
adiabatic. Material properties are constant. Heat conduction between neighbouring triangles is
neglected, so this is a fast estimate, not a replacement for a 3D model (OpenFOAM, ElmerFEM).

- Steady state is solved analytically (linear profile through the slab).
- Transient runs use vertex-centred finite volumes through the thickness and backward Euler in
  time. All slabs share one matrix, which is factorized once; each time step then solves all
  slabs together, one right-hand side per slab.

The surface temperature per triangle is turned into node values by area-weighted averaging
over the triangles that share each node.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import brentq

from irflow.backends.base import ThermalModelBackend
from irflow.data import HeatFluxField, SurfaceMesh, TemperatureField

DEFAULT_N_POINTS = 51
DEFAULT_INITIAL_TEMPERATURE = 300.0  # [K]


@dataclass(frozen=True)
class Material:
    """Constant material properties of the slab."""

    density: float  # [kg/m^3]
    heat_capacity: float  # [J/(kg K)]
    heat_conductivity: float  # [W/(m K)]

    @property
    def diffusivity(self) -> float:
        """Thermal diffusivity k / (rho c) [m^2/s]."""
        return self.heat_conductivity / (self.density * self.heat_capacity)


@dataclass(frozen=True)
class BackBoundary:
    """Boundary condition at the back of the slab (x = L)."""

    kind: str  # "temperature", "htc" or "adiabatic"
    temperature: float = 0.0  # back temperature ("temperature") or T_ext ("htc") [K]
    htc: float = 0.0  # heat transfer coefficient ("htc") [W/(m^2 K)]


def require_key(config: Mapping[str, Any], key: str, where: str) -> Any:
    """Return ``config[key]``; raise ValueError naming ``where`` if the key is missing."""
    if not isinstance(config, Mapping):
        raise ValueError(f"simple1d {where} must be a dict, got {config!r}")
    if key not in config:
        raise ValueError(f"simple1d {where} is missing the key '{key}'")
    return config[key]


def positive_float(value: Any, name: str) -> float:
    """Return ``value`` as a float; raise ValueError if it is not a finite positive number."""
    is_number = isinstance(value, Real) and not isinstance(value, bool)
    if not is_number or not 0 < float(value) < np.inf:
        raise ValueError(f"simple1d: '{name}' must be a positive number, got {value!r}")
    return float(value)


def read_material(material_properties: Mapping[str, Any]) -> Material:
    """Read constant density, heat capacity and heat conductivity.

    Same shape as one region of the OpenFOAM backend's ``material_properties``:
    ``{"density": {"type": "constant", "value": 3060}, "heat_capacity": {...},
    "heat_conductivity": {...}}``.
    """
    values = {}
    for name in ("density", "heat_capacity", "heat_conductivity"):
        prop = require_key(material_properties, name, "material_properties")
        kind = require_key(prop, "type", f"material_properties['{name}']")
        if kind != "constant":
            raise NotImplementedError(
                f"simple1d supports only constant material properties; '{name}' has type {kind!r}."
            )
        value = require_key(prop, "value", f"material_properties['{name}']")
        values[name] = positive_float(value, f"material_properties['{name}']['value']")
    return Material(**values)


def read_back_boundary(back_boundary: Mapping[str, Any]) -> BackBoundary:
    """Read ``{"type": "temperature", "value": T}``, ``{"type": "htc", "value": h,
    "T_ext": T}`` or ``{"type": "adiabatic"}``."""
    kind = require_key(back_boundary, "type", "back_boundary")
    if kind == "temperature":
        value = require_key(back_boundary, "value", "back_boundary")
        return BackBoundary(kind, temperature=positive_float(value, "back_boundary['value']"))
    if kind == "htc":
        htc = require_key(back_boundary, "value", "back_boundary")
        t_ext = require_key(back_boundary, "T_ext", "back_boundary")
        return BackBoundary(
            kind,
            temperature=positive_float(t_ext, "back_boundary['T_ext']"),
            htc=positive_float(htc, "back_boundary['value']"),
        )
    if kind == "adiabatic":
        return BackBoundary(kind)
    raise ValueError(
        f"simple1d back_boundary type must be 'temperature', 'htc' or 'adiabatic', got {kind!r}"
    )


def slab_grid(thickness: float, n_points: int, first_cell_size: float | None = None) -> np.ndarray:
    """Node positions from the front surface (x = 0) to the back (x = thickness).

    Without ``first_cell_size`` the grid is uniform. Otherwise the cell at the front has that
    size and the cells grow geometrically towards the back, which resolves the thin heated
    layer of short transients with few points.
    """
    n_cells = n_points - 1
    uniform_size = thickness / n_cells
    if first_cell_size is None or np.isclose(first_cell_size, uniform_size, rtol=1e-3, atol=0):
        return np.linspace(0.0, thickness, n_points)
    if first_cell_size > uniform_size:
        raise ValueError(
            f"simple1d: first_cell_size ({first_cell_size} m) must not be larger than "
            f"thickness / (n_points - 1) = {uniform_size} m"
        )

    def length_minus_thickness(ratio: float) -> float:
        return first_cell_size * (ratio**n_cells - 1) / (ratio - 1) - thickness

    # At the upper ratio the last cell alone is as thick as the slab, so the root lies below
    upper_ratio = (thickness / first_cell_size) ** (1 / (n_cells - 1))
    ratio = brentq(length_minus_thickness, 1 + 1e-9, upper_ratio)
    nodes_x = np.concatenate([[0.0], np.cumsum(first_cell_size * ratio ** np.arange(n_cells))])
    nodes_x[-1] = thickness  # remove round-off
    return nodes_x


def steady_surface_temperature(
    heat_flux: np.ndarray, thickness: float, material: Material, back: BackBoundary
) -> np.ndarray:
    """Front surface temperature of slabs in steady state [K]: T_back + q L / k."""
    if back.kind == "temperature":
        back_temperature = np.full_like(heat_flux, back.temperature, dtype=float)
    elif back.kind == "htc":
        back_temperature = back.temperature + heat_flux / back.htc
    else:
        raise ValueError("simple1d: steady state with an adiabatic back has no solution.")
    return back_temperature + heat_flux * thickness / material.heat_conductivity


class TransientSlabs:
    """Backward-Euler time stepping of the 1D heat equation in many slabs at once.

    Vertex-centred finite volumes: node i owns the part of the slab between the midpoints to
    its neighbours (the two surface nodes own half a cell). Node 0 is the front surface, so
    its temperature is the surface temperature, and the scheme conserves energy exactly.

    All slabs share the grid, material and back boundary, so the matrix C/dt + K (node heat
    capacities C, conductances K) is the same for all of them and is factorized once. It is
    stored dense: with tens to a few hundred points, multithreaded dense LU solves were faster
    than banded solvers.
    """

    def __init__(
        self, nodes_x: np.ndarray, material: Material, back: BackBoundary, time_step: float
    ) -> None:
        self.back = back
        self.time_step = time_step

        cell_sizes = np.diff(nodes_x)
        node_lengths = np.zeros(len(nodes_x))
        node_lengths[:-1] += cell_sizes / 2
        node_lengths[1:] += cell_sizes / 2
        # Heat capacity of each node per unit surface area [J/(m^2 K)]
        self.capacity = material.density * material.heat_capacity * node_lengths

        matrix = np.diag(self.capacity / time_step)
        for i, cell_size in enumerate(cell_sizes):
            conductance = material.heat_conductivity / cell_size  # [W/(m^2 K)]
            matrix[i : i + 2, i : i + 2] += conductance * np.array([[1.0, -1.0], [-1.0, 1.0]])
        if back.kind == "htc":
            matrix[-1, -1] += back.htc
        elif back.kind == "temperature":
            matrix[-1] = 0.0
            matrix[-1, -1] = 1.0
        self.lu = lu_factor(matrix)

    def step(self, temperature: np.ndarray, heat_flux: np.ndarray) -> np.ndarray:
        """Advance one time step.

        ``temperature`` has shape (n_slabs, n_points) [K]: one row per slab, front surface in
        column 0. ``heat_flux`` has shape (n_slabs,) [W/m^2]. Returns the new temperature.
        """
        rhs = temperature * (self.capacity / self.time_step)
        rhs[:, 0] += heat_flux
        if self.back.kind == "htc":
            rhs[:, -1] += self.back.htc * self.back.temperature
        elif self.back.kind == "temperature":
            rhs[:, -1] = self.back.temperature

        # rhs.T has one contiguous column per slab, so LAPACK solves it without a copy
        return lu_solve(self.lu, rhs.T, overwrite_b=True, check_finite=False).T


def triangle_areas(mesh: SurfaceMesh) -> np.ndarray:
    """Area of each triangle [m^2], shape (M,)."""
    corners = mesh.nodes[mesh.triangles]
    edge_cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    return 0.5 * np.linalg.norm(edge_cross, axis=1)


def cell_to_node_average(mesh: SurfaceMesh, cell_values: np.ndarray) -> np.ndarray:
    """Area-weighted average of per-triangle values at the nodes: (Nt, M) -> (Nt, N).

    Nodes that belong to no triangle (or only to triangles of zero area) get NaN.
    """
    areas = triangle_areas(mesh)
    node_ids = mesh.triangles.ravel()
    cell_ids = np.repeat(np.arange(mesh.n_cells), 3)
    weights = scipy.sparse.csr_matrix(
        (np.repeat(areas, 3), (node_ids, cell_ids)), shape=(mesh.n_nodes, mesh.n_cells)
    )
    total_area = np.asarray(weights.sum(axis=1)).ravel()

    weighted_sums = weights @ cell_values.T  # (N, Nt)
    node_values = np.full(weighted_sums.shape, np.nan)
    has_area = total_area > 0
    node_values[has_area] = weighted_sums[has_area] / total_area[has_area, None]
    return node_values.T


class BackendSimple1D(ThermalModelBackend):
    """Fast thermal model: an independent 1D slab under every heat flux triangle.

    Config keys (SI units):

    - ``thickness`` [m], required: slab thickness L.
    - ``material_properties``, required: ``{"density": {"type": "constant", "value": rho},
      "heat_capacity": {...}, "heat_conductivity": {...}}``; only "constant" is supported.
    - ``back_boundary``, required: ``{"type": "temperature", "value": T}``,
      ``{"type": "htc", "value": h, "T_ext": T}`` or ``{"type": "adiabatic"}``.
    - ``time_stepping``: ``{"type": "steady"}`` (default) or
      ``{"type": "transient", "time_step": dt, "end_time": t_end}``; end_time must be a
      multiple of time_step. Steady with an adiabatic back has no solution.
    - ``initial_temperature`` [K], default 300: whole slab at t = 0 (transient only).
    - ``n_points``, default 51: grid points through the thickness (transient only).
    - ``first_cell_size`` [m], optional: size of the grid cell at the heated surface; cells
      then grow geometrically towards the back. Default: uniform grid. Use it when the heat
      penetrates only a thin layer during the run (depth ~ sqrt(k t / (rho c))).
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.thickness = positive_float(require_key(config, "thickness", "config"), "thickness")
        self.material = read_material(require_key(config, "material_properties", "config"))
        self.back_boundary = read_back_boundary(require_key(config, "back_boundary", "config"))
        self.initial_temperature = positive_float(
            config.get("initial_temperature", DEFAULT_INITIAL_TEMPERATURE), "initial_temperature"
        )

        self.n_points = config.get("n_points", DEFAULT_N_POINTS)
        if isinstance(self.n_points, bool) or not isinstance(self.n_points, (int, np.integer)):
            raise ValueError(f"simple1d: 'n_points' must be an integer, got {self.n_points!r}")
        if self.n_points < 3:
            raise ValueError(f"simple1d: 'n_points' must be at least 3, got {self.n_points}")
        first_cell_size = config.get("first_cell_size")
        if first_cell_size is not None:
            first_cell_size = positive_float(first_cell_size, "first_cell_size")
        self.nodes_x = slab_grid(self.thickness, self.n_points, first_cell_size)

        time_stepping = config.get("time_stepping", {"type": "steady"})
        kind = str(require_key(time_stepping, "type", "time_stepping")).lower()
        if kind == "steady":
            self.is_transient = False
            self.time_step = 0.0
            self.n_steps = 0
            if self.back_boundary.kind == "adiabatic":
                raise ValueError(
                    "simple1d: steady state with an adiabatic back has no solution (the heat "
                    "has nowhere to go). Use a transient run or a 'temperature'/'htc' back."
                )
        elif kind == "transient":
            self.is_transient = True
            self.time_step = positive_float(
                require_key(time_stepping, "time_step", "time_stepping"), "time_step"
            )
            end_time = positive_float(
                require_key(time_stepping, "end_time", "time_stepping"), "end_time"
            )
            self.n_steps = round(end_time / self.time_step)
            exact_steps = np.isclose(self.n_steps * self.time_step, end_time, rtol=1e-9, atol=0)
            if self.n_steps < 1 or not exact_steps:
                raise ValueError(
                    f"simple1d: end_time ({end_time}) must be a multiple of time_step "
                    f"({self.time_step})"
                )
            self.warn_if_surface_cell_too_thick()
        else:
            raise ValueError(
                f"simple1d time_stepping type must be 'steady' or 'transient', got {kind!r}"
            )

        self.result: TemperatureField | None = None

    def warn_if_surface_cell_too_thick(self) -> None:
        """Warn if the first ~10 time steps heat a layer thinner than the surface cell."""
        surface_cell = self.nodes_x[1]
        penetration_depth = np.sqrt(self.material.diffusivity * 10 * self.time_step)
        if surface_cell > penetration_depth:
            warnings.warn(
                f"simple1d: the surface grid cell ({surface_cell:.3g} m) is thicker than the "
                f"heat penetration depth after 10 time steps ({penetration_depth:.3g} m); early "
                "surface temperatures will be inaccurate. Increase 'n_points' or set "
                "'first_cell_size'.",
                stacklevel=2,
            )

    def run(
        self, step_dir: Path, heat_flux: HeatFluxField, mapping_tolerance: float | None = None
    ) -> None:
        """Compute the surface temperature on the heat flux mesh itself.

        Nothing is written to ``step_dir`` (irflow saves the result). There is no second mesh,
        so no overlap check is needed and ``mapping_tolerance`` is ignored.
        """
        q = heat_flux.heat_flux
        if self.is_transient:
            times, surface_temperature = self.solve_transient(q)
        else:
            surface = steady_surface_temperature(
                q, self.thickness, self.material, self.back_boundary
            )
            times, surface_temperature = np.array([0.0]), surface[np.newaxis, :]

        mesh = heat_flux.mesh
        node_temperature = cell_to_node_average(mesh, surface_temperature)
        self.result = TemperatureField(mesh=mesh, times=times, temperature=node_temperature)

    def solve_transient(self, heat_flux: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Times (Nt,) and front surface temperature per slab (Nt, M), including t = 0."""
        solver = TransientSlabs(self.nodes_x, self.material, self.back_boundary, self.time_step)
        temperature = np.full((len(heat_flux), self.n_points), self.initial_temperature)

        surface_temperature = np.empty((self.n_steps + 1, len(heat_flux)))
        surface_temperature[0] = temperature[:, 0]
        for step in range(1, self.n_steps + 1):
            temperature = solver.step(temperature, heat_flux)
            surface_temperature[step] = temperature[:, 0]

        print(
            f"simple1d: {len(heat_flux)} slabs x {self.n_points} points, "
            f"{self.n_steps} time steps of {self.time_step} s"
        )
        return self.time_step * np.arange(self.n_steps + 1), surface_temperature

    def get_result(self) -> TemperatureField:
        """Front surface temperature at the mesh nodes for every time."""
        if self.result is None:
            raise RuntimeError("simple1d has not been run yet.")
        return self.result
