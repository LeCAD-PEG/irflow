"""Read and write irflow data in VTK formats (via pyvista).

- Heat flux: one ``.vtp`` file with cell array ``heat_flux``.
- Temperature: a ``.pvd`` collection (opens as a time series in ParaView) pointing to one
  ``.vtp`` file per time, each with point array ``temperature``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from irflow.data import HeatFluxField, SurfaceMesh, TemperatureField

HEAT_FLUX_FILE = "heat_flux.vtp"
TEMPERATURE_FILE = "temperature.pvd"


def read_triangle_mesh(path: str | Path) -> tuple[SurfaceMesh, dict[str, Any], dict[str, Any]]:
    """Read a triangular surface mesh from any VTK file pyvista can open.

    Returns the mesh, its cell arrays and its point arrays.
    """
    data = pv.read(path)
    triangles: np.ndarray
    if isinstance(data, pv.PolyData):
        triangles = data.regular_faces
    elif isinstance(data, pv.UnstructuredGrid):
        if set(data.celltypes) != {pv.CellType.TRIANGLE}:
            raise ValueError(f"Expected only triangles in {path}")
        triangles = data.cells_dict[pv.CellType.TRIANGLE]
    else:
        raise ValueError(f"Unsupported VTK dataset {type(data).__name__} in {path}")

    mesh = SurfaceMesh(nodes=np.asarray(data.points), triangles=triangles)
    cell_data = {name: np.asarray(data.cell_data[name]) for name in data.cell_data}
    point_data = {name: np.asarray(data.point_data[name]) for name in data.point_data}
    return mesh, cell_data, point_data


def to_polydata(mesh: SurfaceMesh) -> pv.PolyData:
    faces = np.hstack([np.full((mesh.n_cells, 1), 3), mesh.triangles]).ravel()
    return pv.PolyData(mesh.nodes, faces)


def save_heat_flux(field: HeatFluxField, directory: Path) -> Path:
    """Write ``<directory>/heat_flux.vtp``."""
    polydata = to_polydata(field.mesh)
    polydata.cell_data["heat_flux"] = field.heat_flux

    path = Path(directory) / HEAT_FLUX_FILE
    polydata.save(path)
    return path


def load_heat_flux(path: str | Path, array_name: str = "heat_flux") -> HeatFluxField:
    """Read heat flux from the cell array ``array_name`` of a triangular VTK surface.

    Works for files written by irflow and for other codes' output, e.g. GYROHIT
    (``array_name="Heat-flux[W/m^2]"``) or Riptrace (``array_name="Q_cells"``).
    """
    mesh, cell_data, _ = read_triangle_mesh(path)
    if array_name not in cell_data:
        raise KeyError(f"No cell array '{array_name}' in {path}; found {sorted(cell_data)}")
    return HeatFluxField(mesh=mesh, heat_flux=cell_data[array_name])


def save_temperature(field: TemperatureField, directory: Path) -> Path:
    """Write ``<directory>/temperature.pvd`` and a ``temperature/temperature_NNNN.vtp`` per time."""
    directory = Path(directory)
    series_dir = directory / "temperature"
    series_dir.mkdir(parents=True, exist_ok=True)

    collection = ET.Element("Collection")
    for index, time in enumerate(field.times):
        polydata = to_polydata(field.mesh)
        polydata.point_data["temperature"] = field.temperature[index]
        file_name = f"temperature_{index:04d}.vtp"
        polydata.save(series_dir / file_name)
        ET.SubElement(
            collection, "DataSet", timestep=repr(float(time)), file=f"temperature/{file_name}"
        )

    root = ET.Element("VTKFile", type="Collection", version="0.1")
    root.append(collection)
    path = directory / TEMPERATURE_FILE
    ET.ElementTree(root).write(path, xml_declaration=True)
    return path


def load_temperature(path: str | Path) -> TemperatureField:
    """Read a temperature time series from a ``.pvd`` collection written by irflow."""
    path = Path(path)
    datasets = ET.parse(path).getroot().findall("./Collection/DataSet")
    if not datasets:
        raise ValueError(f"No datasets in {path}")

    times = np.array([float(dataset.get("timestep", 0.0)) for dataset in datasets])
    files = [path.parent / dataset.get("file", "") for dataset in datasets]

    mesh, _, _ = read_triangle_mesh(files[0])
    temperature = [read_triangle_mesh(file)[2]["temperature"] for file in files]

    return TemperatureField(mesh=mesh, times=times, temperature=np.array(temperature))
