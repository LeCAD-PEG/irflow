"""Read and write irflow data in VTK formats (via pyvista).

- Heat flux: one ``.vtp`` file with cell array ``heat_flux``.
- Temperature: a ``.pvd`` collection (opens as a time series in ParaView) pointing to one
  ``.vtp`` file per time, each with point array ``temperature``.
- Camera images: a ``.pvd`` collection pointing to one ``.vti`` image per time, each with
  cell arrays ``radiance`` and ``apparent_temperature`` (one cell per pixel) and field arrays
  ``wavelength_range`` and ``assumed_emissivity``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

from irflow.data import CameraImages, HeatFluxField, SurfaceMesh, TemperatureField

HEAT_FLUX_FILE = "heat_flux.vtp"
TEMPERATURE_FILE = "temperature.pvd"
IMAGES_FILE = "images.pvd"


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


def write_collection(path: Path, times: np.ndarray, files: list[str]) -> None:
    """Write a ``.pvd`` collection: ``files`` (relative to ``path``) at ``times``."""
    collection = ET.Element("Collection")
    for time, file in zip(times, files):
        ET.SubElement(collection, "DataSet", timestep=repr(float(time)), file=file)

    root = ET.Element("VTKFile", type="Collection", version="0.1")
    root.append(collection)
    ET.ElementTree(root).write(path, xml_declaration=True)


def read_collection(path: str | Path) -> tuple[np.ndarray, list[Path]]:
    """Read a ``.pvd`` collection; return its times and the paths of its files."""
    path = Path(path)
    datasets = ET.parse(path).getroot().findall("./Collection/DataSet")
    if not datasets:
        raise ValueError(f"No datasets in {path}")

    times = np.array([float(dataset.get("timestep", 0.0)) for dataset in datasets])
    files = [path.parent / dataset.get("file", "") for dataset in datasets]
    return times, files


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

    files = []
    for index in range(len(field.times)):
        polydata = to_polydata(field.mesh)
        polydata.point_data["temperature"] = field.temperature[index]
        file_name = f"temperature_{index:04d}.vtp"
        polydata.save(series_dir / file_name)
        files.append(f"temperature/{file_name}")

    path = directory / TEMPERATURE_FILE
    write_collection(path, field.times, files)
    return path


def load_temperature(path: str | Path) -> TemperatureField:
    """Read a temperature time series from a ``.pvd`` collection written by irflow."""
    times, files = read_collection(path)
    mesh, _, _ = read_triangle_mesh(files[0])
    temperature = [read_triangle_mesh(file)[2]["temperature"] for file in files]

    return TemperatureField(mesh=mesh, times=times, temperature=np.array(temperature))


def save_images(images: CameraImages, directory: Path) -> Path:
    """Write ``<directory>/images.pvd`` and an ``images/image_NNNN.vti`` per time.

    Each pixel is one cell of the ``.vti`` image. VTK images start at the bottom row, so the
    rows are flipped on writing; ParaView then shows the image the right way up.
    """
    directory = Path(directory)
    series_dir = directory / "images"
    series_dir.mkdir(parents=True, exist_ok=True)

    n_rows, n_columns = images.radiance.shape[1:]
    files = []
    for index in range(len(images.times)):
        grid = pv.ImageData(dimensions=(n_columns + 1, n_rows + 1, 1))
        grid.cell_data["radiance"] = images.radiance[index, ::-1, :].ravel()
        grid.cell_data["apparent_temperature"] = images.apparent_temperature[index, ::-1, :].ravel()
        grid.field_data["wavelength_range"] = np.array(images.wavelength_range)
        grid.field_data["assumed_emissivity"] = np.array([images.assumed_emissivity])
        file_name = f"image_{index:04d}.vti"
        grid.save(series_dir / file_name)
        files.append(f"images/{file_name}")

    path = directory / IMAGES_FILE
    write_collection(path, images.times, files)
    return path


def load_images(path: str | Path) -> CameraImages:
    """Read camera images from a ``.pvd`` collection written by irflow."""
    times, files = read_collection(path)
    grids = []
    for file in files:
        grid = pv.read(file)
        if not isinstance(grid, pv.ImageData):
            raise ValueError(f"{file} is not a VTK image (.vti)")
        grids.append(grid)

    return CameraImages(
        times=times,
        radiance=np.array([image_cells(grid, "radiance") for grid in grids]),
        wavelength_range=tuple(grids[0].field_data["wavelength_range"]),
        apparent_temperature=np.array(
            [image_cells(grid, "apparent_temperature") for grid in grids]
        ),
        assumed_emissivity=float(grids[0].field_data["assumed_emissivity"][0]),
    )


def image_cells(grid: pv.ImageData, name: str) -> np.ndarray:
    """Cell array ``name`` of a ``.vti`` image as (rows, columns), row 0 = image top."""
    n_columns, n_rows = grid.dimensions[0] - 1, grid.dimensions[1] - 1
    return np.asarray(grid.cell_data[name]).reshape(n_rows, n_columns)[::-1, :]
