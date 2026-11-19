"""Read surfaces of an OpenFOAM mesh (ASCII ``constant/<region>/polyMesh``) as triangle meshes.

Used to put parts of the thermal model geometry, e.g. a housing, into the optical scene.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from irflow.data import SurfaceMesh

POINT_PATTERN = re.compile(r"\(\s*([-+\deE.]+)\s+([-+\deE.]+)\s+([-+\deE.]+)\s*\)")
FACE_PATTERN = re.compile(r"(\d+)\(([\d\s]*)\)")


def list_body(path: Path) -> str:
    """Text of an OpenFOAM list file after its header (from the entry count on)."""
    text = path.read_text()
    if "format      binary" in text[:1000] or "format binary" in text[:1000]:
        raise ValueError(f"{path} is binary; only ASCII polyMesh files can be read")
    header_end = text.index("}", text.index("FoamFile")) + 1
    return text[header_end:]


def read_points(path: Path) -> np.ndarray:
    """Points of ``polyMesh/points``, shape (N, 3)."""
    points = np.array(POINT_PATTERN.findall(list_body(path)), dtype=float)
    return points.reshape(-1, 3)


def read_faces(path: Path) -> list[np.ndarray]:
    """Faces of ``polyMesh/faces``, each an array of point indices."""
    faces = []
    for count, indices in FACE_PATTERN.findall(list_body(path)):
        face = np.array(indices.split(), dtype=int)
        if len(face) != int(count):
            raise ValueError(f"Malformed face in {path}: {count}({indices})")
        faces.append(face)
    return faces


def read_patches(path: Path) -> dict[str, tuple[int, int]]:
    """Patches of ``polyMesh/boundary``: ``{name: (start_face, n_faces)}``."""
    lines = list_body(path).splitlines()
    patches: dict[str, tuple[int, int]] = {}
    name: str | None = None
    values: dict[str, str] = {}
    previous = ""
    for line in lines:
        stripped = line.strip()
        if name is None:
            # A patch starts with its name on one line and "{" on the next
            if stripped == "{" and previous:
                name, values = previous, {}
        elif line.startswith("    }"):
            patches[name] = (int(values["startFace"]), int(values["nFaces"]))
            name = None
        else:
            parts = stripped.rstrip(";").split()
            if len(parts) == 2 and parts[0] in ("startFace", "nFaces"):
                values[parts[0]] = parts[1]
        if stripped:
            previous = stripped
    return patches


def is_region_interface(patch: str) -> bool:
    """Interfaces between regions created by splitMeshRegions are named ``<a>_to_<b>``."""
    return "_to_" in patch


def read_boundary_surface(poly_mesh: Path, patches: Iterable[str] | None = None) -> SurfaceMesh:
    """Triangle mesh of the given patches of a ``polyMesh`` directory.

    Default: all patches that are not interfaces to other regions, i.e. the outer surface of
    the region. Polygonal faces are split into triangles (fan from their first point); only
    the points used by these faces are kept.
    """
    poly_mesh = Path(poly_mesh)
    all_patches = read_patches(poly_mesh / "boundary")
    if patches is None:
        names = [name for name in all_patches if not is_region_interface(name)]
    else:
        names = list(patches)
        missing = [name for name in names if name not in all_patches]
        if missing:
            raise KeyError(f"Patches {missing} not in {poly_mesh}; found {sorted(all_patches)}")

    faces = read_faces(poly_mesh / "faces")
    triangles = []
    for name in names:
        start, count = all_patches[name]
        for face in faces[start : start + count]:
            for i in range(1, len(face) - 1):
                triangles.append((face[0], face[i], face[i + 1]))
    if not triangles:
        raise ValueError(f"No faces in patches {names} of {poly_mesh}")

    points = read_points(poly_mesh / "points")
    triangles_array = np.array(triangles, dtype=int)
    used = np.unique(triangles_array)
    new_index = np.full(len(points), -1)
    new_index[used] = np.arange(len(used))
    return SurfaceMesh(nodes=points[used], triangles=new_index[triangles_array])


def merge_surfaces(meshes: Sequence[SurfaceMesh]) -> SurfaceMesh:
    """One mesh holding all triangles of ``meshes`` (shared points are not merged)."""
    offsets = np.cumsum([0] + [mesh.n_nodes for mesh in meshes[:-1]])
    nodes = np.vstack([mesh.nodes for mesh in meshes])
    triangles = np.vstack([mesh.triangles + offset for mesh, offset in zip(meshes, offsets)])
    return SurfaceMesh(nodes=nodes, triangles=triangles)
