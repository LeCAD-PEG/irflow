"""Tests for reading OpenFOAM polyMesh patches as triangle surfaces."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from irflow.backends.tm.openfoam.polymesh import (
    merge_surfaces,
    read_boundary_surface,
    read_patches,
)

HEADER = (
    "FoamFile\n{{\n    format      ascii;\n    class       {cls};\n    object      {obj};\n}}\n"
)

POINTS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (9, 9, 9)]
# face 0: internal; face 1: quad on patch "top"; face 2: triangle on interface "a_to_b"
FACES = ["3(0 1 4)", "4(0 1 2 3)", "3(1 2 4)"]
BOUNDARY = """
2
(
    top
    {
        type            patch;
        nFaces          1;
        startFace       1;
    }
    a_to_b
    {
        type            mappedWall;
        inGroups        List<word>
2
(
wall
mapped
)
;
        nFaces          1;
        startFace       2;
    }
)
"""


@pytest.fixture
def poly_mesh(tmp_path: Path) -> Path:
    directory = tmp_path / "polyMesh"
    directory.mkdir()
    points = "".join(f"({x} {y} {z})\n" for x, y, z in POINTS)
    (directory / "points").write_text(
        HEADER.format(cls="vectorField", obj="points") + f"\n{len(POINTS)}\n(\n{points})\n"
    )
    faces = "".join(face + "\n" for face in FACES)
    (directory / "faces").write_text(
        HEADER.format(cls="faceList", obj="faces") + f"\n{len(FACES)}\n(\n{faces})\n"
    )
    (directory / "boundary").write_text(
        HEADER.format(cls="polyBoundaryMesh", obj="boundary") + BOUNDARY
    )
    return directory


def test_read_patches(poly_mesh: Path) -> None:
    assert read_patches(poly_mesh / "boundary") == {"top": (1, 1), "a_to_b": (2, 1)}


def test_outer_surface_skips_region_interfaces(poly_mesh: Path) -> None:
    surface = read_boundary_surface(poly_mesh)
    # the quad is split into two triangles; only its 4 points are kept
    assert surface.n_cells == 2
    np.testing.assert_allclose(surface.nodes, [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    np.testing.assert_array_equal(surface.triangles, [[0, 1, 2], [0, 2, 3]])


def test_selected_patches(poly_mesh: Path) -> None:
    surface = read_boundary_surface(poly_mesh, patches=["a_to_b"])
    np.testing.assert_allclose(surface.nodes, [(1, 0, 0), (1, 1, 0), (0, 0, 1)])
    with pytest.raises(KeyError, match="nope"):
        read_boundary_surface(poly_mesh, patches=["nope"])


def test_merge_surfaces(poly_mesh: Path) -> None:
    first = read_boundary_surface(poly_mesh)
    second = read_boundary_surface(poly_mesh, patches=["a_to_b"])
    merged = merge_surfaces([first, second])
    assert (merged.n_nodes, merged.n_cells) == (7, 3)
    np.testing.assert_array_equal(merged.triangles[-1], [4, 5, 6])
