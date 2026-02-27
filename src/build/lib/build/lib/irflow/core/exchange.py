# src/irflow/core/exchange.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

try:
    import h5py
except ImportError as e:  # pragma: no cover
    raise ImportError("irflow requires 'h5py' for exchange HDF5 I/O.") from e


EXCHANGE_SCHEMA_VERSION = 1


# -----------------------------
# Data containers
# -----------------------------

@dataclass(frozen=True)
class Mesh:
    """Triangular surface mesh."""
    nodes: np.ndarray      # (N,3) float
    triangles: np.ndarray  # (M,3) int


@dataclass(frozen=True)
class Exchange:
    """
    Standard handover object between steps.

    Convention:
      - fields are either cell-centered (len == M) or node-centered (len == N)
      - store a per-field attribute: location in {"cells","nodes"}
    """
    schema_version: int
    step_type: str                 # "flt" | "thermal" | "camera"
    backend: str                   # e.g. "L2G"
    mesh: Mesh
    fields: Mapping[str, np.ndarray]          # name -> array
    field_location: Mapping[str, str]         # name -> "cells" or "nodes"
    units: Mapping[str, str] = None           # name -> unit string
    metadata: Mapping[str, Any] = None        # arbitrary JSON-serializable-ish
    provenance: Mapping[str, Any] = None      # command lines, versions, hashes, etc.


# -----------------------------
# Helpers
# -----------------------------

def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _is_jsonable_scalar(x: Any) -> bool:
    return x is None or isinstance(x, (str, int, float, bool))


def _write_attrs(h5obj, attrs: Mapping[str, Any]) -> None:
    """
    Write attributes to an HDF5 object.
    HDF5 attrs must be scalars or small arrays of basic types.
    For nested metadata, we store a JSON string.
    """
    import json
    for k, v in (attrs or {}).items():
        if _is_jsonable_scalar(v):
            h5obj.attrs[k] = v
        else:
            # store non-scalar or nested dict/list as JSON
            h5obj.attrs[k] = json.dumps(v, sort_keys=True)


def _read_attrs(h5obj) -> dict[str, Any]:
    import json
    out: dict[str, Any] = {}
    for k, v in h5obj.attrs.items():
        # h5py returns bytes sometimes
        if isinstance(v, (bytes, np.bytes_)):
            v = v.decode("utf-8")
        if isinstance(v, str):
            # try parse JSON
            try:
                out[k] = json.loads(v)
            except Exception:
                out[k] = v
        else:
            # numpy scalars ok
            out[k] = v.item() if isinstance(v, np.generic) else v
    return out


def _coerce_nodes(nodes: np.ndarray) -> np.ndarray:
    nodes = np.asarray(nodes)
    _require(nodes.ndim == 2 and nodes.shape[1] == 3, "mesh.nodes must have shape (N,3)")
    _require(np.isfinite(nodes).all(), "mesh.nodes contains NaN/inf")
    return nodes.astype(np.float64, copy=False)


def _coerce_triangles(triangles: np.ndarray, n_nodes: int) -> np.ndarray:
    triangles = np.asarray(triangles)
    _require(triangles.ndim == 2 and triangles.shape[1] == 3, "mesh.triangles must have shape (M,3)")
    _require(np.isfinite(triangles).all(), "mesh.triangles contains NaN/inf")
    if not np.issubdtype(triangles.dtype, np.integer):
        # allow float but cast after checking integral-ish
        _require(np.all(np.equal(triangles, np.round(triangles))), "mesh.triangles must be integer indices")
        triangles = np.round(triangles).astype(np.int64)
    else:
        triangles = triangles.astype(np.int64, copy=False)

    _require(triangles.min() >= 0, "mesh.triangles has negative indices")
    _require(triangles.max() < n_nodes, "mesh.triangles references node index out of bounds")
    return triangles


def _infer_location(arr: np.ndarray, n_nodes: int, n_cells: int) -> str:
    """Infer 'nodes' or 'cells' based on length; raise if ambiguous."""
    arr = np.asarray(arr)
    if arr.ndim == 0:
        raise ValueError("Field arrays must be 1D or 2D; got scalar.")
    if arr.shape[0] == n_cells:
        return "cells"
    if arr.shape[0] == n_nodes:
        return "nodes"
    raise ValueError(
        f"Field length {arr.shape[0]} does not match n_cells={n_cells} or n_nodes={n_nodes}"
    )


# -----------------------------
# Public API
# -----------------------------

def validate_exchange(ex: Exchange) -> None:
    _require(ex.schema_version == EXCHANGE_SCHEMA_VERSION,
             f"Unsupported exchange schema version {ex.schema_version} "
             f"(expected {EXCHANGE_SCHEMA_VERSION})")

    nodes = _coerce_nodes(ex.mesh.nodes)
    tris = _coerce_triangles(ex.mesh.triangles, n_nodes=nodes.shape[0])

    _require(isinstance(ex.step_type, str) and ex.step_type, "exchange.step_type must be a non-empty string")
    _require(isinstance(ex.backend, str) and ex.backend, "exchange.backend must be a non-empty string")

    n_nodes = nodes.shape[0]
    n_cells = tris.shape[0]

    # fields checks
    for name, arr in (ex.fields or {}).items():
        _require(isinstance(name, str) and name, "Field name must be a non-empty string")
        a = np.asarray(arr)
        _require(a.ndim >= 1, f"Field '{name}' must have at least 1 dimension")
        _require(np.isfinite(a).all(), f"Field '{name}' contains NaN/inf")

        loc = (ex.field_location or {}).get(name)
        if loc is None:
            loc = _infer_location(a, n_nodes=n_nodes, n_cells=n_cells)
        _require(loc in ("cells", "nodes"), f"Field '{name}' location must be 'cells' or 'nodes'")
        if loc == "cells":
            _require(a.shape[0] == n_cells, f"Field '{name}' must have length n_cells={n_cells}")
        else:
            _require(a.shape[0] == n_nodes, f"Field '{name}' must have length n_nodes={n_nodes}")


def write_exchange_h5(
    ex: Exchange,
    path: str | Path,
    *,
    compression: Optional[str] = "gzip",
    compression_level: int = 4,
    float_dtype: str = "float32",
    overwrite: bool = True,
) -> Path:
    """
    Write Exchange to an HDF5 file.

    Layout:
      /exchange (attrs: schema_version, step_type, backend, metadata/provenance JSON)
      /mesh/nodes (N,3)
      /mesh/triangles (M,3)
      /fields/<name> dataset (attrs: location, unit)
    """
    path = _as_path(path)
    _ensure_parent_dir(path)

    validate_exchange(ex)

    if path.exists() and not overwrite:
        raise FileExistsError(str(path))

    # normalize dtypes
    nodes = _coerce_nodes(ex.mesh.nodes)
    tris = _coerce_triangles(ex.mesh.triangles, n_nodes=nodes.shape[0])

    # choose float dtype
    float_dt = np.dtype(float_dtype)

    with h5py.File(path, "w") as f:
        g = f.create_group("exchange")
        g.attrs["schema_version"] = int(ex.schema_version)
        g.attrs["step_type"] = ex.step_type
        g.attrs["backend"] = ex.backend

        # metadata + provenance as attrs (JSON if needed)
        _write_attrs(g, {"metadata": ex.metadata or {}, "provenance": ex.provenance or {}})

        gmesh = f.create_group("mesh")
        # mesh typically not huge; compression optional
        gmesh.create_dataset(
            "nodes",
            data=nodes.astype(np.float64, copy=False),
            compression=compression,
            compression_opts=compression_level if compression else None,
        )
        gmesh.create_dataset(
            "triangles",
            data=tris,
            compression=compression,
            compression_opts=compression_level if compression else None,
        )

        gfields = f.create_group("fields")
        units = ex.units or {}
        field_location = dict(ex.field_location or {})

        n_nodes = nodes.shape[0]
        n_cells = tris.shape[0]

        for name, arr in (ex.fields or {}).items():
            a = np.asarray(arr)
            loc = field_location.get(name)
            if loc is None:
                loc = _infer_location(a, n_nodes=n_nodes, n_cells=n_cells)

            # dtype handling: floats -> float_dtype, ints keep
            if np.issubdtype(a.dtype, np.floating):
                a_to_write = a.astype(float_dt, copy=False)
            else:
                a_to_write = a

            ds = gfields.create_dataset(
                name,
                data=a_to_write,
                compression=compression,
                compression_opts=compression_level if compression else None,
            )
            ds.attrs["location"] = loc
            if name in units:
                ds.attrs["unit"] = units[name]

    return path


def read_exchange_h5(path: str | Path) -> Exchange:
    """
    Read Exchange from HDF5 written by write_exchange_h5().
    """
    path = _as_path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    with h5py.File(path, "r") as f:
        if "exchange" not in f or "mesh" not in f or "fields" not in f:
            raise ValueError(f"Not a valid IRFlow exchange file: {path}")

        g = f["exchange"]
        schema_version = int(g.attrs.get("schema_version", -1))
        step_type = g.attrs.get("step_type", "")
        backend = g.attrs.get("backend", "")

        attrs = _read_attrs(g)
        metadata = attrs.get("metadata", {}) or {}
        provenance = attrs.get("provenance", {}) or {}

        gmesh = f["mesh"]
        nodes = np.asarray(gmesh["nodes"])
        triangles = np.asarray(gmesh["triangles"])

        gfields = f["fields"]
        fields: dict[str, np.ndarray] = {}
        field_location: dict[str, str] = {}
        units: dict[str, str] = {}

        for name in gfields.keys():
            ds = gfields[name]
            fields[name] = np.asarray(ds)
            loc = ds.attrs.get("location", None)
            if isinstance(loc, (bytes, np.bytes_)):
                loc = loc.decode("utf-8")
            if isinstance(loc, str):
                field_location[name] = loc
            unit = ds.attrs.get("unit", None)
            if isinstance(unit, (bytes, np.bytes_)):
                unit = unit.decode("utf-8")
            if isinstance(unit, str):
                units[name] = unit

    ex = Exchange(
        schema_version=schema_version,
        step_type=str(step_type),
        backend=str(backend),
        mesh=Mesh(nodes=nodes, triangles=triangles),
        fields=fields,
        field_location=field_location,
        units=units,
        metadata=metadata,
        provenance=provenance,
    )
    # validate now (also enforces supported schema)
    validate_exchange(ex)
    return ex


def exchange_field_info(ex: Exchange) -> dict[str, dict[str, Any]]:
    """
    Convenience: returns per-field {location, unit, shape, dtype}.
    """
    out: dict[str, dict[str, Any]] = {}
    for k, v in (ex.fields or {}).items():
        out[k] = {
            "location": (ex.field_location or {}).get(k, None),
            "unit": (ex.units or {}).get(k, None),
            "shape": tuple(np.asarray(v).shape),
            "dtype": str(np.asarray(v).dtype),
        }
    return out
