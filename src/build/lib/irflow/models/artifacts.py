# src/irflow/models/artifacts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ArtifactOptions:
    """
    Controls what the step writes besides the backend-native outputs.

    These options are intended for consistent user experience:
      - exchange HDF5 handover file
      - optional VTK previews
      - optional images
      - log retention, dtype/compression choices
    """

    # Exchange HDF5
    write_exchange_h5: bool = True
    h5_compression: Optional[str] = "gzip"  # None disables compression
    h5_compression_level: int = 4
    float_dtype: str = "float32"
    overwrite: bool = True

    # VTK (mesh previews)
    write_vtk: bool = True
    vtk_format: str = "binary"  # "binary" | "ascii"
    vtk_fields: Optional[tuple[str, ...]] = None  # None => backend/step defaults

    # Images (camera step)
    write_images: bool = True
    image_formats: tuple[str, ...] = ("png",)  # ("png","exr","npy",...)
    write_depth: bool = False
    write_masks: bool = False

    # Logging / debug
    keep_backend_dir: bool = True
    write_run_log: bool = True
    debug_dump_inputs: bool = False

    def validate(self) -> None:
        if self.h5_compression_level < 0:
            raise ValueError("h5_compression_level must be >= 0")
        if self.vtk_format not in ("binary", "ascii"):
            raise ValueError("vtk_format must be 'binary' or 'ascii'")
        if not self.float_dtype:
            raise ValueError("float_dtype must be a non-empty string")
        if not self.image_formats:
            raise ValueError("image_formats cannot be empty if write_images=True")
