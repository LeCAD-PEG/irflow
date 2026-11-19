"""Checks for the plain-dict configs given to simulation steps.

Configs stay ordinary dicts; these checks only make mistakes (missing keys, typos, bad time
stepping) fail early with a clear message, before any simulation runs.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Mapping
from typing import Any


def check_config(
    config: Mapping[str, Any],
    required: Iterable[str],
    optional: Iterable[str] = (),
    where: str = "config",
) -> None:
    """Raise ValueError if required keys are missing or unknown keys are present.

    ``where`` names the config in the message, e.g. "ThermalModel config (backend 'openfoam')".
    """
    required = list(required)
    allowed = set(required) | set(optional)
    missing = [key for key in required if key not in config]
    unknown = [key for key in config if key not in allowed]
    if not missing and not unknown:
        return

    lines = [f"Invalid {where}:"]
    lines += [f"  missing key '{key}'" for key in missing]
    for key in unknown:
        match = difflib.get_close_matches(key, allowed, n=1)
        hint = f" (did you mean '{match[0]}'?)" if match else ""
        lines.append(f"  unknown key '{key}'{hint}")
    lines.append(f"  allowed keys: {', '.join(sorted(allowed))}")
    raise ValueError("\n".join(lines))


def check_time_stepping(time_stepping: Mapping[str, Any], where: str = "time_stepping") -> None:
    """Check ``{"type": "steady"}`` or ``{"type": "transient", "time_step": dt, "end_time": t}``."""
    kind = str(time_stepping.get("type", "")).lower()
    if kind == "steady":
        check_config(
            time_stepping, required=["type"], optional=["time_step", "end_time"], where=where
        )
        return
    if kind != "transient":
        raise ValueError(f"{where}: 'type' must be 'steady' or 'transient', got {kind!r}")

    check_config(time_stepping, required=["type", "time_step", "end_time"], where=where)
    time_step, end_time = time_stepping["time_step"], time_stepping["end_time"]
    if not time_step > 0 or not end_time > 0:
        raise ValueError(f"{where}: time_step and end_time must be positive")
    if time_step > end_time:
        raise ValueError(f"{where}: time_step ({time_step}) is larger than end_time ({end_time})")
