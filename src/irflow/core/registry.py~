# src/irflow/core/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Type


class RegistryError(Exception):
    pass


@dataclass
class BackendRegistry:
    """
    Maps (step_type, backend_name) -> backend class or factory.

    Example:
      registry.register("flt", "L2G", L2GBackend)
      backend = registry.get("flt", "L2G")  # instance
    """
    _entries: Dict[Tuple[str, str], Callable[[], Any]] = {}  # None

    def __post_init__(self) -> None:
        if self._entries is None:
            self._entries = {}

    def register(self, step_type: str, backend_name: str, backend: Type[Any] | Callable[[], Any]) -> None:
        key = (step_type.lower().strip(), backend_name.strip())
        if key in self._entries:
            raise RegistryError(f"Backend already registered for {key}")
        # store a factory (class -> callable that instantiates)
        if isinstance(backend, type):
            self._entries[key] = backend
        else:
            self._entries[key] = backend  # assume callable

    def get(self, step_type: str, backend_name: str) -> Any:
        key = (step_type.lower().strip(), backend_name.strip())
        if key not in self._entries:
            known = ", ".join(sorted({b for (s, b) in self._entries.keys() if s == key[0]}))
            raise RegistryError(f"Unknown backend '{backend_name}' for step '{step_type}'. Known: [{known}]")
        factory = self._entries[key]
        return factory() if isinstance(factory, type) else factory()

    def list_backends(self, step_type: str | None = None) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for (s, b) in self._entries.keys():
            if step_type is not None and s != step_type.lower().strip():
                continue
            out.setdefault(s, []).append(b)
        for s in out:
            out[s].sort()
        return out
