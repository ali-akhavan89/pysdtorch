from __future__ import annotations

from typing import TYPE_CHECKING

from .config import SimulationConfig  # noqa: F401

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .model import PySDTorchModel, load_model  # noqa: F401


def __getattr__(name):  # pragma: no cover - simple lazy import shim
    if name in {"PySDTorchModel", "load_model"}:
        from . import model as _model  # local import to avoid eager torch load

        return getattr(_model, name)
    raise AttributeError(f"module 'pysdtorch' has no attribute '{name}'")


__all__ = ["SimulationConfig", "PySDTorchModel", "load_model"]
