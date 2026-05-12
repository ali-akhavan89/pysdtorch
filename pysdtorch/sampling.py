from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from pysdtorch.utils import canonical_name


@dataclass(frozen=True)
class ParameterBounds:
    """
    Convenience wrapper with validation helpers.
    """

    bounds: Dict[str, Tuple[float, float]]

    def validate(self) -> None:
        for name, (lo, hi) in self.bounds.items():
            if lo >= hi:
                raise ValueError(f"Lower bound must be < upper bound for '{name}'")


def normalize_bounds(raw: Mapping[str, Tuple[float, float]]) -> ParameterBounds:
    clean = {
        canonical_name(name): (float(lo), float(hi))
        for name, (lo, hi) in raw.items()
    }
    bounds = ParameterBounds(bounds=clean)
    bounds.validate()
    return bounds
