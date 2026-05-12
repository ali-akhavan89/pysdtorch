from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Mapping, Sequence

from pysdtorch.ir import IRModel


class Runtime(ABC):
    """
    Base class for every backend runtime implementation.
    """

    def __init__(self, ir_model: IRModel, config) -> None:
        self.model = ir_model
        self.config = config

    @abstractmethod
    def compile(self) -> None:
        """Prepare backend-specific kernels."""

    @abstractmethod
    def simulate(
        self,
        parameters: Mapping[str, "TensorLike"],
        tracked: Sequence[str] | None = None,
        n_draws: int = 1,
    ) -> Dict[str, "TensorLike"]:
        """
        Run the simulation returning the requested time-series.
        """

    @abstractmethod
    def sample_parameters(
        self,
        bounds: Mapping[str, tuple[float, float]],
        n_draws: int,
    ) -> Dict[str, "TensorLike"]:
        """
        Vectorized RNG helper for benchmark scripts.
        """
