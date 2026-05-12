from __future__ import annotations

from typing import Dict, Mapping, Sequence

from pysdtorch.ir import IRModel
from pysdtorch.runtime.base import Runtime


class NumpyRuntime(Runtime):
    """
    Minimal NumPy reference implementation used for regression tests.
    """

    def __init__(self, ir_model: IRModel, config) -> None:
        super().__init__(ir_model, config)

    def compile(self) -> None:  # pragma: no cover - noop
        return None

    def simulate(
        self,
        parameters: Mapping[str, "ndarray"],
        tracked: Sequence[str] | None = None,
        n_draws: int = 1,
    ) -> Dict[str, "ndarray"]:
        raise NotImplementedError("NumPy runtime not yet implemented.")

    def sample_parameters(
        self,
        bounds: Mapping[str, tuple[float, float]],
        n_draws: int,
    ) -> Dict[str, "ndarray"]:
        raise NotImplementedError("NumPy runtime not yet implemented.")
