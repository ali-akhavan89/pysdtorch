from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from pysdtorch import config as cfg
from pysdtorch.ir import IRModel
from pysdtorch.runtime import NumpyRuntime, Runtime, TorchRuntime
from pysdtorch.translators.vensim_text import build_ir_from_vensim


class PySDTorchModel:
    """
    High level convenience wrapper aggregating the translator and runtime.
    """

    def __init__(
        self,
        ir_model: IRModel,
        runtime: Runtime,
    ) -> None:
        self.ir_model = ir_model
        self.runtime = runtime

    @classmethod
    def load_vensim(
        cls,
        mdl_path: str | Path,
        config: cfg.SimulationConfig | None = None,
    ) -> "PySDTorchModel":
        mdl_path = Path(mdl_path)
        config = config or cfg.SimulationConfig()
        ir_model = build_ir_from_vensim(mdl_path)
        merged = cfg.merge_control_with_config(
            config,
            {name: param.value for name, param in ir_model.controls.items()},
        )
        runtime: Runtime
        if merged.backend == "torch":
            runtime = TorchRuntime(ir_model, merged)
        elif merged.backend == "numpy":
            runtime = NumpyRuntime(ir_model, merged)
        else:
            raise ValueError(f"Unsupported backend '{merged.backend}'")

        runtime.compile()
        return cls(ir_model=ir_model, runtime=runtime)

    def sample_parameters(
        self,
        bounds: Mapping[str, tuple[float, float]],
        n_draws: int,
    ):
        return self.runtime.sample_parameters(bounds, n_draws)

    def simulate(
        self,
        parameters: Mapping[str, "TensorLike"],
        tracked: Sequence[str] | None = None,
        n_draws: int = 1,
    ):
        return self.runtime.simulate(parameters, tracked=tracked, n_draws=n_draws)


def load_model(mdl_path: str | Path, config: cfg.SimulationConfig | None = None) -> PySDTorchModel:
    return PySDTorchModel.load_vensim(mdl_path=mdl_path, config=config)
