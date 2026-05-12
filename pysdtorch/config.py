from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


BackendKind = Literal["torch", "numpy"]


@dataclass(slots=True)
class SimulationConfig:
    """
    Shared runtime configuration.

    The config stays backend agnostic so callers can instantiate it
    without importing heavy dependencies such as ``torch``.
    """

    backend: BackendKind = "torch"
    device: str = "cpu"
    dtype: str = "float32"
    compile: bool = True
    fused_eval: bool = True
    indexed_env_eval: bool = True
    prune_eval: bool = True
    hoist_static_auxiliaries: bool = True
    precompute_time_grid: bool = True
    initial_time: float = 0.0
    final_time: float = 100.0
    time_step: float = 1.0
    saveper: float = 1.0
    rng_seed: Optional[int] = None
    progress: bool = False
    torch_compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = (
        "default"
    )

    @property
    def steps(self) -> int:
        return int(round((self.final_time - self.initial_time) / self.time_step)) + 1


def merge_control_with_config(config: SimulationConfig, controls: dict[str, float]) -> SimulationConfig:
    """
    Produce a new config merging values emitted by the translator with the
    user supplied config.
    """

    data = {
        "initial_time": controls.get("initial_time", config.initial_time),
        "final_time": controls.get("final_time", config.final_time),
        "time_step": controls.get("time_step", config.time_step),
        "saveper": controls.get("saveper", config.saveper),
    }
    return SimulationConfig(
        backend=config.backend,
        device=config.device,
        dtype=config.dtype,
        compile=config.compile,
        fused_eval=config.fused_eval,
        indexed_env_eval=config.indexed_env_eval,
        prune_eval=config.prune_eval,
        hoist_static_auxiliaries=config.hoist_static_auxiliaries,
        precompute_time_grid=config.precompute_time_grid,
        initial_time=data["initial_time"],
        final_time=data["final_time"],
        time_step=data["time_step"],
        saveper=data["saveper"],
        rng_seed=config.rng_seed,
        progress=config.progress,
        torch_compile_mode=config.torch_compile_mode,
    )
