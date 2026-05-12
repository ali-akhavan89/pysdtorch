from __future__ import annotations

import time
from pathlib import Path

from pysdtorch import SimulationConfig, load_model


PARAM_BOUNDS = {
    "AGrowth": (1.0, 1.1),
    "alpha": (0.15, 0.45),
    "b": (0.05, 0.2),
    "delta": (0.01, 0.1),
    "Init K": (1.0, 20.0),
    "Init Tech": (1.0, 3.0),
    "sp1": (0.0, 0.2),
    "sp2": (0.0, 0.05),
}


def run_benchmark(mdl_path: str | Path = "RBC.mdl", draws: int = 100_000) -> None:
    config = SimulationConfig(
        backend="torch",
        device="cpu",
        dtype="float32",
        rng_seed=42,
        compile=False,
    )
    model = load_model(mdl_path, config=config)
    samples = model.sample_parameters(PARAM_BOUNDS, n_draws=draws)
    start = time.perf_counter()
    outputs = model.simulate(samples, tracked=["mn1", "mn2"], n_draws=draws)
    duration = time.perf_counter() - start
    mn1 = outputs["mn1"]
    mn2 = outputs["mn2"]
    print(
        f"Ran {draws:,} draws in {duration:.2f}s "
        f"(MN1 mean={mn1.mean():.4f}, MN2 mean={mn2.mean():.4f})"
    )


if __name__ == "__main__":
    run_benchmark()
