from __future__ import annotations

import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pysdtorch import SimulationConfig, load_model
from pysdtorch.voc import parse_voc_bounds


N_DRAWS = 1024
TRACKED_VARIABLES = ["MN1", "MN2"]


def get_script_path(filename: str) -> Path:
    return Path(__file__).resolve().parent / filename


def parse_voc_file(model_path: Path) -> dict[str, tuple[float, float]]:
    """
    Parse the local Vensim VOC file and extract parameter bounds.

    Returns:
        dict: {parameter_name: (lower_bound, upper_bound)}
    """
    voc_path = get_script_path("Synth_Data_Gen.voc")
    bounds, _ = parse_voc_bounds(
        voc_path,
        mdl_path=model_path,
        expand_subscripts=True,
    )
    return bounds


def main() -> None:
    model_path = get_script_path("LV.mdl")
    bounds = parse_voc_file(model_path)

    config = SimulationConfig(
        backend="torch",
        device="cpu",
        dtype="float32",
        rng_seed=42,
    )
    model = load_model(model_path, config=config)

    parameter_draws = model.sample_parameters(bounds, n_draws=N_DRAWS)

    start = time.perf_counter()
    outputs = model.simulate(
        parameter_draws,
        tracked=TRACKED_VARIABLES,
        n_draws=N_DRAWS,
    )
    elapsed = time.perf_counter() - start

    print(f"Loaded model: {model_path.name}")
    print(f"Parsed {len(bounds)} parameter bounds from Synth_Data_Gen.voc")
    print(f"Simulated {N_DRAWS:,} random parameter draws in {elapsed:.4f} seconds")
    for name in sorted(outputs):
        print(f"{name}: {tuple(outputs[name].shape)}")


if __name__ == "__main__":
    main()
