from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PLOT_CACHE_DIR = Path(tempfile.gettempdir()) / "pysdtorch_plot_cache"
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pysdtorch import SimulationConfig, load_model
from pysdtorch.voc import parse_voc_bounds


N_DRAWS = 50
TRACKED_VARIABLES = ["MN1", "MN2"]
OUTPUT_PDF = "plot.pdf"


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


def build_time_axis(model, n_points: int) -> np.ndarray:
    config = model.runtime.config
    saveper = config.saveper if config.saveper is not None else config.time_step
    return config.initial_time + np.arange(n_points) * saveper


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
    outputs = model.simulate(
        parameter_draws,
        tracked=TRACKED_VARIABLES,
        n_draws=N_DRAWS,
    )

    mn1 = outputs["mn1"].detach().cpu().numpy()
    mn2 = outputs["mn2"].detach().cpu().numpy()
    time_axis = build_time_axis(model, n_points=mn1.shape[1])

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(8.0, 6.0),
        sharex=True,
        constrained_layout=True,
    )
    axes[0].plot(time_axis, mn1.T, color="#1f77b4", alpha=0.28, linewidth=0.8)
    axes[0].set_title("MN1")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time_axis, mn2.T, color="#d62728", alpha=0.28, linewidth=0.8)
    axes[1].set_title("MN2")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Value")
    axes[1].grid(True, alpha=0.25)

    output_path = get_script_path(OUTPUT_PDF)
    fig.savefig(output_path, format="pdf")
    plt.close(fig)

    print(f"Saved {N_DRAWS} simulated time series to {output_path}")
    print(f"MN1: {mn1.shape}")
    print(f"MN2: {mn2.shape}")


if __name__ == "__main__":
    main()
