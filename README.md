# PySDTorch

PySDTorch is an experimental Python package for translating Vensim system
dynamics models into a backend-agnostic intermediate representation and running
batched simulations with PyTorch.

The repository currently focuses on:

- Parsing Vensim `.mdl` text models into a compact intermediate representation.
- Expanding Vensim subscripts and lookup tables into runtime-ready variables.
- Running vectorized Torch simulations over many parameter draws.
- Sampling parameter ensembles from bounds.
- Parsing Vensim VOC files into parameter-bound dictionaries.

## Status

This project is in active development.

- The Torch runtime is the primary implemented backend.
- The NumPy runtime exists as a reference backend stub and currently raises
  `NotImplementedError`.
- Vensim support covers a useful subset of equations and functions, but not the
  full Vensim language.
- The included `Examples/LV.mdl` can be translated, but it contains
  `RANDOM PINK NOISE`, which is not currently implemented in the runtime
  translator.

## Repository Layout

```text
pysdtorch/
  model.py                 High-level load/simulate wrapper
  config.py                Simulation configuration dataclass
  sampling.py              Parameter-bound normalization helpers
  voc.py                   Vensim VOC bounds parsing and expansion
  ir/                      Backend-neutral model representation
  runtime/                 Runtime interface plus Torch/NumPy backends
  translators/             Current Vensim text translator
  legacy_translators/      Older parser structures and Vensim grammar files
  benchmarks/              Translation and runtime benchmark entry points
Examples/
  LV.mdl                   Example Vensim model
```

## Requirements

PySDTorch uses modern Python typing syntax and should be run with Python 3.10 or
newer. The Torch backend requires PyTorch.

This repository does not currently include packaging metadata such as a
`pyproject.toml`, so run commands from the repository root or add the repository
root to `PYTHONPATH`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch
```

Some legacy translator modules import `parsimonious`; install it only if you are
working directly with `pysdtorch/legacy_translators`.

## Quick Start

Translate a Vensim model to the internal representation:

```bash
python -m pysdtorch.benchmarks.translate path/to/model.mdl --no-python-compile
```

Load and simulate a supported Vensim model:

```python
from pysdtorch import SimulationConfig, load_model

config = SimulationConfig(
    backend="torch",
    device="cpu",
    dtype="float32",
    rng_seed=42,
)

model = load_model("path/to/model.mdl", config=config)

samples = model.sample_parameters(
    {
        "parameter name": (0.0, 1.0),
    },
    n_draws=1024,
)

outputs = model.simulate(
    samples,
    tracked=["stock name"],
    n_draws=1024,
)

series = outputs["stock_name"]
print(series.shape)  # torch.Size([1024, number_of_saved_timesteps])
```

Variable names are canonicalized internally by lowercasing and replacing
non-word characters with underscores. For example, `Stock Name` becomes
`stock_name` in returned output dictionaries.

## Configuration

`SimulationConfig` controls the backend and simulation settings:

```python
from pysdtorch import SimulationConfig

config = SimulationConfig(
    backend="torch",
    device="cpu",
    dtype="float32",
    initial_time=0.0,
    final_time=100.0,
    time_step=1.0,
    saveper=1.0,
    rng_seed=None,
)
```

When a Vensim model defines `INITIAL TIME`, `FINAL TIME`, `TIME STEP`, or
`SAVEPER`, those control values are merged into the runtime configuration during
`load_model`.

## VOC Bounds

VOC files can be parsed into simulation bounds:

```python
from pathlib import Path
from pysdtorch.voc import parse_voc_bounds

bounds, expansions = parse_voc_bounds(
    Path("path/to/bounds.voc"),
    mdl_path=Path("path/to/model.mdl"),
)
```

When `expand_subscripts=True`, subscripted parameters such as `Param[Region]`
are expanded into concrete per-element parameter names using the `.mdl`
subscript definitions.

## Benchmarks

Translation benchmark:

```bash
python -m pysdtorch.benchmarks.translate path/to/model.mdl
```

Runtime benchmark:

```bash
python -m pysdtorch.benchmarks.rbc
```

The runtime benchmark expects an `RBC.mdl` file at the path supplied to
`run_benchmark` or in the current working directory.

## Supported Model Features

The current translator/runtime includes support for:

- `INTEG` stocks with flow and initial expressions.
- Control variables: `INITIAL TIME`, `FINAL TIME`, `TIME STEP`, and `SAVEPER`.
- Lookup tables.
- Subscript expansion and mapped subscript handling.
- Common numeric functions such as `MAX`, `MIN`, `ABS`, `SQRT`, `EXP`, `LN`,
  `SIN`, `ZIDZ`, and `XIDZ`.
- Conditional and time functions including `IF THEN ELSE`, `STEP`, `RAMP`,
  `PULSE`, `INITIAL`, and selected delay/allocation helpers.
- Random functions including normal, uniform, Poisson, gamma, and negative
  binomial streams.

Unsupported Vensim functions generally surface as translation or Python compile
errors. Use the translation benchmark with Python compilation enabled to catch
unsupported expression rewrites early.
