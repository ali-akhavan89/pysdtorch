from __future__ import annotations

import argparse
import time
from pathlib import Path

from pysdtorch.translators.vensim_text import build_ir_from_vensim


def run_benchmark(mdl_path: str | Path, python_compile: bool = True) -> None:
    mdl_path = Path(mdl_path)
    start = time.perf_counter()
    model = build_ir_from_vensim(mdl_path)
    translate_time = time.perf_counter() - start
    print(
        f"Translated {mdl_path.name}: "
        f"{len(model.stocks)} stocks, {len(model.auxiliaries)} auxiliaries, "
        f"{len(model.controls)} controls in {translate_time:.3f}s"
    )

    if not python_compile:
        return

    start = time.perf_counter()
    for aux in model.auxiliaries.values():
        compile(aux.expression.source, aux.name, "eval")
    for stock in model.stocks.values():
        compile(stock.flow.source, f"{stock.name}_flow", "eval")
        compile(stock.initial.source, f"{stock.name}_init", "eval")
    compile_time = time.perf_counter() - start
    print(f"Python compile check: {compile_time:.3f}s")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Vensim model translation.")
    parser.add_argument("mdl_path", nargs="?", default="City.mdl")
    parser.add_argument(
        "--no-python-compile",
        action="store_true",
        help="Skip compiling translated expressions with Python's compiler.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_benchmark(args.mdl_path, python_compile=not args.no_python_compile)

