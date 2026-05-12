from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from types import CodeType
from typing import Callable, Dict, Iterable, Mapping, Sequence

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("KMP_CREATE_SHM", "FALSE")
os.environ.setdefault("KMP_USE_SHM", "FALSE")
os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
from torch import Tensor

from pysdtorch.ir import IRModel, dependency_graph, topo_sort
from pysdtorch.runtime.base import Runtime
from pysdtorch.sampling import normalize_bounds
from pysdtorch.utils import canonical_name


_SAVE_RELATIVE_PRECISION = 1e-5
_SMALL_VENSIM = 1e-6
_STAGE_FLAG = "__pysdtorch_initializing"
_ALLOCATE_AVAILABLE_ITERS = 32
_ALLOCATE_AVAILABLE_TARGET_TOL = 1e-6


_FUSED_ENSURE_TENSOR = "__pysdtorch_ensure_tensor"
_FUSED_FLOW_PREFIX = "__pysdtorch_flow_"

_DYNAMIC_AUX_FUNCTIONS = {
    "random_normal",
    "random_uniform",
    "random_poisson",
    "random_gamma",
    "random_negative_binomial",
    "t_step",
    "t_ramp",
    "t_pulse",
    "t_delay_n",
    "t_allocate_available",
}


@dataclass(frozen=True)
class _FusedExecBlock:
    filename: str
    code: CodeType
    line_to_name: Dict[int, str]


@dataclass(frozen=True)
class _IndexedFunctionBlock:
    filename: str
    func: Callable[..., None]
    line_to_name: Dict[int, str]


class _LookupTable:
    def __init__(
        self,
        name: str,
        x: Sequence[float],
        y: Sequence[float],
        runtime: "TorchRuntime",
    ) -> None:
        self._name = name
        self._runtime = runtime
        self._x = torch.tensor(x, dtype=runtime.dtype, device=runtime.device)
        self._y = torch.tensor(y, dtype=runtime.dtype, device=runtime.device)
        if self._x.ndim != 1 or self._y.ndim != 1:
            raise ValueError(f"Lookup '{name}' requires 1D x/y data.")
        if self._x.numel() != self._y.numel():
            raise ValueError(f"Lookup '{name}' has mismatched x/y lengths.")
        if self._x.numel() == 0:
            raise ValueError(f"Lookup '{name}' has no data points.")
        if self._x.numel() > 1:
            if not torch.all(self._x[1:] >= self._x[:-1]):
                raise ValueError(f"Lookup '{name}' x values must be sorted.")

    def __call__(self, x):
        runtime = self._runtime
        x_t = runtime._ensure_tensor(x, runtime._batch)
        n = self._x.numel()
        if n == 1:
            return self._y[0].expand_as(x_t)

        idx = torch.searchsorted(self._x, x_t)
        below = idx == 0
        above = idx == n
        idx0 = torch.clamp(idx - 1, 0, n - 1)
        idx1 = torch.clamp(idx, 0, n - 1)
        x0 = self._x[idx0]
        x1 = self._x[idx1]
        y0 = self._y[idx0]
        y1 = self._y[idx1]
        denom = x1 - x0
        weight = torch.where(denom != 0, (x_t - x0) / denom, torch.zeros_like(x_t))
        out = y0 + weight * (y1 - y0)
        out = torch.where(below, self._y[0], out)
        out = torch.where(above, self._y[-1], out)
        return out


class _EnvIndexTransformer(ast.NodeTransformer):
    def __init__(self, env_index: Mapping[str, int], stage: str | None) -> None:
        super().__init__()
        self._env_index = env_index
        self._stage = stage

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        if (
            self._stage in {"run", "init"}
            and isinstance(node.test, ast.Name)
            and isinstance(node.test.ctx, ast.Load)
            and node.test.id == _STAGE_FLAG
        ):
            branch = node.body if self._stage == "init" else node.orelse
            return self.visit(branch)
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self._env_index:
            idx = self._env_index[node.id]
            replacement = ast.Subscript(
                value=ast.Name(id="env", ctx=ast.Load()),
                slice=ast.Constant(value=idx),
                ctx=node.ctx,
            )
            return ast.copy_location(replacement, node)
        return node


class TorchRuntime(Runtime):
    """
    Vectorised Torch runtime supporting batched simulations.
    """

    def __init__(self, ir_model: IRModel, config) -> None:
        super().__init__(ir_model, config)
        self.device = torch.device(config.device)
        self.dtype = self._resolve_dtype(config.dtype)
        torch.set_num_threads(1)
        self._fused_eval = bool(getattr(config, "fused_eval", True))
        self._indexed_env_eval = bool(getattr(config, "indexed_env_eval", False))
        self._prune_eval = bool(getattr(config, "prune_eval", True))
        self._hoist_static_auxiliaries = bool(
            getattr(config, "hoist_static_auxiliaries", True)
        )
        self._precompute_time_grid = bool(getattr(config, "precompute_time_grid", True))
        self._compiled = False
        self._aux_order_full: list[str] = []
        self._aux_order_init_full: list[str] = []
        self._aux_order_run_static_full: list[str] = []
        self._aux_order_run_dynamic_full: list[str] = []
        self._aux_order_run_static: list[str] = []
        self._aux_order_run_dynamic: list[str] = []
        self._aux_order_init_run: list[str] = []
        self._eval_globals: dict | None = None
        self._batch = 1
        self._dt_tensor: Tensor | None = None
        self._time_tensor: Tensor | None = None
        self._time_grid: Tensor | None = None
        self._overrides: Dict[str, Tensor] = {}
        self._override_aux_names: frozenset[str] = frozenset()
        self._initialising_stocks: set[str] = set()
        self._evaluating_aux: set[str] = set()
        self._stock_names = set()
        self._aux_names = set()
        self._aux_deps_run: Dict[str, tuple[str, ...]] = {}
        self._aux_deps_init: Dict[str, tuple[str, ...]] = {}
        self._flow_aux_start: frozenset[str] = frozenset()
        self._initial_cache: Dict[str, Tensor] = {}
        self._delay_n_state: Dict[str, Tensor] = {}
        self._delay_n_times: Dict[str, Tensor] = {}
        self._allocate_available_cache: Dict[str, Tensor] = {}
        self._eval_stage: str = "run"
        self._flow_fused_block: _FusedExecBlock | None = None
        self._aux_fused_block_run_static: _FusedExecBlock | None = None
        self._aux_fused_block_run_dynamic: _FusedExecBlock | None = None
        self._aux_fused_block_init: _FusedExecBlock | None = None
        self._lookup_tables: Dict[str, _LookupTable] = {}
        self._env_names: list[str] = []
        self._env_index: dict[str, int] = {}
        self._env_time_idx = -1
        self._env_stage_idx = -1
        self._indexed_expr_cache: dict[tuple[str, str], str] = {}
        self._flow_indexed_block: _IndexedFunctionBlock | None = None

    def compile(self) -> None:
        aux_vars = list(self.model.auxiliaries.values())
        exclude = set(self.model.stock_names()) | set(self.model.controls.keys())
        graph = dependency_graph(aux_vars, exclude=exclude)
        self._aux_order_full = topo_sort([aux.name for aux in aux_vars], graph)
        graph_init = {
            aux.name: set(aux.expression.init_dependencies).difference(exclude)
            for aux in aux_vars
        }
        self._aux_order_init_full = topo_sort([aux.name for aux in aux_vars], graph_init)
        self._stock_names = set(self.model.stocks.keys())
        self._aux_names = set(self.model.auxiliaries.keys())
        self._aux_deps_run = {
            aux.name: tuple(dep for dep in aux.expression.dependencies if dep in self._aux_names)
            for aux in aux_vars
        }
        self._aux_deps_init = {
            aux.name: tuple(dep for dep in aux.expression.init_dependencies if dep in self._aux_names)
            for aux in aux_vars
        }
        self._flow_aux_start = frozenset(
            dep
            for stock in self.model.stocks.values()
            for dep in stock.flow.dependencies
            if dep in self._aux_names
        )
        if self._hoist_static_auxiliaries:
            static_aux = self._compute_static_auxiliaries_run(aux_vars)
            self._aux_order_run_static_full = [
                name for name in self._aux_order_full if name in static_aux
            ]
            self._aux_order_run_dynamic_full = [
                name for name in self._aux_order_full if name not in static_aux
            ]
        else:
            self._aux_order_run_static_full = []
            self._aux_order_run_dynamic_full = list(self._aux_order_full)
        self._aux_order_run_static = []
        self._aux_order_run_dynamic = list(self._aux_order_run_dynamic_full)
        self._aux_order_init_run = list(self._aux_order_init_full)

        for aux in aux_vars:
            aux.expression.compiled = compile(aux.expression.source, aux.name, "eval")

        for stock in self.model.stocks.values():
            stock.flow.compiled = compile(
                stock.flow.source, f"{stock.name}_flow", "eval"
            )
            stock.initial.compiled = compile(
                stock.initial.source, f"{stock.name}_init", "eval"
            )

        self._eval_globals = self._build_eval_env()
        self._register_lookups()
        if self._indexed_env_eval:
            self._build_env_layout()
            self._flow_indexed_block = self._build_indexed_flow_block()
        elif self._fused_eval:
            self._flow_fused_block = self._build_fused_flow_block()
        self._compiled = True

    def simulate(
        self,
        parameters: Mapping[str, object],
        tracked: Sequence[str] | None = None,
        n_draws: int = 1,
    ) -> Dict[str, Tensor]:
        self._assert_compiled()
        tracked = tracked or []
        track_set = {canonical_name(name) for name in tracked}
        if not track_set:
            track_set = set(self.model.stock_names())
        batch = self._resolve_batch(parameters, n_draws)
        self._batch = batch
        self._dt_tensor = torch.full(
            (batch,), self.config.time_step, dtype=self.dtype, device=self.device
        )
        self._delay_n_state = {}
        self._delay_n_times = {}
        self._allocate_available_cache = {}
        overrides = self._prepare_parameters(parameters, batch)
        self._overrides = dict(overrides)
        self._override_aux_names = frozenset(
            name for name in self._overrides.keys() if name in self._aux_names
        )
        if self._prune_eval:
            required_run_aux = self._collect_aux_closure(
                set(self._flow_aux_start).union(track_set.intersection(self._aux_names)),
                self._aux_deps_run,
            )
            required_init_aux = self._collect_aux_closure(
                required_run_aux, self._aux_deps_init
            )
        else:
            required_run_aux = frozenset(self._aux_names)
            required_init_aux = frozenset(self._aux_names)

        self._aux_order_run_static = [
            name for name in self._aux_order_run_static_full if name in required_run_aux
        ]
        self._aux_order_run_dynamic = [
            name
            for name in self._aux_order_run_dynamic_full
            if name in required_run_aux
        ]
        self._aux_order_init_run = [
            name for name in self._aux_order_init_full if name in required_init_aux
        ]

        if self._indexed_env_eval:
            return self._simulate_indexed(
                overrides=overrides,
                track_set=track_set,
                batch=batch,
            )

        if self._fused_eval:
            self._aux_fused_block_run_static = self._build_fused_aux_block(
                self._aux_order_run_static,
                filename="<pysdtorch_fused_aux_run_static>",
                override_aux_names=self._override_aux_names,
            )
            self._aux_fused_block_run_dynamic = self._build_fused_aux_block(
                self._aux_order_run_dynamic,
                filename="<pysdtorch_fused_aux_run_dynamic>",
                override_aux_names=self._override_aux_names,
            )
            self._aux_fused_block_init = self._build_fused_aux_block(
                self._aux_order_init_run,
                filename="<pysdtorch_fused_aux_init>",
                override_aux_names=self._override_aux_names,
            )
        self._initial_cache = {}
        context = {**self._static_controls(batch), **overrides}
        context[_STAGE_FLAG] = True
        time_tensor = self._time_for_step(step_idx=0, batch=batch)
        context["time"] = time_tensor
        self._time_tensor = time_tensor
        rng = _RandomStream(batch, self.dtype, self.device, self.config.rng_seed)
        self._eval_globals["random_normal"] = rng.random_normal  # type: ignore[index]
        self._eval_globals["random_uniform"] = rng.random_uniform  # type: ignore[index]
        self._eval_globals["random_poisson"] = rng.random_poisson  # type: ignore[index]
        self._eval_globals["random_gamma"] = rng.random_gamma  # type: ignore[index]
        self._eval_globals["random_negative_binomial"] = rng.random_negative_binomial  # type: ignore[index]

        # Initialise stocks
        self._initialising_stocks.clear()
        self._evaluating_aux.clear()
        for name in self.model.stocks.keys():
            self._initialise_stock(name, context)
        self._initialising_stocks.clear()

        # Populate INITIAL caches and any initialization-stage auxiliaries.
        self._evaluate_auxiliaries_init(context)

        # Switch to run stage and drop initialization-stage auxiliary values.
        context[_STAGE_FLAG] = False
        for name in self._aux_names.difference(self._override_aux_names):
            context.pop(name, None)

        self._evaluate_auxiliaries_static(context)

        record_steps = self._compute_record_steps()
        record_capacity = len(record_steps)
        outputs = self._init_outputs(track_set, batch, record_capacity)

        self._evaluate_auxiliaries(context)
        record_cursor = self._record_if_scheduled(
            outputs,
            track_set,
            context,
            step_idx=0,
            record_cursor=0,
            record_steps=record_steps,
            record_capacity=record_capacity,
        )

        steps = self.config.steps
        for step_idx in range(1, steps):
            self._allocate_available_cache.clear()
            if self._fused_eval and self._flow_fused_block is not None:
                self._exec_fused_flows(context)
                for name, stock in self.model.stocks.items():
                    flow = context[self._flow_var_name(name)]
                    updated = context[name] + flow * self._dt_tensor
                    if stock.non_negative:
                        updated = torch.clamp(updated, min=0.0)
                    context[name] = updated
            else:
                flows = {}
                for name, stock in self.model.stocks.items():
                    try:
                        flows[name] = self._evaluate_expression(stock.flow, context)
                    except Exception as exc:  # pragma: no cover - diagnostics
                        raise RuntimeError(
                            f"Error evaluating flow for stock '{name}': {exc} (expr: {stock.flow.source})"
                        ) from exc
                for name, flow in flows.items():
                    updated = context[name] + flow * self._dt_tensor
                    if self.model.stocks[name].non_negative:
                        updated = torch.clamp(updated, min=0.0)
                    context[name] = updated

            time_tensor = self._time_for_step(step_idx=step_idx, batch=batch)
            context["time"] = time_tensor
            self._time_tensor = time_tensor
            self._evaluate_auxiliaries(context)
            record_cursor = self._record_if_scheduled(
                outputs,
                track_set,
                context,
                step_idx=step_idx,
                record_cursor=record_cursor,
                record_steps=record_steps,
                record_capacity=record_capacity,
            )

        if record_cursor != len(record_steps):
            raise RuntimeError(
                "Saveper schedule mismatch: not all record slots were filled. "
                "Check that saveper aligns with the model's time step."
            )

        self._overrides = {}
        finalized: Dict[str, Tensor] = {}
        for name, tensor in outputs.items():
            if tensor is None:
                continue
            finalized[name] = tensor.transpose(0, 1).contiguous()
        return finalized

    def _simulate_indexed(
        self,
        overrides: Dict[str, Tensor],
        track_set: set[str],
        batch: int,
    ) -> Dict[str, Tensor]:
        if self._eval_globals is None:
            raise RuntimeError("Runtime globals not initialised.")
        if self._dt_tensor is None:
            raise RuntimeError("Time step tensor is not initialised.")
        if not self._env_index:
            self._build_env_layout()
        if self._flow_indexed_block is None:
            self._flow_indexed_block = self._build_indexed_flow_block()

        aux_block_run_static = self._build_indexed_aux_block(
            self._aux_order_run_static,
            filename="<pysdtorch_indexed_aux_run_static>",
            func_name="__pysdtorch_indexed_aux_run_static",
            stage="run",
            override_aux_names=self._override_aux_names,
        )
        aux_block_run_dynamic = self._build_indexed_aux_block(
            self._aux_order_run_dynamic,
            filename="<pysdtorch_indexed_aux_run_dynamic>",
            func_name="__pysdtorch_indexed_aux_run_dynamic",
            stage="run",
            override_aux_names=self._override_aux_names,
        )
        if self._fused_eval:
            self._aux_fused_block_init = self._build_fused_aux_block(
                self._aux_order_init_run,
                filename="<pysdtorch_fused_aux_init>",
                override_aux_names=self._override_aux_names,
            )

        self._initial_cache = {}
        context = {**self._static_controls(batch), **overrides}
        context[_STAGE_FLAG] = True
        time_tensor = self._time_for_step(step_idx=0, batch=batch)
        context["time"] = time_tensor
        self._time_tensor = time_tensor

        rng = _RandomStream(batch, self.dtype, self.device, self.config.rng_seed)
        self._eval_globals["random_normal"] = rng.random_normal  # type: ignore[index]
        self._eval_globals["random_uniform"] = rng.random_uniform  # type: ignore[index]
        self._eval_globals["random_poisson"] = rng.random_poisson  # type: ignore[index]
        self._eval_globals["random_gamma"] = rng.random_gamma  # type: ignore[index]
        self._eval_globals["random_negative_binomial"] = rng.random_negative_binomial  # type: ignore[index]

        # Initialise stocks (dict-based init stage to preserve dependency handling).
        self._initialising_stocks.clear()
        self._evaluating_aux.clear()
        for name in self.model.stocks.keys():
            self._initialise_stock(name, context)
        self._initialising_stocks.clear()

        # Populate INITIAL caches and any initialization-stage auxiliaries.
        self._evaluate_auxiliaries_init(context)

        env: list[object] = [None] * len(self._env_names)
        for name, tensor in self._static_controls(batch).items():
            env[self._env_index[name]] = tensor
        for name, tensor in overrides.items():
            if name in self._env_index:
                env[self._env_index[name]] = tensor
        env[self._env_stage_idx] = False
        env[self._env_time_idx] = time_tensor
        for name in self.model.stocks.keys():
            env[self._env_index[name]] = context[name]

        def exec_aux(block: _IndexedFunctionBlock, label: str) -> None:
            self._allocate_available_cache.clear()
            previous_stage = self._eval_stage
            self._eval_stage = "run"
            try:
                block.func(env)
            except Exception as exc:  # pragma: no cover - diagnostics
                failing = self._resolve_indexed_exception_name(exc, block)
                if failing is None:
                    raise RuntimeError(f"Error evaluating indexed {label}: {exc}") from exc
                expr = self.model.auxiliaries[failing].expression.source
                raise RuntimeError(
                    f"Error evaluating auxiliary '{failing}': {exc} (expr: {expr})"
                ) from exc
            finally:
                self._eval_stage = previous_stage

        exec_aux(aux_block_run_static, label="static auxiliaries")
        exec_aux(aux_block_run_dynamic, label="auxiliaries")

        record_steps = self._compute_record_steps()
        record_capacity = len(record_steps)
        outputs = self._init_outputs(track_set, batch, record_capacity)
        track_indices: Dict[str, int] = {}
        for name in track_set:
            try:
                track_indices[name] = self._env_index[name]
            except KeyError as exc:
                raise KeyError(
                    f"Tracked variable '{name}' not found in indexed environment."
                ) from exc

        record_cursor = self._record_if_scheduled_env(
            outputs,
            track_set,
            env,
            track_indices=track_indices,
            step_idx=0,
            record_cursor=0,
            record_steps=record_steps,
            record_capacity=record_capacity,
        )

        steps = self.config.steps
        for step_idx in range(1, steps):
            self._allocate_available_cache.clear()
            block = self._flow_indexed_block
            if block is None:  # pragma: no cover - defensive
                raise RuntimeError("Indexed flow block not initialised.")
            try:
                block.func(env, self._dt_tensor)
            except Exception as exc:  # pragma: no cover - diagnostics
                failing = self._resolve_indexed_exception_name(exc, block)
                if failing is None:
                    raise RuntimeError(f"Error evaluating indexed flows: {exc}") from exc
                expr = self.model.stocks[failing].flow.source
                raise RuntimeError(
                    f"Error evaluating flow for stock '{failing}': {exc} (expr: {expr})"
                ) from exc

            time_tensor = self._time_for_step(step_idx=step_idx, batch=batch)
            env[self._env_time_idx] = time_tensor
            self._time_tensor = time_tensor
            exec_aux(aux_block_run_dynamic, label="auxiliaries")
            record_cursor = self._record_if_scheduled_env(
                outputs,
                track_set,
                env,
                track_indices=track_indices,
                step_idx=step_idx,
                record_cursor=record_cursor,
                record_steps=record_steps,
                record_capacity=record_capacity,
            )

        if record_cursor != len(record_steps):
            raise RuntimeError(
                "Saveper schedule mismatch: not all record slots were filled. "
                "Check that saveper aligns with the model's time step."
            )

        self._overrides = {}
        finalized: Dict[str, Tensor] = {}
        for name, tensor in outputs.items():
            if tensor is None:
                continue
            finalized[name] = tensor.transpose(0, 1).contiguous()
        return finalized

    def sample_parameters(
        self,
        bounds: Mapping[str, tuple[float, float]],
        n_draws: int,
    ) -> Dict[str, Tensor]:
        normalized = normalize_bounds(bounds)
        generator = torch.Generator(device=self.device)
        if self.config.rng_seed is not None:
            generator.manual_seed(self.config.rng_seed)

        samples: Dict[str, Tensor] = {}
        for name, (lo, hi) in normalized.bounds.items():
            width = hi - lo
            sample = torch.rand(
                (n_draws,),
                dtype=self.dtype,
                device=self.device,
                generator=generator,
            )
            samples[name] = lo + width * sample
        return samples

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _build_eval_env(self) -> Dict[str, object]:
        return {
            "__builtins__": {},
            _FUSED_ENSURE_TENSOR: self._ensure_tensor_fused,
            "torch": torch,
            "t_max": self._t_max,
            "t_min": self._t_min,
            "t_abs": self._t_abs,
            "t_exp": self._t_exp,
            "t_log": self._t_log,
            "t_sqrt": self._t_sqrt,
            "t_sin": self._t_sin,
            "t_if_then_else": self._t_if_then_else,
            "t_zidz": self._t_zidz,
            "t_xidz": self._t_xidz,
            "t_all": self._t_all,
            "t_any": self._t_any,
            "t_not": self._t_not,
            "t_sum": self._t_sum,
            "t_vmin": self._t_vmin,
            "t_vmax": self._t_vmax,
            "t_step": self._t_step,
            "t_ramp": self._t_ramp,
            "t_pulse": self._t_pulse,
            "t_initial": self._t_initial,
            "t_initial_set": self._t_initial_set,
            "t_initial_get": self._t_initial_get,
            "t_int": self._t_int,
            "t_delay_n": self._t_delay_n,
            "t_allocate_available": self._t_allocate_available,
            "random_normal": lambda *args, **kwargs: None,  # placeholder
            "random_uniform": lambda *args, **kwargs: None,  # placeholder
            "random_poisson": lambda *args, **kwargs: None,  # placeholder
            "random_gamma": lambda *args, **kwargs: None,  # placeholder
            "random_negative_binomial": lambda *args, **kwargs: None,  # placeholder
            "vector_elm_map": self._vector_elm_map,
        }

    def _register_lookups(self) -> None:
        if self._eval_globals is None:
            raise RuntimeError("Runtime globals not initialised.")
        self._lookup_tables = {}
        for name, lookup in self.model.lookups.items():
            self._lookup_tables[name] = _LookupTable(name, lookup.x, lookup.y, self)
        self._eval_globals.update(self._lookup_tables)

    def _ensure_tensor_fused(self, value) -> Tensor:
        return self._ensure_tensor(value, self._batch)

    def _flow_var_name(self, stock_name: str) -> str:
        return f"{_FUSED_FLOW_PREFIX}{stock_name}"

    def _build_env_layout(self) -> None:
        controls = ["time_step", "initial_time", "final_time", "saveper"]
        names: list[str] = [
            *controls,
            _STAGE_FLAG,
            "time",
            *self.model.stocks.keys(),
            *self.model.auxiliaries.keys(),
        ]
        seen: set[str] = set()
        collisions: set[str] = set()
        for name in names:
            if name in seen:
                collisions.add(name)
            seen.add(name)
        if collisions:
            raise RuntimeError(
                "Cannot build indexed evaluation environment due to variable name collisions: "
                + ", ".join(sorted(collisions))
            )
        self._env_names = names
        self._env_index = {name: idx for idx, name in enumerate(names)}
        self._env_time_idx = self._env_index["time"]
        self._env_stage_idx = self._env_index[_STAGE_FLAG]
        self._indexed_expr_cache.clear()

    def _to_indexed_expr(self, expr: str, stage: str) -> str:
        if not self._env_index:
            raise RuntimeError("Indexed environment layout not initialised.")
        cache_key = (stage, expr)
        cached = self._indexed_expr_cache.get(cache_key)
        if cached is not None:
            return cached
        root = ast.parse(expr, mode="eval")
        transformer = _EnvIndexTransformer(self._env_index, stage=stage)
        rewritten = transformer.visit(root)
        ast.fix_missing_locations(rewritten)
        if not isinstance(rewritten, ast.Expression):  # pragma: no cover - defensive
            raise RuntimeError("Unexpected AST root after rewriting.")
        result = ast.unparse(rewritten.body)
        self._indexed_expr_cache[cache_key] = result
        return result

    def _build_indexed_aux_block(
        self,
        order: Sequence[str],
        filename: str,
        func_name: str,
        stage: str,
        override_aux_names: frozenset[str],
    ) -> _IndexedFunctionBlock:
        if self._eval_globals is None:
            raise RuntimeError("Runtime globals not initialised.")
        lines: list[str] = []
        line_to_name: Dict[int, str] = {}
        line_no = 2  # first line in function body
        for name in order:
            if name in override_aux_names:
                continue
            expr = self.model.auxiliaries[name].expression.source
            expr_indexed = self._to_indexed_expr(expr, stage=stage)
            idx = self._env_index[name]
            lines.append(f"env[{idx}] = ensure(({expr_indexed}))")
            line_to_name[line_no] = name
            line_no += 1
        body = "\n".join(lines) if lines else "pass"
        indented = "\n".join(f"    {line}" for line in body.splitlines())
        source = f"def {func_name}(env, ensure={_FUSED_ENSURE_TENSOR}):\n{indented}\n"
        code = compile(source, filename, "exec")
        namespace: dict[str, object] = {}
        exec(code, self._eval_globals, namespace)
        func = namespace[func_name]
        if not callable(func):  # pragma: no cover - defensive
            raise RuntimeError("Indexed auxiliary block did not produce a callable.")
        return _IndexedFunctionBlock(filename=filename, func=func, line_to_name=line_to_name)

    def _build_indexed_flow_block(self) -> _IndexedFunctionBlock:
        if self._eval_globals is None:
            raise RuntimeError("Runtime globals not initialised.")
        filename = "<pysdtorch_indexed_flows>"
        func_name = "__pysdtorch_indexed_flow_update"
        lines: list[str] = []
        line_to_name: Dict[int, str] = {}
        line_no = 2

        flow_vars: list[tuple[str, str]] = []
        for idx, (name, stock) in enumerate(self.model.stocks.items()):
            expr_indexed = self._to_indexed_expr(stock.flow.source, stage="run")
            var = f"flow_{idx}"
            flow_vars.append((name, var))
            lines.append(f"{var} = ensure(({expr_indexed}))")
            line_to_name[line_no] = name
            line_no += 1

        dt_name = "dt"
        for name, var in flow_vars:
            stock_idx = self._env_index[name]
            if self.model.stocks[name].non_negative:
                lines.append(
                    f"env[{stock_idx}] = torch.clamp(env[{stock_idx}] + {var} * {dt_name}, min=0.0)"
                )
            else:
                lines.append(f"env[{stock_idx}] = env[{stock_idx}] + {var} * {dt_name}")
            line_to_name[line_no] = name
            line_no += 1

        body = "\n".join(lines) if lines else "pass"
        indented = "\n".join(f"    {line}" for line in body.splitlines())
        source = (
            f"def {func_name}(env, dt, ensure={_FUSED_ENSURE_TENSOR}):\n{indented}\n"
        )
        code = compile(source, filename, "exec")
        namespace: dict[str, object] = {}
        exec(code, self._eval_globals, namespace)
        func = namespace[func_name]
        if not callable(func):  # pragma: no cover - defensive
            raise RuntimeError("Indexed flow block did not produce a callable.")
        return _IndexedFunctionBlock(filename=filename, func=func, line_to_name=line_to_name)

    def _resolve_indexed_exception_name(
        self, exc: BaseException, block: _IndexedFunctionBlock
    ) -> str | None:
        tb = exc.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == block.filename:
                return block.line_to_name.get(tb.tb_lineno)
            tb = tb.tb_next
        return None

    @staticmethod
    def _collect_aux_closure(
        start: Iterable[str],
        deps_map: Dict[str, tuple[str, ...]],
    ) -> frozenset[str]:
        required: set[str] = set()
        stack = list(start)
        while stack:
            name = stack.pop()
            if name in required:
                continue
            required.add(name)
            stack.extend(deps_map.get(name, ()))
        return frozenset(required)

    @staticmethod
    def _expr_uses_dynamic_builtins(expr: str, stage: str) -> bool:
        if stage not in {"run", "init"}:
            raise ValueError(f"Unknown stage '{stage}'")
        try:
            root = ast.parse(expr, mode="eval")
        except SyntaxError:
            return True

        def walk(node: ast.AST) -> bool:
            if isinstance(node, ast.Name):
                return node.id == "time"

            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id in _DYNAMIC_AUX_FUNCTIONS:
                    return True

            if isinstance(node, ast.IfExp):
                # Stage-guarded branches are lazy: only one side is active.
                if isinstance(node.test, ast.Name) and node.test.id == _STAGE_FLAG:
                    branch = node.body if stage == "init" else node.orelse
                    return walk(branch)

                # Unknown conditionals: conservatively include everything.
                if walk(node.test) or walk(node.body) or walk(node.orelse):
                    return True
                return False

            for child in ast.iter_child_nodes(node):
                if walk(child):
                    return True
            return False

        return walk(root.body)

    def _compute_static_auxiliaries_run(self, aux_vars) -> frozenset[str]:
        intrinsic_dynamic: set[str] = set()
        for aux in aux_vars:
            expr = aux.expression
            deps = set(expr.dependencies)
            if deps.intersection(self._stock_names):
                intrinsic_dynamic.add(aux.name)
                continue
            if self._expr_uses_dynamic_builtins(expr.source, stage="run"):
                intrinsic_dynamic.add(aux.name)

        dynamic = set(intrinsic_dynamic)
        remaining = set(self._aux_names).difference(dynamic)
        changed = True
        while changed:
            changed = False
            for name in list(remaining):
                deps = set(self._aux_deps_run.get(name, ()))
                if deps.intersection(dynamic):
                    dynamic.add(name)
                    remaining.remove(name)
                    changed = True
        static = set(self._aux_names).difference(dynamic)
        return frozenset(static)

    def _build_fused_aux_block(
        self,
        order: Sequence[str],
        filename: str,
        override_aux_names: frozenset[str],
    ) -> _FusedExecBlock:
        lines: list[str] = []
        line_to_name: Dict[int, str] = {}
        line_no = 1
        for name in order:
            if name in override_aux_names:
                continue
            expr = self.model.auxiliaries[name].expression.source
            lines.append(f"{name} = {_FUSED_ENSURE_TENSOR}(({expr}))")
            line_to_name[line_no] = name
            line_no += 1
        source = "\n".join(lines) if lines else "pass"
        code = compile(source, filename, "exec")
        return _FusedExecBlock(filename=filename, code=code, line_to_name=line_to_name)

    def _build_fused_flow_block(self) -> _FusedExecBlock:
        filename = "<pysdtorch_fused_flows>"
        lines: list[str] = []
        line_to_name: Dict[int, str] = {}
        line_no = 1
        for name, stock in self.model.stocks.items():
            expr = stock.flow.source
            flow_name = self._flow_var_name(name)
            lines.append(f"{flow_name} = {_FUSED_ENSURE_TENSOR}(({expr}))")
            line_to_name[line_no] = name
            line_no += 1
        source = "\n".join(lines) if lines else "pass"
        code = compile(source, filename, "exec")
        return _FusedExecBlock(filename=filename, code=code, line_to_name=line_to_name)

    def _resolve_fused_exception_name(
        self, exc: BaseException, block: _FusedExecBlock
    ) -> str | None:
        tb = exc.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == block.filename:
                return block.line_to_name.get(tb.tb_lineno)
            tb = tb.tb_next
        return None

    def _exec_fused_flows(self, context: Dict[str, Tensor]) -> None:
        if self._eval_globals is None:
            raise RuntimeError("Runtime globals not initialised.")
        if self._flow_fused_block is None:
            raise RuntimeError("Fused flow block not initialised.")
        try:
            exec(self._flow_fused_block.code, self._eval_globals, context)
        except Exception as exc:  # pragma: no cover - diagnostics
            name = self._resolve_fused_exception_name(exc, self._flow_fused_block)
            if name is None:
                raise RuntimeError(f"Error evaluating fused flows: {exc}") from exc
            expr = self.model.stocks[name].flow.source
            raise RuntimeError(
                f"Error evaluating flow for stock '{name}': {exc} (expr: {expr})"
            ) from exc

    def _evaluate_auxiliaries_static(
        self,
        context: Dict[str, Tensor],
    ) -> None:
        if not self._aux_order_run_static:
            return
        previous_stage = self._eval_stage
        self._eval_stage = "run"
        try:
            self._allocate_available_cache.clear()
            if self._fused_eval:
                for name in self._override_aux_names:
                    context[name] = self._overrides[name]
                if self._aux_fused_block_run_static is None:
                    raise RuntimeError("Fused static auxiliary block not initialised.")
                block = self._aux_fused_block_run_static
                try:
                    exec(block.code, self._eval_globals, context)
                except Exception as exc:  # pragma: no cover - diagnostics
                    failing = self._resolve_fused_exception_name(exc, block)
                    if failing is None:
                        raise RuntimeError(
                            f"Error evaluating fused static auxiliaries: {exc}"
                        ) from exc
                    expr = self.model.auxiliaries[failing].expression.source
                    raise RuntimeError(
                        f"Error evaluating auxiliary '{failing}': {exc} (expr: {expr})"
                    ) from exc
                return
            for name in self._aux_order_run_static:
                if name in self._overrides:
                    context[name] = self._overrides[name]
                    continue
                self._compute_auxiliary(name, context)
        finally:
            self._eval_stage = previous_stage

    def _evaluate_auxiliaries(
        self,
        context: Dict[str, Tensor],
    ) -> None:
        previous_stage = self._eval_stage
        self._eval_stage = "run"
        try:
            self._allocate_available_cache.clear()
            if self._fused_eval:
                for name in self._override_aux_names:
                    context[name] = self._overrides[name]
                if self._aux_fused_block_run_dynamic is None:
                    raise RuntimeError("Fused auxiliary block not initialised.")
                block = self._aux_fused_block_run_dynamic
                try:
                    exec(block.code, self._eval_globals, context)
                except Exception as exc:  # pragma: no cover - diagnostics
                    failing = self._resolve_fused_exception_name(exc, block)
                    if failing is None:
                        raise RuntimeError(f"Error evaluating fused auxiliaries: {exc}") from exc
                    expr = self.model.auxiliaries[failing].expression.source
                    raise RuntimeError(
                        f"Error evaluating auxiliary '{failing}': {exc} (expr: {expr})"
                    ) from exc
                return
            for name in self._aux_order_run_dynamic:
                if name in self._overrides:
                    context[name] = self._overrides[name]
                    continue
                self._compute_auxiliary(name, context)
        finally:
            self._eval_stage = previous_stage

    def _evaluate_auxiliaries_init(
        self,
        context: Dict[str, Tensor],
    ) -> None:
        previous_stage = self._eval_stage
        self._eval_stage = "init"
        try:
            self._allocate_available_cache.clear()
            if self._fused_eval:
                for name in self._override_aux_names:
                    context[name] = self._overrides[name]
                if self._aux_fused_block_init is None:
                    raise RuntimeError("Fused auxiliary init block not initialised.")
                block = self._aux_fused_block_init
                try:
                    exec(block.code, self._eval_globals, context)
                except Exception as exc:  # pragma: no cover - diagnostics
                    failing = self._resolve_fused_exception_name(exc, block)
                    if failing is None:
                        raise RuntimeError(
                            f"Error evaluating fused auxiliaries (init stage): {exc}"
                        ) from exc
                    expr = self.model.auxiliaries[failing].expression.source
                    raise RuntimeError(
                        f"Error evaluating auxiliary '{failing}': {exc} (expr: {expr})"
                    ) from exc
                return
            for name in self._aux_order_init_run:
                if name in self._overrides:
                    context[name] = self._overrides[name]
                    continue
                self._compute_auxiliary(name, context)
        finally:
            self._eval_stage = previous_stage

    def _ensure_auxiliary(self, name: str, context: Dict[str, Tensor]) -> Tensor:
        if name in context:
            return context[name]
        return self._compute_auxiliary(name, context)

    def _compute_auxiliary(self, name: str, context: Dict[str, Tensor]) -> Tensor:
        if name in self._overrides:
            value = self._overrides[name]
            context[name] = value
            return value
        if name in self._evaluating_aux:
            raise RuntimeError(f"Cyclic auxiliary dependency detected for '{name}'.")
        self._evaluating_aux.add(name)
        expr = self.model.auxiliaries[name].expression
        try:
            value = self._evaluate_expression(expr, context)
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            raise RuntimeError(
                f"Error evaluating auxiliary '{name}': {exc} (expr: {expr.source})"
            ) from exc
        context[name] = value
        self._evaluating_aux.remove(name)
        return value

    def _record(
        self,
        outputs: Dict[str, Tensor | None],
        track_set: set[str],
        context: Dict[str, Tensor],
        record_idx: int,
        record_capacity: int,
    ) -> None:
        for name in track_set:
            if name not in context:
                raise KeyError(f"Tracked variable '{name}' not found in context.")
            tensor = context[name]
            if outputs[name] is None:
                shape = (record_capacity, *tensor.shape)
                outputs[name] = torch.empty(shape, dtype=self.dtype, device=self.device)
            outputs[name][record_idx] = tensor

    def _record_if_scheduled(
        self,
        outputs: Dict[str, Tensor | None],
        track_set: set[str],
        context: Dict[str, Tensor],
        step_idx: int,
        record_cursor: int,
        record_steps: Sequence[int],
        record_capacity: int,
    ) -> int:
        if record_cursor >= len(record_steps):
            return record_cursor
        if step_idx != record_steps[record_cursor]:
            return record_cursor
        self._record(
            outputs,
            track_set,
            context,
            record_idx=record_cursor,
            record_capacity=record_capacity,
        )
        return record_cursor + 1

    def _record_env(
        self,
        outputs: Dict[str, Tensor | None],
        track_set: set[str],
        env: Sequence[object],
        track_indices: Mapping[str, int],
        record_idx: int,
        record_capacity: int,
    ) -> None:
        for name in track_set:
            idx = track_indices[name]
            value = env[idx]
            if not isinstance(value, Tensor):
                raise KeyError(f"Tracked variable '{name}' not found in environment.")
            tensor = value
            if outputs[name] is None:
                shape = (record_capacity, *tensor.shape)
                outputs[name] = torch.empty(shape, dtype=self.dtype, device=self.device)
            outputs[name][record_idx] = tensor

    def _record_if_scheduled_env(
        self,
        outputs: Dict[str, Tensor | None],
        track_set: set[str],
        env: Sequence[object],
        track_indices: Mapping[str, int],
        step_idx: int,
        record_cursor: int,
        record_steps: Sequence[int],
        record_capacity: int,
    ) -> int:
        if record_cursor >= len(record_steps):
            return record_cursor
        if step_idx != record_steps[record_cursor]:
            return record_cursor
        self._record_env(
            outputs,
            track_set,
            env,
            track_indices=track_indices,
            record_idx=record_cursor,
            record_capacity=record_capacity,
        )
        return record_cursor + 1

    def _init_outputs(
        self, track_set: set[str], batch: int, num_records: int
    ) -> Dict[str, Tensor | None]:
        return {name: None for name in track_set}

    def _compute_record_steps(self) -> list[int]:
        saveper = (
            self.config.saveper
            if self.config.saveper is not None
            else self.config.time_step
        )
        if saveper <= 0:
            raise ValueError("saveper must be positive.")
        tolerance = self.config.time_step * _SAVE_RELATIVE_PRECISION
        record_steps: list[int] = []
        for step_idx in range(self.config.steps):
            time_value = self.config.initial_time + step_idx * self.config.time_step
            if self._should_record_time(time_value, saveper, tolerance):
                record_steps.append(step_idx)
        if not record_steps:
            record_steps.append(0)
        return record_steps

    def _should_record_time(self, time_value: float, saveper: float, tolerance: float) -> bool:
        time_delay = time_value - self.config.initial_time
        remainder = time_delay % saveper
        if remainder < tolerance:
            return True
        distance_to_next = (-time_delay) % saveper
        return distance_to_next < tolerance

    def _initialise_stock(
        self,
        name: str,
        context: Dict[str, Tensor],
    ) -> Tensor:
        if name in context:
            return context[name]
        if name not in self.model.stocks:
            raise KeyError(f"Unknown stock '{name}' referenced during initialisation.")
        if name in self._initialising_stocks:
            raise RuntimeError(f"Cyclic stock dependency detected for '{name}'.")
        self._initialising_stocks.add(name)
        stock = self.model.stocks[name]
        try:
            value = self._evaluate_expression(stock.initial, context)
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            raise RuntimeError(
                f"Error evaluating stock initial '{name}': {exc} (expr: {stock.initial.source})"
            ) from exc
        context[name] = value
        self._initialising_stocks.remove(name)
        return value

    def _resolve_dependencies(
        self,
        dependencies: Sequence[str],
        context: Dict[str, Tensor],
    ) -> None:
        for dep in dependencies:
            if dep in context:
                continue
            if dep in self._stock_names:
                self._initialise_stock(dep, context)
            elif dep in self._aux_names:
                self._ensure_auxiliary(dep, context)
            elif dep in self._overrides:
                context[dep] = self._overrides[dep]

    def _prepare_parameters(
        self,
        parameters: Mapping[str, object],
        batch: int,
    ) -> Dict[str, Tensor]:
        overrides: Dict[str, Tensor] = {}
        for raw_name, value in parameters.items():
            name = canonical_name(raw_name)
            overrides[name] = self._ensure_tensor(value, batch)
        return overrides

    def _evaluate_expression(
        self, expr_spec, context: Dict[str, Tensor]
    ) -> Tensor:
        if self._eval_globals is None:
            raise RuntimeError("Runtime globals not initialised.")
        deps = (
            expr_spec.init_dependencies
            if context.get(_STAGE_FLAG, False)
            else expr_spec.dependencies
        )
        self._resolve_dependencies(deps, context)
        stage = "init" if context.get(_STAGE_FLAG, False) else "run"
        previous_stage = self._eval_stage
        self._eval_stage = stage
        try:
            result = eval(expr_spec.compiled, self._eval_globals, context)
        finally:
            self._eval_stage = previous_stage
        return self._ensure_tensor(result, self._batch)

    def _static_controls(self, batch: int) -> Dict[str, Tensor]:
        return {
            "time_step": self._scalar_tensor(self.config.time_step, batch),
            "initial_time": self._scalar_tensor(self.config.initial_time, batch),
            "final_time": self._scalar_tensor(self.config.final_time, batch),
            "saveper": self._scalar_tensor(self.config.saveper, batch),
        }

    def _time_for_step(self, step_idx: int, batch: int) -> Tensor:
        if not self._precompute_time_grid:
            time_value = self.config.initial_time + step_idx * self.config.time_step
            return self._scalar_tensor(time_value, batch)

        steps = self.config.steps
        grid = self._time_grid
        if (
            grid is None
            or grid.shape[0] != steps
            or grid.device != self.device
            or grid.dtype != self.dtype
        ):
            values = [
                self.config.initial_time + idx * self.config.time_step
                for idx in range(steps)
            ]
            grid = torch.tensor(values, dtype=self.dtype, device=self.device)
            self._time_grid = grid

        return grid[step_idx].expand(batch)

    def _scalar_tensor(self, value: float, batch: int) -> Tensor:
        tensor = torch.full(
            (batch,), float(value), dtype=self.dtype, device=self.device
        )
        return tensor

    def _ensure_tensor(self, value, batch: int) -> Tensor:
        tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if tensor.ndim == 0:
            tensor = tensor.expand(batch)
        elif tensor.shape[0] != batch:
            if tensor.shape[0] == 1:
                tensor = tensor.expand(batch)
            else:
                raise ValueError(
                    f"Tensor with shape {tuple(tensor.shape)} cannot broadcast to batch {batch}"
                )
        return tensor

    def _resolve_batch(
        self,
        parameters: Mapping[str, "TensorLike"],
        n_draws: int,
    ) -> int:
        batch = int(n_draws) if n_draws else 1
        for value in parameters.values():
            tensor = torch.as_tensor(value)
            if tensor.ndim == 0:
                continue
            size = tensor.shape[0]
            if batch == 1:
                batch = size
            elif size not in (1, batch):
                raise ValueError(
                    f"Parameter batch mismatch: expected {batch}, got {size}"
                )
        return batch

    def _resolve_dtype(self, dtype) -> torch.dtype:
        if isinstance(dtype, torch.dtype):
            return dtype
        target = str(dtype)
        if target.startswith("torch."):
            target = target.split(".", 1)[1]
        attr = getattr(torch, target, None)
        if attr is None:
            raise ValueError(f"Unsupported torch dtype '{dtype}'")
        return attr

    def _assert_compiled(self) -> None:
        if not self._compiled:
            raise RuntimeError("Runtime needs compile() before simulate().")

    # ------------------------------------------------------------------ #
    # Torch helpers                                                      #
    # ------------------------------------------------------------------ #

    def _t_max(self, *args):
        tensors = [self._ensure_tensor(arg, self._batch) for arg in args]
        result = tensors[0]
        for tensor in tensors[1:]:
            result = torch.maximum(result, tensor)
        return result

    def _t_min(self, *args):
        tensors = [self._ensure_tensor(arg, self._batch) for arg in args]
        result = tensors[0]
        for tensor in tensors[1:]:
            result = torch.minimum(result, tensor)
        return result

    def _t_abs(self, value):
        return torch.abs(self._ensure_tensor(value, self._batch))

    def _t_exp(self, value):
        return torch.exp(self._ensure_tensor(value, self._batch))

    def _t_log(self, value):
        return torch.log(self._ensure_tensor(value, self._batch))

    def _t_sqrt(self, value):
        return torch.sqrt(self._ensure_tensor(value, self._batch).clamp_min(0.0))

    def _t_sin(self, value):
        return torch.sin(self._ensure_tensor(value, self._batch))

    def _t_step(self, value, tstep):
        if self._time_tensor is None or self._dt_tensor is None:
            raise RuntimeError("STEP requires an active simulation context.")
        value_t = self._ensure_tensor(value, self._batch)
        tstep_t = self._ensure_tensor(tstep, self._batch)
        active = self._time_tensor + (self._dt_tensor * 0.5) > tstep_t
        return value_t * active.to(dtype=self.dtype)

    def _t_ramp(self, slope, start, finish=None):
        if self._time_tensor is None:
            raise RuntimeError("RAMP requires an active simulation context.")
        slope_t = self._ensure_tensor(slope, self._batch)
        start_t = self._ensure_tensor(start, self._batch)
        if finish is None:
            final = self._time_tensor
        else:
            finish_t = self._ensure_tensor(finish, self._batch)
            final = torch.minimum(finish_t, self._time_tensor)
        active = self._time_tensor + _SMALL_VENSIM > start_t
        return active.to(dtype=self.dtype) * slope_t * (final - start_t)

    def _t_pulse(self, start, width=None):
        if self._time_tensor is None or self._dt_tensor is None:
            raise RuntimeError("PULSE requires an active simulation context.")
        t = self._time_tensor
        start_t = self._ensure_tensor(start, self._batch)
        if width is None:
            width_t = self._dt_tensor * 0.5
        else:
            width_t = torch.clamp(self._ensure_tensor(width, self._batch), min=0.0)
        active = (start_t - _SMALL_VENSIM <= t) & (t < start_t + width_t)
        return active.to(dtype=self.dtype)

    def _t_if_then_else(self, condition, true_value, false_value):
        mask = self._ensure_condition_tensor(condition)
        true_tensor = self._ensure_tensor(true_value, self._batch)
        false_tensor = self._ensure_tensor(false_value, self._batch)
        return torch.where(mask, true_tensor, false_tensor)

    def _t_zidz(self, numerator, denominator):
        numerator = self._ensure_tensor(numerator, self._batch)
        denominator = self._ensure_tensor(denominator, self._batch)
        mask = torch.abs(denominator) >= _SMALL_VENSIM
        safe_denominator = denominator.masked_fill(~mask, 1.0)
        ratio = numerator / safe_denominator
        return ratio * mask.to(dtype=self.dtype)

    def _t_xidz(self, numerator, denominator, x):
        numerator = self._ensure_tensor(numerator, self._batch)
        denominator = self._ensure_tensor(denominator, self._batch)
        fallback = self._ensure_tensor(x, self._batch)
        is_small = torch.abs(denominator) < _SMALL_VENSIM
        safe_denominator = denominator.masked_fill(is_small, 1.0)
        ratio = numerator / safe_denominator
        return torch.where(is_small, fallback, ratio)

    def _t_all(self, *args):
        if not args:
            raise ValueError("t_all requires at least one argument.")
        result = self._ensure_condition_tensor(args[0])
        for arg in args[1:]:
            result = result & self._ensure_condition_tensor(arg)
        return result.to(dtype=self.dtype)

    def _t_any(self, *args):
        if not args:
            raise ValueError("t_any requires at least one argument.")
        result = self._ensure_condition_tensor(args[0])
        for arg in args[1:]:
            result = result | self._ensure_condition_tensor(arg)
        return result.to(dtype=self.dtype)

    def _t_not(self, value):
        result = ~self._ensure_condition_tensor(value)
        return result.to(dtype=self.dtype)

    def _t_sum(self, values):
        if isinstance(values, (list, tuple)):
            tensors = [self._ensure_tensor(val, self._batch) for val in values]
            if not tensors:
                raise ValueError("t_sum requires at least one argument.")
            stacked = torch.stack(tensors, dim=0)
            return torch.sum(stacked, dim=0)
        return self._ensure_tensor(values, self._batch)

    def _t_vmin(self, values):
        if isinstance(values, (list, tuple)):
            tensors = [self._ensure_tensor(val, self._batch) for val in values]
            if not tensors:
                raise ValueError("t_vmin requires at least one argument.")
            stacked = torch.stack(tensors, dim=0)
            return torch.min(stacked, dim=0).values
        return self._ensure_tensor(values, self._batch)

    def _t_vmax(self, values):
        if isinstance(values, (list, tuple)):
            tensors = [self._ensure_tensor(val, self._batch) for val in values]
            if not tensors:
                raise ValueError("t_vmax requires at least one argument.")
            stacked = torch.stack(tensors, dim=0)
            return torch.max(stacked, dim=0).values
        return self._ensure_tensor(values, self._batch)

    def _t_initial(self, value):
        key = id(value)
        if key not in self._initial_cache:
            self._initial_cache[key] = self._ensure_tensor(value, self._batch)
        return self._initial_cache[key]

    def _t_initial_set(self, identifier: str, value):
        tensor = self._ensure_tensor(value, self._batch)
        self._initial_cache[identifier] = tensor
        return tensor

    def _t_initial_get(self, identifier: str):
        try:
            return self._initial_cache[identifier]
        except KeyError as exc:  # pragma: no cover - diagnostics
            raise RuntimeError(
                f"INITIAL cache miss for '{identifier}'. Ensure INITIAL() is evaluated during initialization."
            ) from exc

    def _t_int(self, value):
        tensor = self._ensure_tensor(value, self._batch)
        return torch.floor(tensor)

    def _t_delay_n(self, identifier, inflow, delay_time, initial, order):
        if self._dt_tensor is None:
            raise RuntimeError("Time step tensor is not initialised.")
        batch = self._batch
        inflow_t = self._ensure_tensor(inflow, batch)
        delay_t = torch.maximum(
            self._ensure_tensor(delay_time, batch),
            torch.tensor(_SMALL_VENSIM, dtype=self.dtype, device=self.device),
        )
        initial_t = self._ensure_tensor(initial, batch)
        order_t = torch.maximum(
            self._ensure_tensor(order, batch),
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
        )
        requested_order_int = torch.clamp(torch.floor(order_t), min=1.0).to(torch.int64)
        # Vensim reduces DELAY N order if delay_time <= order * time_step.
        ratio = delay_t / self._dt_tensor
        eps = torch.finfo(self.dtype).eps
        max_order_by_dt = torch.floor(torch.clamp(ratio - eps, min=0.0))
        max_order_by_dt = torch.clamp(max_order_by_dt, min=1.0).to(torch.int64)
        effective_order_int = torch.minimum(requested_order_int, max_order_by_dt)
        requested_max_order = max(1, int(requested_order_int.max().item()))

        state = self._delay_n_state.get(identifier)
        times = self._delay_n_times.get(identifier)
        if (
            state is None
            or times is None
            or state.shape[0] != batch
            or times.shape[0] != batch
        ):
            state = (initial_t * delay_t).unsqueeze(1).expand(batch, requested_max_order).clone()
            times = delay_t.unsqueeze(1).expand(batch, requested_max_order).clone()
            self._delay_n_state[identifier] = state
            self._delay_n_times[identifier] = times
        else:
            state = state.to(dtype=self.dtype, device=self.device)
            times = times.to(dtype=self.dtype, device=self.device)
            if state.shape[1] < requested_max_order:
                extra = requested_max_order - state.shape[1]
                pad_state = state[:, -1:].expand(batch, extra).clone()
                pad_times = times[:, -1:].expand(batch, extra).clone()
                state = torch.cat([state, pad_state], dim=1)
                times = torch.cat([times, pad_times], dim=1)
                self._delay_n_state[identifier] = state
                self._delay_n_times[identifier] = times

        max_order = state.shape[1]
        gather_idx = (effective_order_int - 1).clamp(min=0, max=max_order - 1)
        # DELAY N output uses the previous-step delay time snapshot, not the current one.
        output = state.gather(1, gather_idx.unsqueeze(1)).squeeze(1) / times[:, 0]

        if self._eval_stage == "init":
            return output

        rolled_times = torch.roll(times, shifts=1, dims=1)
        rolled_times[:, 0] = delay_t
        outflows = state / torch.clamp(rolled_times, min=_SMALL_VENSIM)
        inflows = torch.roll(outflows, shifts=1, dims=1)
        inflows[:, 0] = inflow_t
        dstate = (inflows - outflows) * effective_order_int.to(self.dtype).unsqueeze(1)

        stage_idx = torch.arange(max_order, device=self.device).unsqueeze(0)
        active = stage_idx < effective_order_int.unsqueeze(1)
        updated_state = state + dstate * self._dt_tensor.unsqueeze(1)
        new_state = torch.where(active, updated_state, state)
        new_times = torch.where(active, rolled_times, times)
        self._delay_n_state[identifier] = new_state
        self._delay_n_times[identifier] = new_times
        return output

    def _t_allocate_available(self, identifier, request, pp, avail, index):
        batch = self._batch
        key = str(identifier)
        cached = self._allocate_available_cache.get(key)
        if cached is None:
            request_t = self._stack_1d_list(request, batch)
            pp_t = self._stack_pp_matrix(pp, batch)
            avail_t = self._ensure_tensor(avail, batch)
            allocation = self._allocate_available_impl(request_t, pp_t, avail_t)
            self._allocate_available_cache[key] = allocation
        else:
            allocation = cached

        idx = int(index)
        if idx < 0 or idx >= allocation.shape[1]:
            raise IndexError(
                f"ALLOCATE AVAILABLE index {idx} out of bounds for size {allocation.shape[1]}"
            )
        return allocation[:, idx]

    def _stack_1d_list(self, value, batch: int) -> Tensor:
        if isinstance(value, Tensor):
            tensor = value.to(dtype=self.dtype, device=self.device)
            if tensor.ndim == 1:
                if tensor.shape[0] == batch:
                    return tensor.unsqueeze(1)
                if tensor.shape[0] == 1:
                    return tensor.expand(batch).unsqueeze(1)
                return tensor.unsqueeze(0).expand(batch, -1)
            if tensor.ndim == 0:
                return tensor.expand(batch).unsqueeze(1)
            if tensor.ndim != 2:
                raise ValueError(
                    f"Expected request tensor with 2 dims (batch,n), got {tensor.shape}"
                )
            if tensor.shape[0] != batch:
                raise ValueError(
                    f"Request tensor batch mismatch: expected {batch}, got {tensor.shape[0]}"
                )
            return tensor

        if not isinstance(value, (list, tuple)):
            tensor = self._ensure_tensor(value, batch)
            return tensor.unsqueeze(1)

        tensors = [self._ensure_tensor(item, batch) for item in value]
        if not tensors:
            raise ValueError("ALLOCATE AVAILABLE requires at least one request value.")
        return torch.stack(tensors, dim=1)

    def _stack_pp_matrix(self, value, batch: int) -> Tensor:
        if isinstance(value, Tensor):
            tensor = value.to(dtype=self.dtype, device=self.device)
            if tensor.ndim != 3:
                raise ValueError(
                    f"Expected priority profile tensor with 3 dims (batch,n,4), got {tensor.shape}"
                )
            if tensor.shape[0] != batch:
                raise ValueError(
                    f"Priority profile batch mismatch: expected {batch}, got {tensor.shape[0]}"
                )
            if tensor.shape[2] < 4:
                raise ValueError(
                    f"Priority profile last dimension must be >= 4, got {tensor.shape[2]}"
                )
            return tensor[..., :4]

        if not isinstance(value, (list, tuple)):
            raise ValueError("Priority profile must be a matrix (list of rows).")

        rows = []
        for row in value:
            if not isinstance(row, (list, tuple)):
                raise ValueError("Priority profile rows must be lists.")
            if len(row) < 4:
                raise ValueError(
                    f"Priority profile rows must have at least 4 elements, got {len(row)}"
                )
            row_tensors = [self._ensure_tensor(item, batch) for item in row[:4]]
            rows.append(torch.stack(row_tensors, dim=1))

        if not rows:
            raise ValueError("Priority profile matrix cannot be empty.")

        stacked = torch.stack(rows, dim=1)  # (batch, n, 4)
        return stacked

    def _allocate_available_impl(self, request: Tensor, pp: Tensor, avail: Tensor) -> Tensor:
        """
        Torch implementation of Vensim's ALLOCATE AVAILABLE.

        Shapes:
        - request: (batch, n_targets)
        - pp: (batch, n_targets, 4) with [ptype, ppriority, pwidth, pextra]
        - avail: (batch,)
        """
        request = torch.clamp(request, min=0.0)
        avail = torch.clamp(avail, min=0.0)
        total_request = torch.sum(request, dim=1)
        target = torch.minimum(avail, total_request)

        full_fill = avail >= (total_request - _ALLOCATE_AVAILABLE_TARGET_TOL)
        none_fill = target <= _ALLOCATE_AVAILABLE_TARGET_TOL
        partial = ~(full_fill | none_fill)

        ptype = torch.round(pp[..., 0]).to(torch.int64)
        ppriority = pp[..., 1]
        pwidth = torch.clamp(pp[..., 2], min=_SMALL_VENSIM)

        ptype_min = int(ptype.min().item())
        ptype_max = int(ptype.max().item())
        if ptype_min == 3 and ptype_max == 3:
            k = 8.2923611
            lower = ppriority - k * pwidth
            upper = ppriority + k * pwidth

            low = lower.min(dim=1).values
            high = upper.max(dim=1).values
            span = torch.clamp(high - low, min=_SMALL_VENSIM)
            low = low - span
            high = high + span

            def allocation_at(x: Tensor) -> Tensor:
                x = x.unsqueeze(1)
                arg = (ppriority - x) / (1.41421356237 * pwidth)
                alloc = request * 0.5 * (1.0 + torch.erf(arg))
                alloc = torch.clamp(alloc, min=0.0)
                return torch.minimum(alloc, request)

        else:
            supported = (ptype == 1) | (ptype == 2) | (ptype == 3) | (ptype == 4)
            if not bool(torch.all(supported).item()):  # pragma: no cover - defensive
                bad = sorted({int(x) for x in ptype[~supported].unique().tolist()})
                raise NotImplementedError(
                    "ALLOCATE AVAILABLE priority profile types not supported: "
                    f"{bad}. Supported: 1,2,3,4."
                )

            k = torch.where(
                ptype == 3,
                torch.tensor(8.2923611, dtype=self.dtype, device=self.device),
                torch.tensor(0.5, dtype=self.dtype, device=self.device),
            )
            k = torch.where(
                ptype == 4,
                torch.tensor(36.7368005696, dtype=self.dtype, device=self.device),
                k,
            )
            k = torch.where((ptype == 1) | (ptype == 2), torch.tensor(0.5, dtype=self.dtype, device=self.device), k)

            lower = ppriority - k * pwidth
            upper = ppriority + k * pwidth
            low = lower.min(dim=1).values
            high = upper.max(dim=1).values
            span = torch.clamp(high - low, min=_SMALL_VENSIM)
            low = low - span
            high = high + span

            def allocation_at(x: Tensor) -> Tensor:
                x = x.unsqueeze(1)
                left = ppriority - 0.5 * pwidth
                right = ppriority + 0.5 * pwidth

                rect = torch.where(
                    x <= left,
                    request,
                    torch.where(
                        x < right,
                        request * (1.0 - (x - left) / pwidth),
                        torch.zeros_like(request),
                    ),
                )

                tri = torch.where(
                    x <= left,
                    request,
                    torch.where(
                        x < ppriority,
                        request * (1.0 - 2.0 * (x - left) ** 2 / (pwidth**2)),
                        torch.where(
                            x < right,
                            2.0 * request * (right - x) ** 2 / (pwidth**2),
                            torch.zeros_like(request),
                        ),
                    ),
                )

                arg = (ppriority - x) / (1.41421356237 * pwidth)
                normal = request * 0.5 * (1.0 + torch.erf(arg))

                exp_alloc = torch.where(
                    x < ppriority,
                    request * (1.0 - 0.5 * torch.exp((x - ppriority) / pwidth)),
                    request * 0.5 * torch.exp((ppriority - x) / pwidth),
                )

                alloc = torch.zeros_like(request)
                alloc = torch.where(ptype == 1, rect, alloc)
                alloc = torch.where(ptype == 2, tri, alloc)
                alloc = torch.where(ptype == 3, normal, alloc)
                alloc = torch.where(ptype == 4, exp_alloc, alloc)
                alloc = torch.clamp(alloc, min=0.0)
                return torch.minimum(alloc, request)

        if not bool(torch.any(partial).item()):
            out = torch.zeros_like(request)
            out = torch.where(full_fill.unsqueeze(1), request, out)
            return out

        for _ in range(_ALLOCATE_AVAILABLE_ITERS):
            mid = (low + high) * 0.5
            sum_mid = torch.sum(allocation_at(mid), dim=1)
            increase = sum_mid >= target
            low = torch.where(partial & increase, mid, low)
            high = torch.where(partial & ~increase, mid, high)

        alloc = allocation_at((low + high) * 0.5)
        alloc = torch.where(full_fill.unsqueeze(1), request, alloc)
        alloc = torch.where(none_fill.unsqueeze(1), torch.zeros_like(alloc), alloc)
        return alloc

    def _vector_elm_map(self, values, *_args, **_kwargs):
        return self._ensure_tensor(values, self._batch)

    def _ensure_condition_tensor(self, value) -> Tensor:
        tensor = torch.as_tensor(value, device=self.device)
        if tensor.ndim == 0:
            tensor = tensor.expand(self._batch)
        elif tensor.shape[0] != self._batch:
            if tensor.shape[0] == 1:
                tensor = tensor.expand(self._batch)
            else:
                raise ValueError(
                    f"Condition tensor with shape {tuple(tensor.shape)} "
                    f"cannot broadcast to batch {self._batch}"
                )
        if tensor.dtype != torch.bool:
            tensor = tensor != 0
        return tensor


class _RandomStream:
    def __init__(
        self,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
        seed: int | None,
    ) -> None:
        self.batch = batch
        self.dtype = dtype
        self.device = device
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(seed)

    def random_normal(
        self,
        minimum,
        maximum,
        mean,
        std,
        seed=None,  # noqa: D417 - matches Vensim signature
    ):
        minimum = self._tensorize(minimum)
        maximum = self._tensorize(maximum)
        mean = self._tensorize(mean)
        std = self._tensorize(std)
        noise = torch.randn(
            (self.batch,),
            dtype=self.dtype,
            device=self.device,
            generator=self.generator,
        )
        sample = mean + std * noise
        return torch.clamp(sample, min=minimum, max=maximum)

    def random_uniform(
        self,
        minimum,
        maximum,
        seed=None,  # noqa: D417 - Vensim signature compatibility
    ):
        minimum = self._tensorize(minimum)
        maximum = self._tensorize(maximum)
        noise = torch.rand(
            (self.batch,),
            dtype=self.dtype,
            device=self.device,
            generator=self.generator,
        )
        return minimum + (maximum - minimum) * noise

    def random_poisson(
        self,
        minimum,
        maximum,
        mean,
        *_args,
        **_kwargs,
    ):
        minimum = self._tensorize(minimum)
        maximum = self._tensorize(maximum)
        rate = torch.clamp(self._tensorize(mean), min=_SMALL_VENSIM)
        sample = torch.poisson(rate, generator=self.generator)
        return torch.clamp(sample, min=minimum, max=maximum)

    def random_gamma(
        self,
        minimum,
        maximum,
        alpha,
        offset,
        scale,
        seed=None,  # noqa: D417 - matches Vensim signature
    ):
        minimum = self._tensorize(minimum)
        maximum = self._tensorize(maximum)
        concentration = torch.clamp(self._tensorize(alpha), min=_SMALL_VENSIM)
        offset = self._tensorize(offset)
        scale = torch.clamp(self._tensorize(scale), min=_SMALL_VENSIM)

        sample = None
        if hasattr(torch, "_standard_gamma"):
            try:
                base = torch._standard_gamma(concentration, generator=self.generator)
                sample = base * scale
            except TypeError:
                base = torch._standard_gamma(concentration)
                sample = base * scale
        if sample is None:
            rate = torch.reciprocal(scale)
            dist = torch.distributions.Gamma(concentration, rate)
            sample = dist.sample()

        sample = offset + sample
        return torch.clamp(sample, min=minimum, max=maximum)

    def random_negative_binomial(
        self,
        minimum,
        maximum,
        prob,
        total_count,
        offset,
        scale,
        seed=None,  # noqa: D417 - matches Vensim signature
    ):
        minimum = self._tensorize(minimum)
        maximum = self._tensorize(maximum)
        probs = torch.clamp(
            self._tensorize(prob), min=_SMALL_VENSIM, max=1.0 - _SMALL_VENSIM
        )
        total_count = torch.clamp(
            self._tensorize(total_count), min=_SMALL_VENSIM
        )
        offset = self._tensorize(offset)
        scale = self._tensorize(scale)

        rate_scale = (1.0 - probs) / probs
        gamma_rate = None
        if hasattr(torch, "_standard_gamma"):
            try:
                base = torch._standard_gamma(total_count, generator=self.generator)
                gamma_rate = base * rate_scale
            except TypeError:
                base = torch._standard_gamma(total_count)
                gamma_rate = base * rate_scale
        if gamma_rate is None:
            rate = torch.reciprocal(rate_scale)
            dist = torch.distributions.Gamma(total_count, rate)
            gamma_rate = dist.sample()

        try:
            sample = torch.poisson(gamma_rate, generator=self.generator)
        except TypeError:
            sample = torch.poisson(gamma_rate)
        sample = offset + scale * sample
        return torch.clamp(sample, min=minimum, max=maximum)

    def _tensorize(self, value) -> Tensor:
        tensor = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        if tensor.ndim == 0:
            tensor = tensor.expand(self.batch)
        elif tensor.shape[0] != self.batch:
            if tensor.shape[0] == 1:
                tensor = tensor.expand(self.batch)
            else:
                raise ValueError(
                    f"Cannot broadcast tensor with shape {tuple(tensor.shape)} "
                    f"to batch {self.batch}"
                )
        return tensor
