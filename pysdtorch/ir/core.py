from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


@dataclass(slots=True)
class VariableMeta:
    """Metadata carried over from the source model."""

    original_name: str
    units: str = ""
    documentation: str = ""
    limits: Optional[Tuple[Optional[float], Optional[float]]] = None


@dataclass(slots=True)
class ExpressionSpec:
    """
    Representation of an expression before it is compiled for a backend.

    ``source`` keeps the cleaned textual form (Python-compatible) while
    ``compiled`` can store a backend-specific artifact (e.g., ``code``
    objects for the Torch/Numpy runtimes).
    """

    source: str
    compiled: Optional[object] = None
    dependencies: Tuple[str, ...] = field(default_factory=tuple)
    init_dependencies: Tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class Auxiliary:
    """An algebraic equation evaluated every simulation step."""

    name: str
    expression: ExpressionSpec
    metadata: VariableMeta
    kind: str = "auxiliary"


@dataclass(slots=True)
class LookupTable:
    """A hardcoded lookup table defined in the source model."""

    name: str
    x: Tuple[float, ...]
    y: Tuple[float, ...]
    metadata: VariableMeta
    kind: str = "lookup"


@dataclass(slots=True)
class Stock:
    """A first-order stock defined via ``INTEG(flow, initial)``."""

    name: str
    flow: ExpressionSpec
    initial: ExpressionSpec
    metadata: VariableMeta
    non_negative: bool = False
    kind: str = "stock"


@dataclass(slots=True)
class ControlParameter:
    """Simulation control knobs such as ``final_time`` and ``time_step``."""

    name: str
    value: float
    metadata: VariableMeta
    kind: str = "control"


@dataclass
class IRModel:
    """
    Backend-agnostic intermediate representation of a System Dynamics model.
    """

    name: str
    stocks: Dict[str, Stock]
    auxiliaries: Dict[str, Auxiliary]
    controls: Dict[str, ControlParameter]
    lookups: Dict[str, LookupTable] = field(default_factory=dict)
    documentation: str = ""

    def variables(self) -> Dict[str, Auxiliary | Stock]:
        """Convenience dict with every dynamic variable."""
        merged: Dict[str, Auxiliary | Stock] = {}
        merged.update(self.auxiliaries)
        merged.update(self.stocks)
        return merged

    def stock_names(self) -> Tuple[str, ...]:
        return tuple(self.stocks.keys())

    def control_value(self, name: str, default: Optional[float] = None) -> float:
        try:
            return self.controls[name].value
        except KeyError:
            if default is None:
                raise
            return default


def topo_sort(nodes: Iterable[str], edges: Dict[str, Set[str]]) -> List[str]:
    """
    Classic Kahn topological sort. ``edges`` maps node -> required parents.
    """
    remaining = {node: set(edges.get(node, set())) for node in nodes}
    ready = [node for node, deps in remaining.items() if not deps]
    order: List[str] = []

    while ready:
        node = ready.pop()
        order.append(node)
        remaining.pop(node, None)
        for follower, deps in list(remaining.items()):
            if node in deps:
                deps.remove(node)
                if not deps:
                    ready.append(follower)

    if remaining:
        cycle = ", ".join(sorted(remaining.keys()))
        raise ValueError(f"Cyclic dependency detected: {cycle}")

    return order


def dependency_graph(
    variables: Sequence[Auxiliary | Stock],
    exclude: Optional[Set[str]] = None,
) -> Dict[str, Set[str]]:
    """
    Build a mapping ``var -> dependencies`` ignoring ``exclude`` nodes.
    """
    exclude = exclude or set()
    graph: Dict[str, Set[str]] = {}
    for var in variables:
        deps = set(var.expression.dependencies) if isinstance(var, Auxiliary) else set()
        if isinstance(var, Stock):
            deps.update(var.flow.dependencies)
            deps.discard(var.name)  # flows can refer to the stock itself.
        deps.difference_update(exclude)
        graph[var.name] = deps
    return graph
