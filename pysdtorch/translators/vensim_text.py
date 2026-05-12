from __future__ import annotations

import ast
import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from pysdtorch.ir import (
    Auxiliary,
    ControlParameter,
    ExpressionSpec,
    IRModel,
    LookupTable,
    Stock,
    VariableMeta,
)
from pysdtorch.utils import canonical_name


CONTROL_VARIABLES = {
    "initial time": "initial_time",
    "final time": "final_time",
    "time step": "time_step",
    "saveper": "saveper",
}

FUNCTION_REWRITES = {
    "MAX": "t_max",
    "MIN": "t_min",
    "ABS": "t_abs",
    "SQRT": "t_sqrt",
    "EXP": "t_exp",
    "LN": "t_log",
    "LOG": "t_log",
    "SIN": "t_sin",
    "ZIDZ": "t_zidz",
    "XIDZ": "t_xidz",
    "RANDOM NORMAL": "random_normal",
    "RANDOM_NORMAL": "random_normal",
    "RANDOM UNIFORM": "random_uniform",
    "RANDOM_UNIFORM": "random_uniform",
    "RANDOM POISSON": "random_poisson",
    "RANDOM_POISSON": "random_poisson",
    "RANDOM NEGATIVE BINOMIAL": "random_negative_binomial",
    "RANDOM_NEGATIVE_BINOMIAL": "random_negative_binomial",
    "RANDOM GAMMA": "random_gamma",
    "RANDOM_GAMMA": "random_gamma",
    "IF THEN ELSE": "t_if_then_else",
    "VECTOR ELM MAP": "vector_elm_map",
    "DELAY1I": "delay1i",
    "DELAY N": "delay_n",
    "DELAY_N": "delay_n",
    "STEP": "t_step",
    "RAMP": "t_ramp",
    "PULSE": "t_pulse",
    "SMOOTH": "t_smooth",
    "ALLOCATE AVAILABLE": "t_allocate_available",
    "ACTIVE INITIAL": "t_active_initial",
    "INITIAL": "t_initial",
    "INTEGER": "t_int",
}

SPECIAL_IDENTIFIERS = {
    "t_max",
    "t_min",
    "t_abs",
    "t_sqrt",
    "t_exp",
    "t_log",
    "t_sin",
    "t_zidz",
    "t_xidz",
    "random_normal",
    "random_uniform",
    "random_poisson",
    "random_negative_binomial",
    "random_gamma",
    "t_if_then_else",
    "t_all",
    "t_any",
    "t_not",
    "t_sum",
    "t_vmin",
    "t_vmax",
    "vector_elm_map",
    "delay1i",
    "t_step",
    "t_ramp",
    "t_pulse",
    "t_smooth",
    "t_initial",
    "t_active_initial",
    "t_initial_set",
    "t_initial_get",
    "t_int",
    "t_delay_n",
    "t_allocate_available",
    "__pysdtorch_initializing",
    "time",
    "time_step",
    "initial_time",
    "final_time",
    "saveper",
}

_STAGE_FLAG = "__pysdtorch_initializing"

_IDENT_BOUNDARY = r"[A-Za-z0-9_]"


def _normalize_name_key(name: str) -> str:
    return " ".join(name.strip().split()).lower()


def _normalize_function_key(token: str) -> str:
    return " ".join(token.strip().split()).upper()


_FUNCTION_REWRITE_MAP = {
    _normalize_function_key(token): replacement
    for token, replacement in FUNCTION_REWRITES.items()
}


def _build_function_rewrite_pattern() -> re.Pattern[str]:
    snippets: List[str] = []
    for token in FUNCTION_REWRITES.keys():
        words = token.split()
        if not words:
            continue
        snippet = r"\s+".join(re.escape(word) for word in words)
        snippets.append(snippet)
    snippets.sort(key=len, reverse=True)
    body = "|".join(snippets)
    return re.compile(
        rf"(?<!{_IDENT_BOUNDARY})({body})(?!{_IDENT_BOUNDARY})(?=\s*\()",
        re.IGNORECASE,
    )


_FUNCTION_REWRITE_PATTERN = _build_function_rewrite_pattern()
_ELMCOUNT_PATTERN = re.compile(r"elmcount\(([^)]+)\)", re.IGNORECASE)
# Require at least one non-space character in the name so we don't
# accidentally match Python list literals like ``[a, b]``.
_REFERENCE_PATTERN = re.compile(r"([A-Za-z0-9_][A-Za-z0-9 _]*?)\[(.*?)\]")
_TOKEN_PATTERN_CACHE: Dict[str, re.Pattern[str]] = {}
_LOOKUP_NUMBER_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)


@dataclass(frozen=True)
class _BaseNameReplacer:
    pattern: re.Pattern[str] | None
    replacements: Dict[str, str]

    def apply(self, expr: str) -> str:
        if self.pattern is None:
            return expr

        def _repl(match: re.Match[str]) -> str:
            key = _normalize_name_key(match.group(1))
            return self.replacements.get(key, match.group(0))

        return self.pattern.sub(_repl, expr)


def _build_base_name_replacer(base_name_map: Dict[str, str]) -> _BaseNameReplacer:
    if not base_name_map:
        return _BaseNameReplacer(pattern=None, replacements=base_name_map)

    keys = sorted(base_name_map.keys(), key=len, reverse=True)
    body = "|".join(re.escape(key) for key in keys)
    pattern = re.compile(
        rf"(?<!{_IDENT_BOUNDARY})({body})(?!{_IDENT_BOUNDARY})",
        re.IGNORECASE,
    )
    return _BaseNameReplacer(pattern=pattern, replacements=base_name_map)


@dataclass
class RawComponent:
    original_name: str
    base_name: str
    selectors: Tuple[str, ...]
    expression: str
    units: str
    documentation: str
    limits: Tuple[float | None, float | None] | None
    is_control: bool = False
    section: str = "main"


@dataclass
class RawLookup:
    original_name: str
    base_name: str
    selectors: Tuple[str, ...]
    expression: str
    units: str
    documentation: str
    limits: Tuple[float | None, float | None] | None


@dataclass
class SubscriptRange:
    name: str
    elements: Tuple[str, ...]
    alias_of: str | None = None
    mapped_to: Tuple[str, ...] = ()


class SubscriptManager:
    """
    Minimal helper to expand and index Vensim subscript ranges.
    """

    def __init__(self, ranges: Dict[str, SubscriptRange]):
        self._ranges = ranges
        self._index_maps: Dict[str, Dict[str, int]] = {
            name: self._build_index(range_def.elements) for name, range_def in ranges.items()
        }
        self._position_maps: Dict[str, Dict[str, int]] = {
            name: {element: idx for idx, element in enumerate(range_def.elements)}
            for name, range_def in ranges.items()
        }
        self._element_to_ranges: Dict[str, set[str]] = {}
        for name, range_def in ranges.items():
            for element in range_def.elements:
                self._element_to_ranges.setdefault(element, set()).add(name)

        self._mapping_adj: Dict[str, set[str]] = {name: set() for name in ranges}
        self._mapping_group: Dict[str, int] = {}
        self._mapping_group_members: Dict[int, Tuple[str, ...]] = {}
        self._build_mapping_graph()

    def _build_mapping_graph(self) -> None:
        def add_edge(left: str, right: str, label: str) -> None:
            if left not in self._ranges:
                raise ValueError(f"Unknown subscript range '{left}' referenced in mapping.")
            if right not in self._ranges:
                raise ValueError(
                    f"Unknown subscript range '{right}' referenced in mapping for '{left}'."
                )
            if len(self._ranges[left].elements) != len(self._ranges[right].elements):
                raise ValueError(
                    f"Subscript mapping '{left} {label} {right}' requires ranges of equal length "
                    f"({len(self._ranges[left].elements)} vs {len(self._ranges[right].elements)})."
                )
            self._mapping_adj[left].add(right)
            self._mapping_adj[right].add(left)

        for name, range_def in self._ranges.items():
            if range_def.alias_of is not None:
                add_edge(name, range_def.alias_of, label="alias")
            for target in range_def.mapped_to:
                add_edge(name, target, label="->")

        group_id = 0
        seen: set[str] = set()
        for name in self._ranges:
            if name in seen:
                continue
            stack = [name]
            members: list[str] = []
            while stack:
                current = stack.pop()
                if current in seen:
                    continue
                seen.add(current)
                self._mapping_group[current] = group_id
                members.append(current)
                stack.extend(self._mapping_adj.get(current, ()))
            self._mapping_group_members[group_id] = tuple(sorted(members))
            group_id += 1

    def has_range(self, name: str) -> bool:
        return name in self._ranges

    def has_element(self, label: str) -> bool:
        return label in self._element_to_ranges

    def ranges_for_element(self, label: str) -> Tuple[str, ...]:
        return tuple(sorted(self._element_to_ranges.get(label, set())))

    def elements(self, name: str) -> Tuple[str, ...]:
        try:
            return self._ranges[name].elements
        except KeyError:
            return ()

    def mapping_group(self, name: str) -> Tuple[str, ...]:
        """
        Return all range names participating in the same Vensim mapping group.

        Ranges are grouped when they are declared as aliases or mapped via ``->``.
        """
        group = self._mapping_group.get(name)
        if group is None:
            return ()
        return self._mapping_group_members.get(group, ())

    def map_element(self, from_range: str, element: str, to_range: str) -> str | None:
        if from_range == to_range:
            return element
        if not self.has_range(from_range) or not self.has_range(to_range):
            return None
        group_from = self._mapping_group.get(from_range)
        group_to = self._mapping_group.get(to_range)
        if group_from is None or group_to is None or group_from != group_to:
            return None
        pos_map = self._position_maps.get(from_range)
        if pos_map is None:
            return None
        pos = pos_map.get(element)
        if pos is None:
            return None
        target_elements = self._ranges[to_range].elements
        if pos >= len(target_elements):
            return None
        return target_elements[pos]

    def resolve_element(self, range_name: str, context_subs: Dict[str, str]) -> str | None:
        """
        Resolve a range selector token into a concrete element label.

        This supports Vensim mapped subscripts where a range can be used
        interchangeably with other mapped ranges of the same length.
        """
        if range_name in context_subs:
            return context_subs[range_name]
        if not self.has_range(range_name):
            return None
        for ctx_range, ctx_element in context_subs.items():
            mapped = self.map_element(ctx_range, ctx_element, range_name)
            if mapped is not None:
                return mapped

        # Support Vensim subranges: if one range is a subset/superset of the other,
        # resolve by shared element labels (e.g., Edu2 element e3 inside EduExt).
        target_elements = set(self.elements(range_name))
        candidates: set[str] = set()
        for ctx_range, ctx_element in context_subs.items():
            if not self.has_range(ctx_range):
                continue
            if ctx_element not in target_elements:
                continue
            ctx_elements = set(self.elements(ctx_range))
            if ctx_elements.issubset(target_elements) or target_elements.issubset(ctx_elements):
                candidates.add(ctx_element)
        if len(candidates) == 1:
            return next(iter(candidates))
        if len(candidates) > 1:
            raise ValueError(
                f"Ambiguous subscript resolution for range '{range_name}': candidates={sorted(candidates)}"
            )
        return None

    def selector_elements(self, token: str) -> List[str]:
        clean = token.strip().rstrip("!")
        if self.has_range(clean):
            return list(self.elements(clean))
        if self.has_element(clean):
            return [clean]
        return [clean]

    def index_of(self, range_name: str, element: str) -> int:
        if range_name in self._index_maps and element in self._index_maps[range_name]:
            return self._index_maps[range_name][element]

        if element in self._element_to_ranges:
            for candidate in self._element_to_ranges[element]:
                if candidate in self._index_maps and element in self._index_maps[candidate]:
                    return self._index_maps[candidate][element]
        raise KeyError(f"Unknown subscript element '{element}' for range '{range_name}'.")

    def previous_element(self, element: str) -> str | None:
        match = re.search(r"(\d+)$", element)
        if not match:
            return None
        prefix = element[: match.start(1)]
        number = int(match.group(1))
        if number <= 1:
            return None
        return f"{prefix}{number - 1}"

    @staticmethod
    def _build_index(elements: Sequence[str]) -> Dict[str, int]:
        index: Dict[str, int] = {}
        for idx, element in enumerate(elements, start=1):
            match = re.search(r"(\d+)$", element)
            value = int(match.group(1)) if match else idx
            index[element] = value
        return index


@dataclass
class _ExpandedComponent:
    name: str
    kind: str  # "aux", "stock", or "control"
    expression: str | None
    flow: str | None
    initial: str | None
    metadata: VariableMeta


def build_ir_from_vensim(mdl_path: str | Path) -> IRModel:
    path = Path(mdl_path)
    # Explicit UTF-8 avoids platform-default decoding issues (e.g. cp1252 on Windows).
    text = _strip_sketch(path.read_text(encoding="utf-8"))
    blocks = _split_blocks(_preprocess_lines(text))
    sub_blocks, component_blocks = _partition_blocks(blocks)
    sub_mgr = _build_subscript_manager(sub_blocks)
    components: List[RawComponent] = []
    raw_lookups: List[RawLookup] = []
    for block in component_blocks:
        lookup = _parse_lookup_block(block)
        if lookup is not None:
            raw_lookups.append(lookup)
            continue
        comp = _parse_block(block)
        if comp is not None:
            components.append(comp)
    base_name_map = {
        _normalize_name_key(comp.base_name): canonical_name(comp.base_name) for comp in components
    }
    for lookup in raw_lookups:
        base_name_map.setdefault(
            _normalize_name_key(lookup.base_name),
            canonical_name(lookup.base_name),
        )
    base_name_replacer = _build_base_name_replacer(base_name_map)
    expanded = _expand_components(components, sub_mgr, base_name_replacer)
    lookups = _expand_lookup_definitions(raw_lookups, sub_mgr)
    candidate_names = [item.name for item in expanded if item.kind in {"aux", "stock"}]

    stocks: Dict[str, Stock] = {}
    auxiliaries: Dict[str, Auxiliary] = {}
    control_specs: Dict[str, Tuple[str, VariableMeta]] = {}

    for item in expanded:
        if item.kind == "control":
            if item.expression is None:
                raise ValueError(f"Control '{item.name}' missing expression.")
            control_specs[item.name] = (item.expression, item.metadata)
            continue

        if item.kind == "stock":
            if item.flow is None or item.initial is None:
                raise ValueError(f"Stock '{item.name}' missing flow or initial expression.")
            flow_deps_run, flow_deps_init = _find_dependencies_by_stage(
                item.flow, candidate_names
            )
            init_deps_run, init_deps_init = _find_dependencies_by_stage(
                item.initial, candidate_names
            )
            stocks[item.name] = Stock(
                name=item.name,
                flow=ExpressionSpec(
                    source=item.flow,
                    dependencies=flow_deps_run,
                    init_dependencies=flow_deps_init,
                ),
                initial=ExpressionSpec(
                    source=item.initial,
                    dependencies=init_deps_run,
                    init_dependencies=init_deps_init,
                ),
                metadata=item.metadata,
            )
        else:
            if item.expression is None:
                raise ValueError(f"Auxiliary '{item.name}' missing expression.")
            deps_run, deps_init = _find_dependencies_by_stage(
                item.expression, candidate_names
            )
            auxiliaries[item.name] = Auxiliary(
                name=item.name,
                expression=ExpressionSpec(
                    source=item.expression,
                    dependencies=deps_run,
                    init_dependencies=deps_init,
                ),
                metadata=item.metadata,
            )

    controls = _resolve_controls(control_specs)

    return IRModel(
        name=path.stem,
        stocks=stocks,
        auxiliaries=auxiliaries,
        controls=controls,
        lookups=lookups,
        documentation=f"Translated from {path.name}",
    )


def build_subscript_manager_from_vensim(mdl_path: str | Path) -> SubscriptManager:
    """
    Parse a Vensim ``.mdl`` file and return a :class:`SubscriptManager` for its subscript ranges.
    """
    path = Path(mdl_path)
    # Ignore decoding errors to tolerate sketch sections containing non-UTF8 bytes.
    text = _strip_sketch(path.read_text(encoding="utf-8", errors="ignore"))
    blocks = _split_blocks(_preprocess_lines(text))
    sub_blocks, _component_blocks = _partition_blocks(blocks)
    return _build_subscript_manager(sub_blocks)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _strip_sketch(text: str) -> str:
    split = text.split("\\\\\\---///", 1)
    return split[0]


def _preprocess_lines(text: str) -> List[str]:
    lines = text.splitlines()
    cleaned: List[str] = []
    buffer = ""
    for raw in lines:
        if not raw.strip():
            if buffer:
                cleaned.append(buffer)
                buffer = ""
            cleaned.append("")
            continue

        line = raw.rstrip()
        if buffer:
            line = buffer + line.lstrip()
            buffer = ""

        if line.endswith("\\"):
            buffer = line[:-1]
            continue

        cleaned.append(line)

    if buffer:
        cleaned.append(buffer)
    return cleaned


def _split_blocks(lines: Sequence[str]) -> List[List[str]]:
    blocks: List[List[str]] = []
    block: List[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("****"):
            continue
        if stripped.startswith("$-") or stripped.startswith("*View"):
            continue
        if stripped.endswith("|"):
            content = re.sub(r"\|+$", "", line).rstrip()
            content = content.rstrip("~ ").rstrip()
            if content.strip():
                block.append(content)
            if block:
                blocks.append(block)
                block = []
            continue
        if stripped.startswith("{"):
            continue
        if stripped.startswith("\\\\"):
            break
        if stripped or block:
            block.append(line)
    if block:
        blocks.append(block)
    return blocks


def _partition_blocks(blocks: Sequence[Sequence[str]]) -> Tuple[List[List[str]], List[List[str]]]:
    sub_blocks: List[List[str]] = []
    component_blocks: List[List[str]] = []
    for block in blocks:
        if _is_subscript_block(block):
            sub_blocks.append(list(block))
        else:
            component_blocks.append(list(block))
    return sub_blocks, component_blocks


def _is_subscript_block(lines: Sequence[str]) -> bool:
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if "=" in stripped:
            return False
        if re.search(r":\s*raw\s*:?", stripped, re.IGNORECASE):
            return False
        return ":" in stripped
    return False


def _parse_subscript_block(lines: Sequence[str]) -> Tuple[str, str] | None:
    header = None
    payload: List[str] = []
    in_metadata = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if header is None:
            header = stripped
            continue
        if stripped.startswith("~"):
            in_metadata = True
            continue
        if in_metadata:
            continue
        payload.append(stripped)

    if header is None or ":" not in header or "=" in header:
        return None
    name, rest = header.split(":", 1)
    definition = rest.strip()
    if not definition and payload:
        definition = " ".join(payload).strip()
    return name.strip(), definition


def _build_subscript_manager(blocks: Sequence[Sequence[str]]) -> SubscriptManager:
    definitions: List[Tuple[str, str]] = []
    for block in blocks:
        parsed = _parse_subscript_block(block)
        if parsed is not None:
            definitions.append(parsed)

    ranges: Dict[str, SubscriptRange] = {}
    pending = list(definitions)
    while pending:
        progressed = False
        for name, definition in list(pending):
            try:
                elements, alias_of, mapped_to = _expand_subscript_definition(
                    definition, ranges, pending_names={name for name, _ in pending}
                )
            except KeyError:
                continue
            ranges[name] = SubscriptRange(
                name=name,
                elements=tuple(elements),
                alias_of=alias_of,
                mapped_to=tuple(mapped_to),
            )
            pending.remove((name, definition))
            progressed = True
        if not progressed:
            unresolved = ", ".join(name for name, _ in pending)
            raise ValueError(f"Unable to resolve subscript definitions: {unresolved}")

    return SubscriptManager(ranges)


def _expand_subscript_definition(
    definition: str,
    known: Dict[str, SubscriptRange],
    pending_names: set[str] | None = None,
) -> Tuple[List[str], str | None, List[str]]:
    definition = definition.strip()
    mapped_to: List[str] = []
    if "->" in definition:
        # Vensim allows mapping ranges (e.g. ``Edu2->xjt,xlt``). The torch backend
        # needs the full mapping so mapped subscripts can be resolved correctly.
        left, right = definition.split("->", 1)
        definition = left.strip()
        mapped_to = [tok.strip() for tok in right.split(",") if tok.strip()]
    if not definition:
        return [], None, mapped_to
    tokens = [tok.strip() for tok in definition.split(",") if tok.strip()]
    if len(tokens) == 1 and tokens[0] in known:
        return list(known[tokens[0]].elements), tokens[0], mapped_to
    if pending_names is None:
        pending_names = set()

    elements: List[str] = []
    for token in tokens:
        if token in known:
            elements.extend(list(known[token].elements))
            continue
        if token.startswith("(") and token.endswith(")"):
            elements.extend(_expand_numeric_range(token[1:-1]))
            continue
        if token in pending_names:
            # symbolic range not yet expanded; defer.
            raise KeyError(token)
        elements.append(token)
    return elements, None, mapped_to


def _expand_numeric_range(body: str) -> List[str]:
    if "-" not in body:
        return [body.strip()]
    start, end = body.split("-", 1)
    start = start.strip()
    end = end.strip()
    start_match = re.match(r"([A-Za-z_]*)(\d+)$", start)
    end_match = re.match(r"([A-Za-z_]*)(\d+)$", end)
    if not start_match or not end_match or start_match.group(1) != end_match.group(1):
        raise ValueError(f"Malformed subscript range '{body}'")
    prefix = start_match.group(1)
    start_num = int(start_match.group(2))
    end_num = int(end_match.group(2))
    if end_num < start_num:
        raise ValueError(f"Invalid numeric subscript range '{body}'")
    return [f"{prefix}{idx}" for idx in range(start_num, end_num + 1)]


def _parse_block(lines: Sequence[str]) -> RawComponent | None:
    expr_lines: List[str] = []
    units_line = ""
    doc_lines: List[str] = []
    in_metadata = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "|":
            in_metadata = True
            continue
        if stripped.startswith("~"):
            in_metadata = True
            payload = stripped.strip("~ ").strip()
            if not units_line:
                units_line = payload
            else:
                doc_lines.append(payload)
            continue
        if in_metadata:
            doc_lines.append(stripped)
            continue
        expr_lines.append(stripped)

    if not expr_lines:
        return None

    first_line = expr_lines[0]
    raw_match = re.match(r"(.+?):\s*RAW\s*:?\s*$", first_line, re.IGNORECASE)
    if raw_match:
        original_name = raw_match.group(1).strip()
        base_name, selectors = _split_name_subscripts(original_name)
        expression = "0"
        units, limits = _parse_units(units_line)
        documentation = " ".join(doc_lines).strip()
        return RawComponent(
            original_name=original_name,
            base_name=base_name,
            selectors=selectors,
            expression=expression,
            units=units,
            documentation=documentation,
            limits=limits,
            is_control=False,
        )

    if "=" not in first_line:
        return None

    name_part, expr_part = first_line.split("=", 1)
    original_name = name_part.strip()
    base_name, selectors = _split_name_subscripts(original_name)

    expression_lines: List[str] = []
    if expr_part.strip():
        expression_lines.append(expr_part.strip())
    expression_lines.extend(expr_lines[1:])
    expression = " ".join(expression_lines).strip()
    units, limits = _parse_units(units_line)
    documentation = " ".join(doc_lines).strip()
    key = base_name.lower()
    return RawComponent(
        original_name=original_name,
        base_name=base_name,
        selectors=selectors,
        expression=expression,
        units=units,
        documentation=documentation,
        limits=limits,
        is_control=key in CONTROL_VARIABLES,
    )


def _parse_lookup_block(lines: Sequence[str]) -> RawLookup | None:
    expr_lines: List[str] = []
    units_line = ""
    doc_lines: List[str] = []
    in_metadata = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "|":
            in_metadata = True
            continue
        if stripped.startswith("~"):
            in_metadata = True
            payload = stripped.strip("~ ").strip()
            if not units_line:
                units_line = payload
            else:
                doc_lines.append(payload)
            continue
        if in_metadata:
            doc_lines.append(stripped)
            continue
        expr_lines.append(stripped)

    if not expr_lines:
        return None

    first_line = expr_lines[0]
    if "=" in first_line:
        return None

    raw_match = re.match(r"(.+?):\s*RAW\s*:?\s*$", first_line, re.IGNORECASE)
    if raw_match:
        return None

    open_idx = first_line.find("(")
    if open_idx == -1:
        return None

    name_part = first_line[:open_idx].strip()
    if not name_part:
        return None

    expression_lines: List[str] = [first_line[open_idx:]]
    expression_lines.extend(expr_lines[1:])
    expression = " ".join(expression_lines).strip()

    original_name = name_part
    base_name, selectors = _split_name_subscripts(original_name)
    units, limits = _parse_units(units_line)
    documentation = " ".join(doc_lines).strip()
    return RawLookup(
        original_name=original_name,
        base_name=base_name,
        selectors=selectors,
        expression=expression,
        units=units,
        documentation=documentation,
        limits=limits,
    )


def _parse_units(units_line: str) -> Tuple[str, Tuple[float | None, float | None] | None]:
    if not units_line:
        return "", None

    if "[" not in units_line:
        return units_line.strip(), None

    units, rest = units_line.split("[", 1)
    rest = rest.strip(" ]")
    parts = [part.strip() for part in rest.split(",")]

    def _to_float(value: str) -> float | None:
        if not value or value == "?":
            return None
        try:
            return float(value)
        except ValueError:
            return None

    bounds = (_to_float(parts[0]), _to_float(parts[1]) if len(parts) > 1 else None)
    return units.strip(), bounds


def _split_name_subscripts(name: str) -> Tuple[str, Tuple[str, ...]]:
    if "[" not in name or "]" not in name:
        return name.strip(), tuple()
    base, rest = name.split("[", 1)
    selectors = rest.rsplit("]", 1)[0]
    tokens = [tok.strip() for tok in selectors.split(",") if tok.strip()]
    return base.strip(), tuple(tokens)


def _parse_lookup_points(expr: str, name: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    text = expr.strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()

    text = re.sub(
        r"^\[\s*\([^)]*\)\s*-\s*\([^)]*\)\s*\]\s*,?\s*",
        "",
        text,
    )

    pair_pattern = re.compile(
        rf"\(\s*({_LOOKUP_NUMBER_RE.pattern})\s*,\s*({_LOOKUP_NUMBER_RE.pattern})\s*\)"
    )
    pairs = pair_pattern.findall(text)
    if not pairs:
        raise ValueError(f"Lookup '{name}' has no data points.")

    x_vals = [float(x) for x, _ in pairs]
    y_vals = [float(y) for _, y in pairs]
    if len(x_vals) != len(y_vals):
        raise ValueError(f"Lookup '{name}' has mismatched x/y lengths.")

    combined = sorted(zip(x_vals, y_vals), key=lambda item: item[0])
    x_sorted = [x for x, _ in combined]
    y_sorted = [y for _, y in combined]
    for idx in range(1, len(x_sorted)):
        if x_sorted[idx] == x_sorted[idx - 1]:
            raise ValueError(
                f"Lookup '{name}' has repeated x value {x_sorted[idx]}."
            )

    return tuple(x_sorted), tuple(y_sorted)


# ---------------------------------------------------------------------------
# Expression utilities
# ---------------------------------------------------------------------------


def _normalize_raw_expression(expr: str) -> str:
    clean = expr.strip()
    clean = clean.replace("^", "**")

    def _rewrite_function(match: re.Match[str]) -> str:
        key = _normalize_function_key(match.group(1))
        return _FUNCTION_REWRITE_MAP.get(key, match.group(0))

    clean = _FUNCTION_REWRITE_PATTERN.sub(_rewrite_function, clean)
    clean = _rewrite_comparisons(clean)

    clean = _replace_token(clean, "TIME STEP", "time_step")
    clean = _replace_token(clean, "INITIAL TIME", "initial_time")
    clean = _replace_token(clean, "FINAL TIME", "final_time")
    clean = _replace_token(clean, "SAVEPER", "saveper")
    clean = _replace_token(clean, "TIME", "time")
    clean = _rewrite_logical_operators(clean)
    return " ".join(clean.split())


def _expand_components(
    components: Sequence[RawComponent],
    sub_mgr: SubscriptManager,
    base_name_replacer: _BaseNameReplacer,
) -> List[_ExpandedComponent]:
    expanded: List[_ExpandedComponent] = []
    for comp in components:
        normalized_expr = _normalize_raw_expression(comp.expression)
        selector_choices = [sub_mgr.selector_elements(sel) for sel in comp.selectors]
        for combo in itertools.product(*selector_choices):
            context: Dict[str, str] = {}
            for sel, value in zip(comp.selectors, combo):
                clean = sel.rstrip("!").strip()
                if sub_mgr.has_range(clean):
                    context[clean] = value
                    continue
                if sub_mgr.has_element(clean):
                    for range_name in sub_mgr.ranges_for_element(clean):
                        context.setdefault(range_name, clean)
                    context.setdefault(clean, clean)
                    continue
                context[clean] = value
            original_name = _format_original_name(comp.base_name, combo)
            safe_name = canonical_name(original_name)
            rewritten_expr = _rewrite_allocate_available(
                normalized_expr, canonical_name(comp.base_name), context, sub_mgr
            )
            translated_expr = _translate_expression(
                rewritten_expr, context, sub_mgr, base_name_replacer
            )
            translated_expr = _rewrite_active_initial(translated_expr)
            translated_expr = _rewrite_initial(translated_expr, safe_name)
            translated_expr = _rewrite_smooth(translated_expr, safe_name)
            translated_expr = _rewrite_delay_n(translated_expr, safe_name)
            if comp.is_control:
                expanded.append(
                    _ExpandedComponent(
                        name=safe_name,
                        kind="control",
                        expression=translated_expr,
                        flow=None,
                        initial=None,
                        metadata=_meta(original_name, comp),
                    )
                )
                continue

            integ = _parse_integ(translated_expr)
            delay_parts = _parse_delay1i(translated_expr, safe_name)
            if integ:
                flow_expr = integ.flow
                init_expr = integ.initial
                expanded.append(
                    _ExpandedComponent(
                        name=safe_name,
                        kind="stock",
                        expression=None,
                        flow=flow_expr,
                        initial=init_expr,
                        metadata=_meta(original_name, comp),
                    )
                )
            elif delay_parts:
                flow_expr, init_expr = delay_parts
                expanded.append(
                    _ExpandedComponent(
                        name=safe_name,
                        kind="stock",
                        expression=None,
                        flow=flow_expr,
                        initial=init_expr,
                        metadata=_meta(original_name, comp),
                    )
                )
            else:
                expanded.append(
                    _ExpandedComponent(
                        name=safe_name,
                        kind="aux",
                        expression=translated_expr,
                        flow=None,
                        initial=None,
                        metadata=_meta(original_name, comp),
                    )
                )
    return expanded


def _expand_lookup_definitions(
    lookups: Sequence[RawLookup],
    sub_mgr: SubscriptManager,
) -> Dict[str, LookupTable]:
    expanded: Dict[str, LookupTable] = {}
    for lookup in lookups:
        x_vals, y_vals = _parse_lookup_points(lookup.expression, lookup.original_name)
        selector_choices = [sub_mgr.selector_elements(sel) for sel in lookup.selectors]
        for combo in itertools.product(*selector_choices):
            original_name = _format_original_name(lookup.base_name, combo)
            safe_name = canonical_name(original_name)
            expanded[safe_name] = LookupTable(
                name=safe_name,
                x=x_vals,
                y=y_vals,
                metadata=VariableMeta(
                    original_name=original_name,
                    units=lookup.units,
                    documentation=lookup.documentation,
                    limits=lookup.limits,
                ),
            )
    return expanded


def _rewrite_smooth(expr: str, safe_name: str) -> str:
    """
    Rewrite ``t_smooth(input, smooth_time)`` into a stateful call.

    Vensim's SMOOTH is an exponential smoothing (1st order) whose initial
    value defaults to the input at initialization time. We implement it as a
    1st-order ``t_delay_n`` with ``initial=input`` and ``order=1``.
    """
    lower = expr.lower()
    idx = 0
    result: List[str] = []
    counter = 0
    token = "t_smooth"
    while idx < len(expr):
        pos = lower.find(token, idx)
        if pos == -1:
            result.append(expr[idx:])
            break
        if pos > 0 and (expr[pos - 1].isalnum() or expr[pos - 1] == "_"):
            result.append(expr[idx : pos + len(token)])
            idx = pos + len(token)
            continue
        open_idx = pos + len(token)
        while open_idx < len(expr) and expr[open_idx].isspace():
            open_idx += 1
        if open_idx >= len(expr) or expr[open_idx] != "(":
            result.append(expr[idx : pos + len(token)])
            idx = pos + len(token)
            continue
        end = _find_matching_paren(expr, open_idx)
        if end == -1:
            result.append(expr[idx:])
            break
        inner = expr[open_idx + 1 : end]
        parts = _split_top_level_commas(inner)
        if len(parts) != 2:
            raise ValueError(f"SMOOTH expects two arguments, got {expr}")
        smooth_input = parts[0].strip()
        smooth_time = parts[1].strip()
        identifier = f"'__smooth_{safe_name}_{counter}'"
        replacement = (
            f"t_delay_n({identifier}, ({smooth_input}), ({smooth_time}), "
            f"({smooth_input}), 1)"
        )
        result.append(expr[idx:pos])
        result.append(replacement)
        idx = end + 1
        counter += 1
    return "".join(result)


def _format_original_name(base_name: str, selectors: Sequence[str]) -> str:
    if not selectors:
        return base_name
    return f"{base_name}[{', '.join(selectors)}]"


def _translate_expression(
    expr: str,
    context_subs: Dict[str, str],
    sub_mgr: SubscriptManager,
    base_name_replacer: _BaseNameReplacer,
) -> str:
    literal = _select_literal_vector(expr, context_subs, sub_mgr)
    if literal is not None:
        return literal
    expr = _expand_aggregator(expr, "sum", context_subs, sub_mgr, base_name_replacer)
    expr = _expand_aggregator(expr, "vmax", context_subs, sub_mgr, base_name_replacer)
    expr = _expand_aggregator(expr, "vmin", context_subs, sub_mgr, base_name_replacer)
    expr = _expand_vector_elm_map(expr, context_subs, sub_mgr)
    expr = _replace_elmcount(expr, sub_mgr)
    expr = _translate_references(expr, context_subs, sub_mgr)
    expr = base_name_replacer.apply(expr)
    expr = _replace_subscript_tokens(expr, context_subs, sub_mgr)
    expr = " ".join(expr.split())
    return expr


def _rewrite_delay_n(expr: str, safe_name: str) -> str:
    lower = expr.lower()
    idx = 0
    result: List[str] = []
    counter = 0
    token = "delay_n"
    while idx < len(expr):
        pos = lower.find(token, idx)
        if pos == -1:
            result.append(expr[idx:])
            break
        if pos > 0 and (expr[pos - 1].isalnum() or expr[pos - 1] == "_"):
            result.append(expr[idx : pos + len(token)])
            idx = pos + len(token)
            continue
        open_idx = pos + len(token)
        while open_idx < len(expr) and expr[open_idx].isspace():
            open_idx += 1
        if open_idx >= len(expr) or expr[open_idx] != "(":
            result.append(expr[idx : pos + len(token)])
            idx = pos + len(token)
            continue
        end = _find_matching_paren(expr, open_idx)
        if end == -1:
            result.append(expr[idx:])
            break
        inner = expr[open_idx + 1 : end]
        parts = _split_top_level(inner)
        if len(parts) != 4:
            raise ValueError(f"DELAY N expects four arguments, got {expr}")
        identifier = f"'__delay_n_{safe_name}_{counter}'"
        replacement = f"t_delay_n({identifier}, {', '.join(p.strip() for p in parts)})"
        result.append(expr[idx:pos])
        result.append(replacement)
        idx = end + 1
        counter += 1
    return "".join(result)


def _rewrite_allocate_available(
    expr: str,
    identifier_base: str,
    context_subs: Dict[str, str],
    sub_mgr: SubscriptManager,
) -> str:
    """
    Rewrite ``t_allocate_available(request, pp, avail)`` into a scalar expression.

    The torch backend evaluates a scalar per expanded subscript element. Vensim's
    ALLOCATE AVAILABLE returns a vector across the allocation dimension, so we:

    - Expand ``request`` into a Python list across the allocation dimension.
    - Expand ``pp`` into a matrix list (targets × pprofile elements).
    - Add an identifier for per-step caching, and an index selecting the current
      element from the computed allocation vector.
    """

    def _parse_reference(ref: str) -> Tuple[str, List[str]]:
        match = re.match(r"(.+?)\[(.+)\]\s*$", ref)
        if not match:
            raise ValueError(
                "ALLOCATE AVAILABLE expects subscripted references for request/pp; "
                f"got '{ref}'"
            )
        base = match.group(1).strip()
        selectors = [tok.strip() for tok in match.group(2).split(",") if tok.strip()]
        return base, selectors

    def _resolve_allocation_dim(selectors: Sequence[str]) -> str:
        dims: List[str] = []
        for tok in selectors:
            clean = tok.rstrip("!").strip()
            if sub_mgr.has_range(clean) and sub_mgr.resolve_element(clean, context_subs) is not None:
                dims.append(clean)
        dims = list(dict.fromkeys(dims))
        if len(dims) != 1:
            raise ValueError(
                "ALLOCATE AVAILABLE expects exactly one allocation dimension "
                f"present in the component context; got {dims or 'none'}"
            )
        return dims[0]

    def _substitute_selector(token: str, alloc_dim: str, alloc_element: str) -> str:
        clean = token.rstrip("!").strip()
        if clean == alloc_dim:
            return alloc_element
        context_with_alloc = dict(context_subs)
        context_with_alloc[alloc_dim] = alloc_element
        if clean in context_with_alloc:
            return context_with_alloc[clean]
        if sub_mgr.has_range(clean):
            mapped = sub_mgr.resolve_element(clean, context_with_alloc)
            if mapped is not None:
                return mapped
            raise ValueError(
                "ALLOCATE AVAILABLE cannot expand a scalar backend over "
                f"unresolved range '{clean}'."
            )
        return clean

    def _expand_vector(ref: str, alloc_dim: str) -> str:
        base, selectors = _parse_reference(ref)
        expanded: List[str] = []
        for elem in sub_mgr.elements(alloc_dim):
            subs = [_substitute_selector(tok, alloc_dim, elem) for tok in selectors]
            expanded.append(f"{base}[{', '.join(subs)}]")
        return f"[{', '.join(expanded)}]"

    def _expand_pp_matrix(ref: str, alloc_dim: str) -> str:
        base, selectors = _parse_reference(ref)
        alloc_positions = [
            idx
            for idx, tok in enumerate(selectors)
            if tok.rstrip("!").strip() == alloc_dim
        ]
        if len(alloc_positions) != 1:
            raise ValueError(
                f"ALLOCATE AVAILABLE pp reference must include '{alloc_dim}' exactly once: {ref}"
            )
        alloc_pos = alloc_positions[0]
        profile_pos = len(selectors) - 1
        profile_token = selectors[profile_pos].rstrip("!").strip()

        if sub_mgr.has_range(profile_token):
            profile_range = profile_token
        elif sub_mgr.has_element(profile_token):
            candidates = sub_mgr.ranges_for_element(profile_token)
            if not candidates:
                raise ValueError(
                    f"Unable to resolve priority profile range for '{profile_token}'."
                )
            expected = {"ptype", "ppriority", "pwidth", "pextra"}
            profile_range = next(
                (
                    name
                    for name in candidates
                    if expected.issubset(set(sub_mgr.elements(name)))
                ),
                candidates[0],
            )
        else:
            raise ValueError(
                f"Unable to resolve priority profile range for '{profile_token}'."
            )

        profile_elements = sub_mgr.elements(profile_range)
        if not profile_elements:
            raise ValueError(
                f"Priority profile range '{profile_range}' has no elements."
            )

        rows: List[str] = []
        alloc_elements = sub_mgr.elements(alloc_dim)
        for alloc_elem in alloc_elements:
            row_terms: List[str] = []
            for profile_elem in profile_elements:
                subs: List[str] = []
                for idx, tok in enumerate(selectors):
                    if idx == alloc_pos:
                        subs.append(alloc_elem)
                    elif idx == profile_pos:
                        subs.append(profile_elem)
                    else:
                        subs.append(_substitute_selector(tok, alloc_dim, alloc_elem))
                row_terms.append(f"{base}[{', '.join(subs)}]")
            rows.append(f"[{', '.join(row_terms)}]")
        return f"[{', '.join(rows)}]"

    def _allocation_index(alloc_dim: str) -> int:
        element = sub_mgr.resolve_element(alloc_dim, context_subs)
        if element is None:
            raise ValueError(
                f"ALLOCATE AVAILABLE allocation dimension '{alloc_dim}' could not be resolved from context."
            )
        elements = list(sub_mgr.elements(alloc_dim))
        try:
            return elements.index(element)
        except ValueError as exc:
            raise ValueError(
                f"ALLOCATE AVAILABLE element '{element}' not found in range '{alloc_dim}'."
            ) from exc

    lower = expr.lower()
    idx = 0
    result: List[str] = []
    token = "t_allocate_available"
    counter = 0

    while idx < len(expr):
        pos = lower.find(token, idx)
        if pos == -1:
            result.append(expr[idx:])
            break
        if pos > 0 and (expr[pos - 1].isalnum() or expr[pos - 1] == "_"):
            result.append(expr[idx : pos + len(token)])
            idx = pos + len(token)
            continue

        open_idx = pos + len(token)
        while open_idx < len(expr) and expr[open_idx].isspace():
            open_idx += 1
        if open_idx >= len(expr) or expr[open_idx] != "(":
            result.append(expr[idx : pos + len(token)])
            idx = pos + len(token)
            continue

        end = _find_matching_paren(expr, open_idx)
        if end == -1:
            result.append(expr[idx:])
            break

        inner = expr[open_idx + 1 : end]
        parts = _split_top_level_commas(inner)
        if len(parts) != 3:
            raise ValueError(
                f"ALLOCATE AVAILABLE expects three arguments, got {expr}"
            )

        request_ref = parts[0].strip()
        pp_ref = parts[1].strip()
        avail_expr = parts[2].strip()

        req_base, req_selectors = _parse_reference(request_ref)
        alloc_dim = _resolve_allocation_dim(req_selectors)
        req_vector = _expand_vector(
            f"{req_base}[{', '.join(req_selectors)}]",
            alloc_dim,
        )
        pp_matrix = _expand_pp_matrix(pp_ref, alloc_dim)
        element_index = _allocation_index(alloc_dim)

        suffix = "_".join(
            [
                f"{dim}_{context_subs[dim]}"
                for dim in sorted(context_subs)
                if dim != alloc_dim and sub_mgr.has_range(dim)
            ]
        )
        middle = f"_{suffix}" if suffix else ""
        identifier = f"'__allocate_available_{identifier_base}{middle}_{counter}'"
        replacement = (
            f"t_allocate_available({identifier}, {req_vector}, {pp_matrix}, "
            f"({avail_expr}), {element_index})"
        )

        result.append(expr[idx:pos])
        result.append(replacement)
        idx = end + 1
        counter += 1

    return "".join(result)


def _rewrite_active_initial(expr: str) -> str:
    """
    Rewrite ``t_active_initial(run_expr, init_expr)`` into a Python conditional
    expression that is only evaluated for the active stage.
    """
    lower = expr.lower()
    idx = 0
    result: List[str] = []
    token = "t_active_initial"
    while idx < len(expr):
        pos = lower.find(token, idx)
        if pos == -1:
            result.append(expr[idx:])
            break
        if pos > 0 and (expr[pos - 1].isalnum() or expr[pos - 1] == "_"):
            result.append(expr[idx : pos + len(token)])
            idx = pos + len(token)
            continue
        open_idx = pos + len(token)
        while open_idx < len(expr) and expr[open_idx].isspace():
            open_idx += 1
        if open_idx >= len(expr) or expr[open_idx] != "(":
            result.append(expr[idx : pos + len(token)])
            idx = pos + len(token)
            continue
        end = _find_matching_paren(expr, open_idx)
        if end == -1:
            result.append(expr[idx:])
            break
        inner = expr[open_idx + 1 : end]
        parts = _split_top_level(inner)
        if len(parts) != 2:
            raise ValueError(f"ACTIVE INITIAL expects two arguments, got {expr}")
        run_expr = parts[0].strip()
        init_expr = parts[1].strip()
        replacement = f"(({init_expr}) if {_STAGE_FLAG} else ({run_expr}))"
        result.append(expr[idx:pos])
        result.append(replacement)
        idx = end + 1
    return "".join(result)


def _rewrite_initial(expr: str, safe_name: str) -> str:
    """
    Rewrite ``t_initial(expr)`` into a stage-aware cache:

    - During initialization: cache the expression value.
    - During run: read the cached value without re-evaluating the expression.
    """
    lower = expr.lower()
    idx = 0
    result: List[str] = []
    counter = 0
    token = "t_initial"
    while idx < len(expr):
        pos = lower.find(token, idx)
        if pos == -1:
            result.append(expr[idx:])
            break
        if pos > 0 and (expr[pos - 1].isalnum() or expr[pos - 1] == "_"):
            result.append(expr[idx : pos + len(token)])
            idx = pos + len(token)
            continue
        open_idx = pos + len(token)
        while open_idx < len(expr) and expr[open_idx].isspace():
            open_idx += 1
        if open_idx >= len(expr) or expr[open_idx] != "(":
            result.append(expr[idx : pos + len(token)])
            idx = pos + len(token)
            continue
        end = _find_matching_paren(expr, open_idx)
        if end == -1:
            result.append(expr[idx:])
            break
        inner = expr[open_idx + 1 : end]
        parts = _split_top_level(inner)
        if len(parts) != 1:
            raise ValueError(f"INITIAL expects one argument, got {expr}")
        arg = parts[0].strip()
        identifier = f"'__initial_{safe_name}_{counter}'"
        replacement = (
            f"(t_initial_set({identifier}, ({arg})) if {_STAGE_FLAG} "
            f"else t_initial_get({identifier}))"
        )
        result.append(expr[idx:pos])
        result.append(replacement)
        idx = end + 1
        counter += 1
    return "".join(result)


def _expand_aggregator(
    expr: str,
    func: str,
    context_subs: Dict[str, str],
    sub_mgr: SubscriptManager,
    base_name_replacer: _BaseNameReplacer,
) -> str:
    lower = expr.lower()
    result: List[str] = []
    idx = 0
    func_token = func.lower()
    while idx < len(expr):
        pos = lower.find(f"{func_token}(", idx)
        if pos == -1:
            result.append(expr[idx:])
            break
        if pos > 0 and (expr[pos - 1].isalnum() or expr[pos - 1] == "_"):
            result.append(expr[idx : pos + len(func_token)])
            idx = pos + len(func_token)
            continue
        if pos > idx:
            result.append(expr[idx:pos])
        open_idx = pos + len(func_token)
        while open_idx < len(expr) and expr[open_idx].isspace():
            open_idx += 1
        if open_idx >= len(expr) or expr[open_idx] != "(":
            result.append(expr[pos])
            idx = pos + 1
            continue
        end = _find_matching_paren(expr, open_idx)
        if end == -1:
            result.append(expr[pos:])
            break
        inner = expr[open_idx + 1 : end]
        flagged = _find_flagged_dims(inner)
        if not flagged:
            translated_inner = _translate_expression(
                inner, context_subs, sub_mgr, base_name_replacer
            )
            replacement = translated_inner
        else:
            choices = [sub_mgr.selector_elements(dim) for dim in flagged]
            terms: List[str] = []
            for combo in itertools.product(*choices):
                inner_context = dict(context_subs)
                for dim, value in zip(flagged, combo):
                    inner_context[dim] = value
                cleaned_inner = inner.replace("!", "")
                translated = _translate_expression(
                    cleaned_inner, inner_context, sub_mgr, base_name_replacer
                )
                terms.append(translated)
            wrappers = {"sum": "t_sum", "vmax": "t_vmax", "vmin": "t_vmin"}
            try:
                wrapper = wrappers[func_token]
            except KeyError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported aggregator '{func}'") from exc
            replacement = f"{wrapper}([{', '.join(terms)}])"
        result.append(replacement)
        idx = end + 1
    return "".join(result)


def _find_flagged_dims(expr: str) -> List[str]:
    dims: List[str] = []
    seen: set[str] = set()
    for content in re.findall(r"\[(.*?)\]", expr):
        tokens = [tok.strip() for tok in content.split(",") if tok.strip()]
        for tok in tokens:
            if tok.endswith("!"):
                clean = tok.rstrip("!").strip()
                if clean not in seen:
                    seen.add(clean)
                    dims.append(clean)
    return dims


def _replace_elmcount(expr: str, sub_mgr: SubscriptManager) -> str:
    def _repl(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        if sub_mgr.has_range(name):
            return str(len(sub_mgr.elements(name)))
        return "0"

    return _ELMCOUNT_PATTERN.sub(_repl, expr)


def _expand_vector_elm_map(
    expr: str, context_subs: Dict[str, str], sub_mgr: SubscriptManager
) -> str:
    lower = expr.lower()
    result: List[str] = []
    idx = 0
    token = "vector_elm_map("
    while idx < len(expr):
        pos = lower.find(token, idx)
        if pos == -1:
            result.append(expr[idx:])
            break
        result.append(expr[idx:pos])
        start = pos + len(token) - 1
        end = _find_matching_paren(expr, start)
        if end == -1:
            result.append(expr[pos:])
            break
        inner = expr[start + 1 : end]
        parts = _split_top_level(inner)
        replacement = expr[pos : end + 1]
        if len(parts) == 2:
            ref = parts[0].strip()
            offset_expr = _replace_elmcount(parts[1].strip(), sub_mgr)
            try:
                offset = int(float(eval(offset_expr, {"__builtins__": {}}, {})))
            except Exception:
                offset = 0
            ref_match = re.match(r"(.+?)\[(.+)\]$", ref)
            if ref_match:
                ref_base = ref_match.group(1).strip()
                selectors = [tok.strip() for tok in ref_match.group(2).split(",") if tok.strip()]
                if selectors and selectors[0] in context_subs and len(selectors) >= 2:
                    current_first = context_subs.get(selectors[0], selectors[0])
                    second_label = context_subs.get(selectors[1], selectors[1])
                    prev_first = sub_mgr.previous_element(current_first) if offset else current_first
                    if prev_first is None:
                        replacement = "0"
                    else:
                        replacement = canonical_name(
                            " ".join([ref_base, prev_first, second_label])
                        )
        result.append(replacement)
        idx = end + 1
    return "".join(result)


def _translate_references(
    expr: str, context_subs: Dict[str, str], sub_mgr: SubscriptManager
) -> str:
    def _repl(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        selectors = match.group(2)
        return _translate_reference(name, selectors, context_subs, sub_mgr)

    return _REFERENCE_PATTERN.sub(_repl, expr)


def _translate_reference(
    name: str, selectors: str, context_subs: Dict[str, str], sub_mgr: SubscriptManager
) -> str:
    tokens = [tok.strip() for tok in selectors.split(",") if tok.strip()]
    elements: List[str] = []
    for token in tokens:
        clean = token.rstrip("!").strip()
        if clean in context_subs:
            elements.append(context_subs[clean])
        elif sub_mgr.has_range(clean):
            resolved = sub_mgr.resolve_element(clean, context_subs)
            if resolved is not None:
                elements.append(resolved)
            else:
                options = sub_mgr.selector_elements(clean)
                if len(options) == 1:
                    elements.append(options[0])
                else:
                    raise ValueError(
                        f"Unable to resolve mapped subscript range '{clean}' while translating "
                        f"reference '{name}[{selectors}]'."
                    )
        else:
            elements.append(clean)
    return canonical_name(" ".join([name] + elements))


def _replace_subscript_tokens(
    expr: str, context_subs: Dict[str, str], sub_mgr: SubscriptManager
) -> str:
    candidates: set[str] = set()
    for token in context_subs:
        if not sub_mgr.has_range(token):
            continue
        candidates.add(token)
        candidates.update(sub_mgr.mapping_group(token))
        try:
            resolved = sub_mgr.resolve_element(token, context_subs)
        except ValueError:
            resolved = None
        if resolved is None:
            continue
        for range_name in sub_mgr.ranges_for_element(resolved):
            candidates.add(range_name)
            candidates.update(sub_mgr.mapping_group(range_name))

    for token in candidates:
        pattern = _TOKEN_PATTERN_CACHE.get(token.lower())
        if pattern is None:
            pattern = re.compile(
                rf"(?<!{_IDENT_BOUNDARY}){re.escape(token)}(?!{_IDENT_BOUNDARY})",
                re.IGNORECASE,
            )
            _TOKEN_PATTERN_CACHE[token.lower()] = pattern
        if pattern.search(expr) is None:
            continue
        resolved = sub_mgr.resolve_element(token, context_subs)
        if resolved is None:
            continue
        try:
            index = sub_mgr.index_of(token, resolved)
        except KeyError:
            continue
        expr = _replace_token(expr, token, str(index))
    return expr


# ---------------------------------------------------------------------------
# INTEG and delay parsing
# ---------------------------------------------------------------------------


@dataclass
class _IntegParts:
    flow: str
    initial: str


def _parse_integ(expr: str) -> _IntegParts | None:
    if not expr.lower().startswith("integ"):
        return None
    start = expr.find("(")
    end = expr.rfind(")")
    if start == -1 or end == -1:
        raise ValueError(f"Malformed INTEG expression: {expr}")
    body = expr[start + 1 : end]
    parts = _split_top_level(body)
    if len(parts) != 2:
        raise ValueError(f"INTEG expects two arguments, got {expr}")
    return _IntegParts(flow=parts[0].strip(), initial=parts[1].strip())


def _parse_delay1i(expr: str, safe_name: str) -> Tuple[str, str] | None:
    if not expr.lower().startswith("delay1i"):
        return None
    start = expr.find("(")
    end = expr.rfind(")")
    if start == -1 or end == -1:
        raise ValueError(f"Malformed DELAY1I expression: {expr}")
    body = expr[start + 1 : end]
    parts = _split_top_level(body)
    if len(parts) != 3:
        raise ValueError(f"DELAY1I expects three arguments, got {expr}")
    inflow = parts[0].strip()
    delay = parts[1].strip()
    initial = parts[2].strip()
    flow_expr = f"({inflow} - {safe_name}) / ({delay})"
    return flow_expr, initial


def _split_top_level(text: str) -> List[str]:
    depth = 0
    parts: List[str] = []
    start = 0
    for idx, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:idx])
            start = idx + 1
    parts.append(text[start:])
    return parts


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------


def _meta(original_name: str, comp: RawComponent) -> VariableMeta:
    return VariableMeta(
        original_name=original_name,
        units=comp.units,
        documentation=comp.documentation,
        limits=comp.limits,
    )


def _resolve_controls(
    specs: Dict[str, Tuple[str, VariableMeta]]
) -> Dict[str, ControlParameter]:
    values: Dict[str, float] = {}
    controls: Dict[str, ControlParameter] = {}
    pending = dict(specs)

    while pending:
        progressed = False
        for name, (expr, meta) in list(pending.items()):
            try:
                value = _evaluate_constant(expr, values)
            except NameError:
                continue
            controls[name] = ControlParameter(
                name=name,
                value=value,
                metadata=meta,
            )
            values[name] = value
            pending.pop(name)
            progressed = True
        if not progressed:
            unresolved = ", ".join(sorted(pending.keys()))
            raise ValueError(f"Unable to resolve control parameters: {unresolved}")

    return controls


def _evaluate_constant(expr: str, context: Dict[str, float] | None = None) -> float:
    local_context = dict(context or {})
    return float(eval(expr, {"__builtins__": {}}, local_context))


def _replace_function_token(expr: str, token: str, replacement: str) -> str:
    """
    Replace a Vensim function token only when it is used as a function call
    (i.e. followed by an opening parenthesis), so variable names starting
    with the same word are left intact.
    """
    parts = [re.escape(part) for part in token.split()]
    pattern_body = r"\s+".join(parts)
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_]){pattern_body}(?![A-Za-z0-9_])(?=\s*\()",
        re.IGNORECASE,
    )
    return pattern.sub(replacement, expr)


def _replace_token(expr: str, token: str, replacement: str) -> str:
    cache_key = token.lower()
    pattern = _TOKEN_PATTERN_CACHE.get(cache_key)
    if pattern is None:
        pattern = re.compile(
            rf"(?<!{_IDENT_BOUNDARY}){re.escape(token)}(?!{_IDENT_BOUNDARY})",
            re.IGNORECASE,
        )
        _TOKEN_PATTERN_CACHE[cache_key] = pattern
    return pattern.sub(replacement, expr)


def _find_dependencies_by_stage(
    expr: str, candidates: Iterable[str]
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """
    Extract dependencies for run vs initialization stages.

    The translator rewrites Vensim stage-dependent functions (INITIAL, ACTIVE INITIAL)
    into Python conditional expressions guarded by ``_STAGE_FLAG``. Since those
    conditionals are lazy, dependency graphs can differ between stages.
    """
    candidates_set = set(candidates)
    try:
        root = ast.parse(expr, mode="eval")
    except SyntaxError:
        deps = _find_dependencies_fallback(expr, candidates_set)
        return deps, deps

    def deps_for_stage(node: ast.AST, stage: str) -> set[str]:
        if isinstance(node, ast.Name):
            if node.id in candidates_set and node.id not in SPECIAL_IDENTIFIERS:
                return {node.id}
            return set()
        if isinstance(node, ast.IfExp):
            deps = deps_for_stage(node.test, stage)
            if isinstance(node.test, ast.Name) and node.test.id == _STAGE_FLAG:
                branch = node.body if stage == "init" else node.orelse
                deps.update(deps_for_stage(branch, stage))
                return deps
            # Unknown conditional: conservatively include both branches.
            deps.update(deps_for_stage(node.body, stage))
            deps.update(deps_for_stage(node.orelse, stage))
            return deps

        deps: set[str] = set()
        for child in ast.iter_child_nodes(node):
            deps.update(deps_for_stage(child, stage))
        return deps

    deps_run = sorted(deps_for_stage(root.body, "run"))
    deps_init = sorted(deps_for_stage(root.body, "init"))
    return tuple(deps_run), tuple(deps_init)


def _find_dependencies_fallback(expr: str, candidates: set[str]) -> Tuple[str, ...]:
    stripped = re.sub(r"(\"[^\"]*\"|'[^']*')", " ", expr)
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", stripped))
    deps = sorted(
        token
        for token in tokens
        if token in candidates and token not in SPECIAL_IDENTIFIERS
    )
    return tuple(deps)


def _rewrite_comparisons(expr: str) -> str:
    expr = _replace_token(expr, "<>", "!=")
    expr = re.sub(r"(?<![<>=!])=(?!=)", "==", expr)
    return expr


def _select_literal_vector(
    expr: str,
    context_subs: Dict[str, str],
    sub_mgr: SubscriptManager,
) -> str | None:
    """
    Handle Vensim literal vectors like ``0,0.3`` assigned to subscripted variables.
    Returns the scalar element matching the current subscript context, or None if
    the expression is not a plain literal vector.
    """
    expr = expr.strip()
    if "," not in expr:
        return None
    # Skip if commas are likely argument separators inside parentheses.
    if "(" in expr or ")" in expr or "[" in expr or "]" in expr:
        return None

    if ";" in expr:
        # Handle 2D literal matrices written as comma-separated rows split by ';'.
        try:
            rows = [row.strip() for row in expr.split(";") if row.strip()]
            matrix = [
                [float(part.strip()) for part in row.split(",") if part.strip()]
                for row in rows
            ]
        except ValueError:
            return None
        if not matrix or not matrix[0]:
            return None
        ncols = len(matrix[0])
        if any(len(row) != ncols for row in matrix):
            return None

        nrows = len(matrix)
        tokens = list(context_subs.keys())
        for row_token in tokens:
            row_elements = sub_mgr.elements(row_token)
            if not row_elements or len(row_elements) != nrows:
                continue
            try:
                row_idx = list(row_elements).index(context_subs[row_token])
            except ValueError:
                continue
            for col_token in tokens:
                if col_token == row_token:
                    continue
                col_elements = sub_mgr.elements(col_token)
                if not col_elements or len(col_elements) != ncols:
                    continue
                try:
                    col_idx = list(col_elements).index(context_subs[col_token])
                except ValueError:
                    continue
                return str(matrix[row_idx][col_idx])
        return None

    parts = [part.strip() for part in expr.split(",") if part.strip()]
    if len(parts) <= 1:
        return None
    try:
        literals = [float(p) for p in parts]
    except ValueError:
        return None

    for token, element in context_subs.items():
        elements = sub_mgr.elements(token)
        if elements and len(elements) == len(literals):
            try:
                idx = list(elements).index(element)
            except ValueError:
                continue
            return str(literals[idx])
    return None


def _rewrite_logical_operators(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return expr

    comma_parts = _split_top_level_commas(expr)
    if len(comma_parts) > 1:
        rewritten = [_rewrite_logical_operators(part) for part in comma_parts]
        return ", ".join(rewritten)

    rebuilt: List[str] = []
    idx = 0
    while idx < len(expr):
        char = expr[idx]
        if char == "(":
            end = _find_matching_paren(expr, idx)
            if end == -1:
                return expr
            inner = _rewrite_logical_operators(expr[idx + 1 : end])
            rebuilt.append(f"({inner})")
            idx = end + 1
        else:
            rebuilt.append(char)
            idx += 1
    expr = "".join(rebuilt)

    stripped = expr.lstrip()
    leading = expr[: len(expr) - len(stripped)]
    if stripped.lower().startswith(":not:"):
        remainder = stripped[5:]
        rewritten = _rewrite_logical_operators(remainder)
        return leading + f"t_not({rewritten})"

    for token, func in ((":and:", "t_all"), (":or:", "t_any")):
        parts = _split_at_token(expr, token)
        if len(parts) > 1:
            rewritten_parts = [
                _rewrite_logical_operators(part) for part in parts if part.strip()
            ]
            return f"{func}({', '.join(rewritten_parts)})"

    return expr


def _split_at_token(expr: str, token: str) -> List[str]:
    lower = expr.lower()
    token = token.lower()
    token_len = len(token)
    parts: List[str] = []
    depth = 0
    last = 0
    idx = 0
    while idx < len(expr):
        char = expr[idx]
        if char == "(":
            depth += 1
            idx += 1
            continue
        if char == ")":
            depth = max(depth - 1, 0)
            idx += 1
            continue
        if char == "[":
            depth += 1
            idx += 1
            continue
        if char == "]":
            depth = max(depth - 1, 0)
            idx += 1
            continue
        if depth == 0 and lower.startswith(token, idx):
            parts.append(expr[last:idx])
            idx += token_len
            last = idx
            continue
        idx += 1
    if parts:
        parts.append(expr[last:])
    return parts if parts else [expr]


def _find_matching_paren(text: str, start: int) -> int:
    depth = 0
    for idx in range(start, len(text)):
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _split_top_level_commas(expr: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    last = 0
    for idx, char in enumerate(expr):
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(depth - 1, 0)
        elif char == "[":
            depth += 1
        elif char == "]":
            depth = max(depth - 1, 0)
        elif char == "," and depth == 0:
            parts.append(expr[last:idx])
            last = idx + 1
    if parts:
        parts.append(expr[last:])
    return parts if parts else [expr]
