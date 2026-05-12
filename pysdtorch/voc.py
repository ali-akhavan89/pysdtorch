from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Iterable, Tuple

from pysdtorch.translators.vensim_text import SubscriptManager, build_subscript_manager_from_vensim


def _split_name_subscripts(name: str) -> Tuple[str, Tuple[str, ...]]:
    if "[" not in name or "]" not in name:
        return name.strip(), tuple()
    base, rest = name.split("[", 1)
    selectors = rest.rsplit("]", 1)[0]
    tokens = [tok.strip() for tok in selectors.split(",") if tok.strip()]
    return base.strip(), tuple(tokens)


def expand_subscripted_name(name: str, sub_mgr: SubscriptManager) -> Tuple[str, ...]:
    """
    Expand a Vensim name like ``Foo[Edu]`` into concrete element names like
    ``Foo[e1]``, ``Foo[e2]``, ... using the model's subscript definitions.
    """
    base, selectors = _split_name_subscripts(name)
    if not selectors:
        clean = base.strip()
        return (clean,) if clean else tuple()

    choices: list[list[str]] = []
    for sel in selectors:
        clean = sel.rstrip("!").strip()
        if sub_mgr.has_range(clean):
            elements = list(sub_mgr.elements(clean))
            if not elements:
                raise ValueError(f"Subscript range '{clean}' has no elements (in '{name}').")
            choices.append(elements)
        else:
            choices.append([clean])

    expanded = [
        f"{base}[{', '.join(combo)}]"
        for combo in itertools.product(*choices)
    ]
    return tuple(expanded)


def parse_voc_bounds(
    voc_path: Path,
    *,
    mdl_path: Path | None = None,
    expand_subscripts: bool = True,
) -> Tuple[Dict[str, tuple[float, float]], Dict[str, Tuple[str, ...]]]:
    """
    Parse a Vensim VOC bounds file.

    If ``expand_subscripts`` is True, ``mdl_path`` must be provided so any
    subscripted parameters (e.g., ``Param[Edu]``) are expanded into per-element
    bounds (e.g., ``Param[e1]``, ``Param[e2]``, ...).

    Returns:
        (expanded_bounds, raw_to_expanded)
    """
    if expand_subscripts and mdl_path is None:
        raise ValueError("mdl_path is required when expand_subscripts=True.")

    sub_mgr: SubscriptManager | None = None
    if expand_subscripts:
        sub_mgr = build_subscript_manager_from_vensim(mdl_path)  # type: ignore[arg-type]

    bounds: Dict[str, tuple[float, float]] = {}
    raw_to_expanded: Dict[str, Tuple[str, ...]] = {}

    with open(voc_path, "r") as file:
        for line in file:
            stripped = line.strip()
            if not stripped or stripped.startswith(":"):
                continue

            raw_line = stripped.split(",", 1)[0].strip()
            parts = raw_line.split("<=")
            if len(parts) != 3:
                continue

            lower = float(parts[0].strip())
            raw_name = parts[1].strip()
            upper = float(parts[2].strip())

            expanded: Tuple[str, ...]
            if sub_mgr is not None:
                expanded = expand_subscripted_name(raw_name, sub_mgr)
            else:
                expanded = (raw_name,)
            raw_to_expanded[raw_name] = expanded

            for name in expanded:
                if name in bounds:
                    raise ValueError(f"Duplicate VOC bounds entry for '{name}'.")
                bounds[name] = (lower, upper)

    if not bounds:
        raise ValueError(f"No parameters parsed from VOC file: {voc_path}")

    return bounds, raw_to_expanded


def expand_names_from_voc(
    names: Iterable[str],
    voc_expansions: Dict[str, Tuple[str, ...]],
) -> Tuple[str, ...]:
    """
    Expand a list of VOC parameter identifiers using a mapping returned by
    :func:`parse_voc_bounds`.

    This is useful for handling user-supplied lists like process/measurement noise
    that may refer to unexpanded names (e.g., ``Param[Edu]``).
    """
    expanded: list[str] = []
    for raw in names:
        key = str(raw).strip()
        if not key:
            continue
        expanded.extend(voc_expansions.get(key, (key,)))
    # stable unique
    return tuple(dict.fromkeys(expanded))

