from __future__ import annotations

import re


def canonical_name(name: str) -> str:
    """
    Convert a Vensim variable name into a Python-friendly identifier.
    """
    return re.sub(r"\W+", "_", name.strip().lower()).strip("_")
