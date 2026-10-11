"""Benchmarks: ``match`` (are the references real, correct and not retracted?) and
``support`` (does the cited work back the citing sentence?).

Run with ``citeright bench SUITE [options]``; each suite documents its options. They
need the ``bench`` extra, and ``support`` also needs ``models``.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence


def run(suite: str, argv: Sequence[str]) -> int:
    module = importlib.import_module(f"citeright.bench.{suite}")
    code: int = module.main(list(argv))
    return code
