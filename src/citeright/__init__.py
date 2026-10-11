"""Citeright: check that every citation in a draft is real, correct, not retracted, and
backs the sentence that cites it."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("citeright")
except PackageNotFoundError:  # pragma: no cover - only when running from a source tree
    __version__ = "0.0.0"

__all__ = ["__version__"]
