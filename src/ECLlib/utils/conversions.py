"""Numeric conversion utilities."""
from __future__ import annotations


def ceildiv(a: int, b: int) -> int:
    """Return the ceiling of the division ``a / b``."""
    return -(-a // b)


__all__ = ["ceildiv"]
