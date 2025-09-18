"""Numeric conversion utilities."""
from __future__ import annotations

from molmass import Formula


def ppm2molL(specie: str) -> float:
    """Convert parts-per-million to mol/L for the provided chemical ``specie``."""
    return 1 / 1000 / Formula(specie).mass


def molL2ppm(specie: str) -> float:
    """Convert mol/L to parts-per-million for the provided chemical ``specie``."""
    return 1000 * Formula(specie).mass


def ceildiv(a: int, b: int) -> int:
    """Return the ceiling of the division ``a / b``."""
    return -(-a // b)


__all__ = ["ceildiv", "molL2ppm", "ppm2molL"]
