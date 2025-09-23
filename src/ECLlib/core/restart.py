"""Simulation restart metadata."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Restart:
    """Restart metadata captured from Eclipse files."""

    start: datetime | None = None
    days: float = 0
    step: int = 0


__all__ = ["Restart"]
