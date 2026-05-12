"""Global constants used across :mod:`ECLlib`.

The module intentionally avoids importing :mod:`.core.unformatted` at import
time to prevent a circular dependency between the two modules.  The ``ENDSOL``
marker is therefore provided lazily via ``__getattr__``.
"""

# from __future__ import annotations

# from typing import TYPE_CHECKING, Any

DEBUG = False
ENDIAN = '>'  # Big-endian

__all__ = ["DEBUG", "ENDIAN"] #, "ENDSOL"]

# if TYPE_CHECKING:  # pragma: no cover - import only needed for type checking
#     from .core.unformatted import unfmt_block as _unfmt_block

#     ENDSOL: _unfmt_block


# def __getattr__(name: str) -> Any:
#     """Provide lazy access to ``ENDSOL`` without creating import cycles."""

#     if name == "ENDSOL":
#         from .core.unformatted import ENDSOL as endsol

#         globals()[name] = endsol
#         return endsol
#     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
