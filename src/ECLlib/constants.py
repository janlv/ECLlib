"""Global constants used across :mod:`ECLlib`.

The module intentionally avoids importing :mod:`.unformatted_base` at import
time to prevent a circular dependency between the two modules.  The ``ENDSOL``
marker is therefore provided lazily via ``__getattr__``.
"""

# from __future__ import annotations

# from typing import TYPE_CHECKING, Any

DEBUG = False
ENDIAN = '>'  # Big-endian
ECL2IX_LOG = 'ecl2ix.log'

__all__ = ["DEBUG", "ENDIAN", "ECL2IX_LOG"] #, "ENDSOL"]

# if TYPE_CHECKING:  # pragma: no cover - import only needed for type checking
#     from .unformatted_base import unfmt_block as _unfmt_block

#     ENDSOL: _unfmt_block


# def __getattr__(name: str) -> Any:
#     """Provide lazy access to ``ENDSOL`` without creating import cycles."""

#     if name == "ENDSOL":
#         from .unformatted_base import ENDSOL as endsol

#         globals()[name] = endsol
#         return endsol
#     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
