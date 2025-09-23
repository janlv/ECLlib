"""Binary datatype descriptors for Eclipse files."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Dtyp:
    """Descriptor for an Eclipse binary datatype."""

    name: str = ""
    unpack: str = ""
    size: int = 0
    max: int = 0
    nptype: type | None = None
    max_bytes: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "max_bytes", self.max * self.size)


DTYPE = {
    b"INTE": Dtyp("INTE", "i", 4, 1000, "i4"),
    b"REAL": Dtyp("REAL", "f", 4, 1000, "f4"),
    b"DOUB": Dtyp("DOUB", "d", 8, 1000, "f8"),
    b"LOGI": Dtyp("LOGI", "i", 4, 1000, "b1"),
    b"CHAR": Dtyp("CHAR", "s", 8, 105, "S8"),
    b"C008": Dtyp("C008", "s", 8, 105, "S8"),
    b"C009": Dtyp("C009", "s", 9, 105, "S9"),
    b"MESS": Dtyp("MESS", "s", 1, 1, "S1"),
}

DTYPE_LIST = [v.name for v in DTYPE.values()]

__all__ = ["Dtyp", "DTYPE", "DTYPE_LIST"]
