"""Shared typed objects and small helpers for GSG files."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "GSGIndexEntry",
    "GSGProperty",
    "GSGPillars",
    "GSGGrid",
    "GSG_DTYPES",
    "GSG_ENCODINGS",
    "DTYPE_CODES",
    "ENCODING_CODES",
    "as_text_tuple",
    "decode_token",
    "entry_from_record",
    "normalize_encoding",
    "property_dtype",
    "reshape_values",
]

GSG_DTYPES = {
    0: np.dtype("<i4"),
    1: np.dtype("<f4"),
    2: np.dtype("<f8"),
}
GSG_ENCODINGS = {
    0: "full",
    1: "rle",
}
DTYPE_CODES = {dtype: code for code, dtype in GSG_DTYPES.items()}
ENCODING_CODES = {encoding: code for code, encoding in GSG_ENCODINGS.items()}


@dataclass(frozen=True, slots=True)
#===================================================================================================
class GSGIndexEntry:                                                                 # GSGIndexEntry
#===================================================================================================
    """Ordered metadata for one indexed GSG block."""

    key: str
    value: int
    data_pos: int
    block_start: int
    block_end: int


@dataclass(frozen=True, slots=True)
#===================================================================================================
class GSGProperty:                                                                     # GSGProperty
#===================================================================================================
    """Decoded GSG property values and their CASE_PROPS names."""

    names: tuple[str, ...]
    roles: tuple[str, ...]
    alias: str
    dtype: np.dtype
    encoding: str
    size: int
    values: np.ndarray

    @classmethod
    #-----------------------------------------------------------------------------------------------
    def from_array(cls, name, alias, values, role="g", dtype=None, encoding="full"):   # GSGProperty
    #-----------------------------------------------------------------------------------------------
        """Create a property definition from array-like values."""
        names = as_text_tuple(name, "name")
        roles = as_text_tuple(role, "role")
        if len(roles) == 1 and len(names) > 1:
            roles = len(names) * roles
        if len(roles) != len(names):
            raise ValueError("role must be one value or match the number of names")

        dtype = property_dtype(values, dtype)
        array = np.asarray(values, dtype=dtype)
        encoding = normalize_encoding(encoding)
        return cls(names, roles, str(alias), dtype, encoding, int(array.size), array)


@dataclass(frozen=True, slots=True)
#===================================================================================================
class GSGPillars:                                                                       # GSGPillars
#===================================================================================================
    """PILLARS metadata and optionally decoded pillar rows."""

    header: tuple[int, ...]
    grid_type: int
    span: GSGIndexEntry | None = None
    values: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
#===================================================================================================
class GSGGrid:                                                                             # GSGGrid
#===================================================================================================
    """Decoded grid metadata and optional geometry payloads."""

    name: str
    dimensions: tuple[int, int, int]
    axes: tuple
    areal: np.ndarray | None = None
    pillars: GSGPillars | None = None
    faults: tuple[GSGIndexEntry, ...] = ()
    defined_cells: tuple[tuple, ...] = ()
    active: np.ndarray | None = None

    @classmethod
    #-----------------------------------------------------------------------------------------------
    def from_egrid(                                                                      # GSGGrid
        cls, source, *, name=None, active=True
    ):
    #-----------------------------------------------------------------------------------------------
        """Create a writable simple GSG grid from an Eclipse/INTERSECT EGRID file."""
        from .grid import grid_from_egrid

        return grid_from_egrid(cls, source, name=name, active=active)


#---------------------------------------------------------------------------------------------------
def decode_token(value):
#---------------------------------------------------------------------------------------------------
    """Decode a fixed-width GSG byte token to text."""
    if isinstance(value, bytes):
        return value.decode("utf-8").strip(" \x00")
    return str(value).strip()


#---------------------------------------------------------------------------------------------------
def reshape_values(values, shape):
#---------------------------------------------------------------------------------------------------
    """Reshape expanded GSG property values using Fortran ordering."""
    if shape is None:
        return values
    return values.reshape(tuple(shape), order="F")


#---------------------------------------------------------------------------------------------------
def entry_from_record(record):
#---------------------------------------------------------------------------------------------------
    """Return a public index entry from a private index tuple."""
    return GSGIndexEntry(*record)


#---------------------------------------------------------------------------------------------------
def as_text_tuple(value, label):
#---------------------------------------------------------------------------------------------------
    """Return a non-empty tuple of strings."""
    if isinstance(value, str):
        values = (value,)
    else:
        values = tuple(str(item) for item in value)
    if not values or any(not item for item in values):
        raise ValueError(f"{label} must contain at least one non-empty value")
    return values


#---------------------------------------------------------------------------------------------------
def property_dtype(values, dtype=None):
#---------------------------------------------------------------------------------------------------
    """Return a supported little-endian GSG property dtype."""
    if dtype is not None:
        candidate = np.dtype(dtype)
    else:
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.integer):
            candidate = np.dtype("int32")
        elif np.issubdtype(array.dtype, np.floating):
            candidate = np.dtype("float32")
        else:
            raise TypeError(f"Unsupported property value dtype {array.dtype}")
    candidate = candidate.newbyteorder("<")
    if candidate not in DTYPE_CODES:
        raise TypeError(f"Unsupported GSG property dtype {candidate}")
    return candidate


#---------------------------------------------------------------------------------------------------
def normalize_encoding(encoding):
#---------------------------------------------------------------------------------------------------
    """Return a supported property encoding name."""
    encoding = str(encoding).lower()
    if encoding not in ENCODING_CODES:
        raise ValueError(f"Unsupported GSG property encoding {encoding!r}")
    return encoding
