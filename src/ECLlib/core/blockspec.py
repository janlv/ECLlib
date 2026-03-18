"""Structured block specifications for Eclipse-style file formats."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from numpy import asarray

from .datatypes import DTYPE

BlockDtype = Literal["int", "float", "double", "bool", "char", "mess"]

_VALID_DTYPES = ("int", "float", "double", "bool", "char", "mess")
_NUMPY_DTYPES = {
    "int": "i4",
    "float": "f4",
    "double": "f8",
    "bool": bool,
}
_CHAR_SIZE = DTYPE[b"CHAR"].size


def _flatten_array(array):
    """Return a one-dimensional array using Eclipse/Fortran ordering."""
    if array.ndim == 0:
        return array.reshape(1)
    if array.ndim > 1:
        return array.ravel(order="F")
    return array


def _flatten_values(values):
    """Return flattened Python values using Eclipse/Fortran ordering."""
    array = asarray(values)
    if array.ndim == 0:
        return (array.item(),)
    return tuple(array.ravel(order="F").tolist())


def _encode_char(value):
    """Encode a character payload entry to the fixed Eclipse CHAR width."""
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    elif type(value).__name__ == "bytes_":
        raw = bytes(value)
    elif isinstance(value, str):
        raw = value.encode("utf-8")
    elif type(value).__name__ == "str_":
        raw = str(value).encode("utf-8")
    else:
        raise TypeError(
            "CHAR blocks only accept str/bytes values; "
            f"got {type(value).__name__}"
        )
    if len(raw) > _CHAR_SIZE:
        raise ValueError(
            f"CHAR values must be <= {_CHAR_SIZE} bytes; got {len(raw)} for {raw!r}"
        )
    return raw


@dataclass(frozen=True, slots=True)
class BlockSpec:
    """User-facing payload description for one structured Eclipse block."""

    key: str
    data: object
    dtype: BlockDtype

    def __post_init__(self):
        if not isinstance(self.key, str):
            raise TypeError("Block keyword must be a string")
        key = self.key.strip()
        if not key:
            raise ValueError("Block keyword must not be empty")
        if len(key) > 8:
            raise ValueError(f"Block keyword must be 1-8 characters, got {self.key!r}")
        object.__setattr__(self, "key", key)
        if self.dtype not in _VALID_DTYPES:
            raise ValueError(f"Unsupported block dtype {self.dtype!r}. Valid values: {_VALID_DTYPES}")
        if self.dtype == "mess" and asarray(self.data).size:
            raise ValueError("MESS blocks must use an empty payload")
        if self.dtype == "char":
            for value in _flatten_values(self.data):
                _encode_char(value)

    def array(self):
        """Return the payload normalized for serialization."""
        if self.dtype == "mess":
            return asarray([], dtype="S1")
        if self.dtype == "char":
            encoded = [_encode_char(value) for value in _flatten_values(self.data)]
            return asarray(encoded, dtype=f"S{_CHAR_SIZE}")
        return _flatten_array(asarray(self.data, dtype=_NUMPY_DTYPES[self.dtype]))


__all__ = ["BlockSpec", "BlockDtype"]
