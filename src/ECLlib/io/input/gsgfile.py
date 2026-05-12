"""Read Petrel/INTERSECT GSG property and grid files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ._gsg_binary import (
    GSGFormatError,
    GSGMetadataError,
    UnsupportedGSGBlock,
    read_exact,
    read_header,
    read_index_records,
    read_keyword,
    read_struct,
)

__all__ = [
    "GSGFile",
    "GSGIndexEntry",
    "GSGProperty",
    "GSGGrid",
    "GSGPillars",
    "GSGFormatError",
    "GSGMetadataError",
    "UnsupportedGSGBlock",
]

_GSG_DTYPES = {
    0: np.dtype("<i4"),
    1: np.dtype("<f4"),
    2: np.dtype("<f8"),
}
_ENCODINGS = {
    0: "full",
    1: "rle",
}


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


@dataclass(frozen=True, slots=True)
#===================================================================================================
class GSGPillars:                                                                       # GSGPillars
#===================================================================================================
    """PILLARS metadata and optionally decoded pillar rows."""

    header: tuple[int, ...]
    grid_type: int
    span: GSGIndexEntry
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


#---------------------------------------------------------------------------------------------------
def _decode_token(value):
#---------------------------------------------------------------------------------------------------
    """Decode a fixed-width GSG byte token to text."""
    if isinstance(value, bytes):
        return value.decode("utf-8").strip(" \x00")
    return str(value).strip()


#---------------------------------------------------------------------------------------------------
def _reshape_values(values, shape):
#---------------------------------------------------------------------------------------------------
    """Reshape expanded GSG property values using Fortran ordering."""
    if shape is None:
        return values
    return values.reshape(tuple(shape), order="F")


#---------------------------------------------------------------------------------------------------
def _entry_from_record(record):
#---------------------------------------------------------------------------------------------------
    """Return a public index entry from a private index tuple."""
    return GSGIndexEntry(*record)


#===================================================================================================
class GSGFile:                                                                             # GSGFile
#===================================================================================================
    """Read-only parser for Petrel/INTERSECT ``.GSG`` files."""

    #-----------------------------------------------------------------------------------------------
    def __init__(self, path):                                                            # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Initialize the parser for ``path``."""
        self.path = Path(path)
        self.creator = ""
        self.version = ""
        self._data_start = 0
        self._format = None
        self._index = None
        with self.path.open("rb") as file_obj:
            self.creator, self.version, self._data_start = read_header(file_obj)
            self._index = tuple(_entry_from_record(record) for record in read_index_records(file_obj))
        if not self._index:
            raise GSGMetadataError("GSG file has no indexed blocks")
        self._format = self._index[0].key

    @property
    #-----------------------------------------------------------------------------------------------
    def format(self):                                                                    # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Return the first indexed GSG keyword, usually ``PROP`` or ``AXES``."""
        return self._format

    #-----------------------------------------------------------------------------------------------
    def index(self):                                                                     # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Return ordered index entries, preserving repeated keys."""
        if self._index is None:
            with self.path.open("rb") as file_obj:
                self._index = tuple(_entry_from_record(record) for record in read_index_records(file_obj))
        return self._index

    #-----------------------------------------------------------------------------------------------
    def blocks(self, key=None):                                                          # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Yield indexed block spans, optionally filtered by keyword."""
        for entry in self.index():
            if key is None or entry.key == key:
                yield entry

    #-----------------------------------------------------------------------------------------------
    def properties(self, expand=True, shape=None):                                       # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Yield decoded PROP records as :class:`GSGProperty` objects."""
        if self.format != "PROP":
            raise GSGFormatError(f"properties() requires a PROP GSG file, got {self.format!r}")
        if shape is not None and not expand:
            raise ValueError("shape requires expand=True")

        names_by_prop = self._case_property_names()
        with self.path.open("rb") as file_obj:
            for entry in self.blocks("PROP"):
                yield self._read_property(file_obj, entry, names_by_prop, expand, shape)

    #-----------------------------------------------------------------------------------------------
    def property(self, alias_or_name, expand=True, shape=None):                          # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Return one property matched by alias or CASE_PROPS name."""
        target = alias_or_name.casefold()
        for prop in self.properties(expand=expand, shape=shape):
            names = (prop.alias, *prop.names)
            if any(name.casefold() == target for name in names):
                return prop
        raise KeyError(alias_or_name)

    #-----------------------------------------------------------------------------------------------
    def grid(self, read_areal=True, read_pillars=False, read_faults=False):              # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Return decoded AXES grid metadata and optional payload arrays."""
        if self.format != "AXES":
            raise GSGFormatError(f"grid() requires an AXES GSG file, got {self.format!r}")

        entries = self.index()
        entry_by_key = {entry.key: entry for entry in entries if entry.key not in {"FAULTS"}}
        fault_entries = tuple(entry for entry in entries if entry.key == "FAULTS")
        with self.path.open("rb") as file_obj:
            axes = self._read_axes(file_obj, entry_by_key["AXES"])
            name, dimensions = self._read_grid_case(file_obj, entry_by_key["GRID"])
            areal = None
            if read_areal and "AREAL" in entry_by_key:
                areal = self._read_areal(file_obj, entry_by_key["AREAL"])
            pillars = None
            if "PILLARS" in entry_by_key:
                pillars = self._read_pillars(file_obj, entry_by_key["PILLARS"], read_pillars)
            defined_cells = self._read_defined_cells(file_obj, entries)
        faults = fault_entries if read_faults else ()
        return GSGGrid(name, dimensions, axes, areal, pillars, faults, defined_cells)

    #-----------------------------------------------------------------------------------------------
    def _case_property_names(self):                                                       # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Return CASE_PROPS names grouped by one-based property number."""
        entries = tuple(self.blocks("CASE_PROPS"))
        if not entries:
            return {}
        entry = entries[-1]
        grouped = {}
        with self.path.open("rb") as file_obj:
            key, data, _, _ = read_keyword(file_obj, "i4s2i", offset=entry.block_start)
            if key != "CASE_PROPS":
                raise GSGMetadataError(f"Expected CASE_PROPS, got {key!r}")
            for _ in range(data[3]):
                read_struct(file_obj, "8si")
                name, payload, _, _ = read_keyword(file_obj, "4s2i")
                role = _decode_token(payload[0])
                prop_number = payload[2]
                names, roles = grouped.setdefault(prop_number, ([], []))
                names.append(name)
                roles.append(role)
        return {
            number: (tuple(names), tuple(roles))
            for number, (names, roles) in grouped.items()
        }

    #-----------------------------------------------------------------------------------------------
    def _read_property(self, file_obj, entry, names_by_prop, expand, shape):              # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Read one indexed PROP block."""
        key, prop_data, _, _ = read_keyword(file_obj, "2i", offset=entry.block_start)
        if key != "PROP":
            raise GSGMetadataError(f"Expected PROP at byte {entry.block_start}, got {key!r}")

        encoding_code, prop_number = prop_data
        encoding = _ENCODINGS.get(encoding_code)
        if encoding is None:
            raise UnsupportedGSGBlock(f"Unsupported PROP encoding {encoding_code}")

        alias, payload, _, _ = read_keyword(file_obj, "4sqi")
        metadata, dtype_code, size = payload
        if _decode_token(metadata) != "ca":
            raise GSGMetadataError(f"Unexpected PROP metadata token {metadata!r}")
        dtype = _GSG_DTYPES.get(dtype_code)
        if dtype is None:
            raise UnsupportedGSGBlock(f"Unsupported PROP dtype code {dtype_code}")

        values = self._read_property_values(file_obj, entry.block_end, dtype, encoding, size, expand)
        values = _reshape_values(values, shape)
        names, roles = names_by_prop.get(prop_number, ((), ()))
        return GSGProperty(names, roles, alias, dtype, encoding, size, values)

    #-----------------------------------------------------------------------------------------------
    def _read_property_values(self, file_obj, block_end, dtype, encoding, size, expand):  # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Read full or RLE property payload values."""
        remaining = block_end - file_obj.tell()
        if remaining < 0:
            raise GSGMetadataError("Property payload extends beyond its indexed block")
        if encoding == "full":
            expected = size * dtype.itemsize
            if remaining != expected:
                raise GSGMetadataError(f"Expected {expected} bytes for full PROP, got {remaining}")
            values = np.fromfile(file_obj, dtype=dtype, count=size)
            if values.size != size:
                raise GSGFormatError(f"Expected {size} PROP values, got {values.size}")
            return values

        read_struct(file_obj, "i")
        pair_dtype = np.dtype([("count", "<i4"), ("value", dtype)])
        payload_size = remaining - 4
        if payload_size < 0 or payload_size % pair_dtype.itemsize:
            raise GSGMetadataError(f"Invalid RLE payload size {remaining}")
        pairs = np.frombuffer(read_exact(file_obj, payload_size), dtype=pair_dtype)
        if pairs["count"].sum(dtype=np.int64) != size:
            raise GSGMetadataError(f"RLE counts do not expand to {size} values")
        if not expand:
            return pairs.copy()
        values = np.repeat(pairs["value"], pairs["count"])
        if values.size != size:
            raise GSGFormatError(f"Expected {size} expanded PROP values, got {values.size}")
        return values

    #-----------------------------------------------------------------------------------------------
    def _read_axes(self, file_obj, entry):                                                  # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Read the AXES payload."""
        key, data, _, _ = read_keyword(file_obj, "i12s2i6f", offset=entry.block_start)
        if key != "AXES":
            raise GSGMetadataError(f"Expected AXES, got {key!r}")
        return data

    #-----------------------------------------------------------------------------------------------
    def _read_grid_case(self, file_obj, entry):                                           # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Read GRID metadata and the following case-name keyword."""
        key, _, _, _ = read_keyword(file_obj, "2i", offset=entry.block_start)
        if key != "GRID":
            raise GSGMetadataError(f"Expected GRID, got {key!r}")
        case_name, data, _, _ = read_keyword(file_obj, "4i")
        dimensions = tuple(int(value) for value in data[1:4])
        return case_name, dimensions

    #-----------------------------------------------------------------------------------------------
    def _read_areal(self, file_obj, entry):                                               # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Read AREAL rows as an ``int32`` array."""
        key, data, _, _ = read_keyword(file_obj, "5i", offset=entry.block_start)
        if key != "AREAL":
            raise GSGMetadataError(f"Expected AREAL, got {key!r}")
        row_count = data[3]
        values = np.fromfile(file_obj, dtype=np.dtype("<i4"), count=row_count * 6)
        if values.size != row_count * 6:
            raise GSGFormatError(f"Expected {row_count * 6} AREAL integers, got {values.size}")
        return values.reshape((row_count, 6))

    #-----------------------------------------------------------------------------------------------
    def _read_pillars(self, file_obj, entry, read_values):                                # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Read PILLARS metadata and optionally simple-grid pillar rows."""
        key, header, _, _ = read_keyword(file_obj, "17i", offset=entry.block_start)
        if key != "PILLARS":
            raise GSGMetadataError(f"Expected PILLARS, got {key!r}")
        grid_type = header[14]
        if not read_values:
            return GSGPillars(header, grid_type, entry)
        if grid_type != 0:
            raise UnsupportedGSGBlock(f"Cannot decode PILLARS for grid type {grid_type}")

        row_count = header[2]
        row_width = header[3] + 5
        row_dtype = np.dtype([("pillar", "<i4"), ("values", "<f4", row_width - 1)])
        rows = np.fromfile(file_obj, dtype=row_dtype, count=row_count)
        if rows.size != row_count:
            raise GSGFormatError(f"Expected {row_count} PILLARS rows, got {rows.size}")
        values = np.empty((row_count, row_width), dtype=np.float32)
        values[:, 0] = rows["pillar"]
        values[:, 1:] = rows["values"]
        return GSGPillars(header, grid_type, entry, values)

    #-----------------------------------------------------------------------------------------------
    def _read_defined_cells(self, file_obj, entries):                                     # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Read child DEFINED_CELLS records from compound grid spans."""
        cells = []
        for entry in entries:
            if entry.key == "PROP":
                read_keyword(file_obj, "2i", offset=entry.block_start)
                fmt = "4s6i"
            elif entry.key == "CASE_PROPS":
                read_keyword(file_obj, "i4si8si", offset=entry.block_start)
                fmt = "4s3i"
            else:
                continue
            if file_obj.tell() >= entry.block_end:
                continue
            position = file_obj.tell()
            key, _, _, _ = read_keyword(file_obj, offset=position)
            if key == "DEFINED_CELLS":
                _, data, _, _ = read_keyword(file_obj, fmt, offset=position)
                cells.append(data)
        return tuple(cells)
