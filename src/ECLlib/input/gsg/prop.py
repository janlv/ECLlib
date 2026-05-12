"""PROP read/write helpers for Petrel/INTERSECT GSG files."""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .binary import (
    GSGFormatError,
    GSGMetadataError,
    UnsupportedGSGBlock,
    read_exact,
    read_header,
    read_index_records,
    read_keyword,
    read_struct,
    write_header,
    write_index_records,
    write_keyword,
)
from .types import (
    DTYPE_CODES,
    ENCODING_CODES,
    GSG_DTYPES,
    GSG_ENCODINGS,
    GSGProperty,
    as_text_tuple,
    decode_token,
    entry_from_record,
    normalize_encoding,
    property_dtype,
    reshape_values,
)

__all__ = [
    "active_property",
    "case_property_names",
    "data_position",
    "find_property",
    "read_properties",
    "read_property",
    "write_case_props",
    "write_prop_file",
    "write_property_values",
]


#---------------------------------------------------------------------------------------------------
def read_properties(path, entries, expand=True, shape=None):
#---------------------------------------------------------------------------------------------------
    """Yield decoded PROP records as :class:`GSGProperty` objects."""
    if shape is not None and not expand:
        raise ValueError("shape requires expand=True")

    names_by_prop = case_property_names(path, entries)
    with Path(path).open("rb") as file_obj:
        for entry in entries:
            if entry.key == "PROP":
                yield read_property(file_obj, entry, names_by_prop, expand, shape)


#---------------------------------------------------------------------------------------------------
def find_property(path, entries, alias_or_name, expand=True, shape=None):
#---------------------------------------------------------------------------------------------------
    """Return one property matched by alias or CASE_PROPS name."""
    target = alias_or_name.casefold()
    for prop in read_properties(path, entries, expand=expand, shape=shape):
        names = (prop.alias, *prop.names)
        if any(name.casefold() == target for name in names):
            return prop
    raise KeyError(alias_or_name)


#---------------------------------------------------------------------------------------------------
def case_property_names(path, entries):
#---------------------------------------------------------------------------------------------------
    """Return CASE_PROPS names grouped by one-based property number."""
    case_entries = tuple(entry for entry in entries if entry.key == "CASE_PROPS")
    if not case_entries:
        return {}
    entry = case_entries[-1]
    grouped = {}
    with Path(path).open("rb") as file_obj:
        key, data, _, _ = read_keyword(file_obj, "i4s2i", offset=entry.block_start)
        if key != "CASE_PROPS":
            raise GSGMetadataError(f"Expected CASE_PROPS, got {key!r}")
        row_count = data[3]
        remaining = entry.block_end - file_obj.tell()
        if row_count < 0 or row_count * 16 > remaining:
            raise GSGMetadataError("CASE_PROPS block is not PROP-style metadata")
        for _ in range(row_count):
            read_struct(file_obj, "8si")
            name, payload, _, _ = read_keyword(file_obj, "4s2i")
            role = decode_token(payload[0])
            prop_number = payload[2]
            names, roles = grouped.setdefault(prop_number, ([], []))
            names.append(name)
            roles.append(role)
    return {
        number: (tuple(names), tuple(roles))
        for number, (names, roles) in grouped.items()
    }


#---------------------------------------------------------------------------------------------------
def read_property(file_obj, entry, names_by_prop, expand, shape):
#---------------------------------------------------------------------------------------------------
    """Read one indexed PROP block."""
    key, prop_data, _, _ = read_keyword(file_obj, "2i", offset=entry.block_start)
    if key != "PROP":
        raise GSGMetadataError(f"Expected PROP at byte {entry.block_start}, got {key!r}")

    encoding_code, prop_number = prop_data
    encoding = GSG_ENCODINGS.get(encoding_code)
    if encoding is None:
        raise UnsupportedGSGBlock(f"Unsupported PROP encoding {encoding_code}")

    alias, payload, _, _ = read_keyword(file_obj, "4sqi")
    metadata, dtype_code, size = payload
    if decode_token(metadata) != "ca":
        raise GSGMetadataError(f"Unexpected PROP metadata token {metadata!r}")
    dtype = GSG_DTYPES.get(dtype_code)
    if dtype is None:
        raise UnsupportedGSGBlock(f"Unsupported PROP dtype code {dtype_code}")

    values = read_property_values(file_obj, entry.block_end, dtype, encoding, size, expand)
    values = reshape_values(values, shape)
    names, roles = names_by_prop.get(prop_number, ((), ()))
    return GSGProperty(names, roles, alias, dtype, encoding, size, values)


#---------------------------------------------------------------------------------------------------
def read_property_values(file_obj, block_end, dtype, encoding, size, expand):
#---------------------------------------------------------------------------------------------------
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


#---------------------------------------------------------------------------------------------------
def write_prop_file(path, *properties, creator="PetrelForIx", version="2022.9.0", overwrite=False):
#---------------------------------------------------------------------------------------------------
    """Write a PROP GSG file and return the output path."""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    if not properties:
        raise ValueError("At least one GSGProperty is required")

    normalized = tuple(_normalize_property(property_) for property_ in properties)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with NamedTemporaryFile(
            "wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_obj:
            temp_path = Path(file_obj.name)
            _write_prop_payload(file_obj, normalized, creator, version)
        _validate_written_properties(temp_path, normalized)
        temp_path.replace(output_path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return output_path


#---------------------------------------------------------------------------------------------------
def active_property(grid):
#---------------------------------------------------------------------------------------------------
    """Return a writable ACTNUM property for a grid, if active values are present."""
    if grid.active is None:
        return None
    return GSGProperty.from_array(
        "ACTIVE_CELL_FLAG",
        "ACTNUM",
        grid.active,
        role="g",
        dtype=np.int32,
        encoding="full",
    )


#---------------------------------------------------------------------------------------------------
def write_property_values(file_obj, property_):
#---------------------------------------------------------------------------------------------------
    """Write full or RLE property values."""
    values = _flatten_property_values(property_)
    if property_.encoding == "full":
        if values.size != property_.size:
            raise ValueError(
                f"{property_.alias}: expected {property_.size} values, got {values.size}"
            )
        file_obj.write(np.asarray(values, dtype=property_.dtype).tobytes())
        return

    pairs = _property_rle_pairs(property_, values)
    file_obj.write(np.asarray((1,), dtype="<i4").tobytes())
    file_obj.write(pairs.tobytes())


#---------------------------------------------------------------------------------------------------
def write_case_props(file_obj, properties):
#---------------------------------------------------------------------------------------------------
    """Write a PROP-style CASE_PROPS block for the given properties."""
    row_count = sum(len(property_.names) for property_ in properties)
    write_keyword(file_obj, "CASE_PROPS", "i4s2i", (0, b"s   ", 0, row_count))
    for number, property_ in enumerate(properties, start=1):
        for name, role in zip(property_.names, property_.roles):
            file_obj.write(
                np.asarray([(b"ca  s   ", 0)], dtype=[("token", "S8"), ("value", "<i4")]).tobytes()
            )
            write_keyword(file_obj, name, "4s2i", (_token4(role, "role"), 1, number))


#---------------------------------------------------------------------------------------------------
def data_position(keyword, block_start):
#---------------------------------------------------------------------------------------------------
    """Return the index data-position value for a keyword block."""
    return block_start + 4 + len(keyword.encode("utf-8")) + 4


#---------------------------------------------------------------------------------------------------
def _normalize_property(property_):
#---------------------------------------------------------------------------------------------------
    """Return a validated GSGProperty instance."""
    if not isinstance(property_, GSGProperty):
        raise TypeError(f"Expected GSGProperty, got {type(property_).__name__}")
    dtype = property_dtype(property_.values, property_.dtype)
    encoding = normalize_encoding(property_.encoding)
    names = as_text_tuple(property_.names, "names")
    roles = as_text_tuple(property_.roles, "roles")
    if len(roles) == 1 and len(names) > 1:
        roles = len(names) * roles
    if len(roles) != len(names):
        raise ValueError(f"{property_.alias}: roles must match names")
    values = np.asarray(property_.values)
    size = int(property_.size or values.size)
    return GSGProperty(names, roles, property_.alias, dtype, encoding, size, values)


#---------------------------------------------------------------------------------------------------
def _flatten_property_values(property_):
#---------------------------------------------------------------------------------------------------
    """Return property values flattened in GSG/Fortran order."""
    values = np.asarray(property_.values)
    if values.dtype.names:
        return values
    return np.asarray(values, dtype=property_.dtype).ravel(order="F")


#---------------------------------------------------------------------------------------------------
def _rle_pairs(values):
#---------------------------------------------------------------------------------------------------
    """Return run-length encoded ``(count, value)`` pairs for one-dimensional values."""
    values = np.asarray(values)
    if values.size == 0:
        return np.empty(0, dtype=[("count", "<i4"), ("value", values.dtype)])
    starts = np.empty(values.size, dtype=bool)
    starts[0] = True
    starts[1:] = values[1:] != values[:-1]
    indices = np.flatnonzero(starts)
    counts = np.diff(np.append(indices, values.size))
    if counts.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError("RLE run length exceeds int32 range")
    pairs = np.empty(indices.size, dtype=[("count", "<i4"), ("value", values.dtype)])
    pairs["count"] = counts.astype("<i4")
    pairs["value"] = values[indices]
    return pairs


#---------------------------------------------------------------------------------------------------
def _write_prop_payload(file_obj, properties, creator, version):
#---------------------------------------------------------------------------------------------------
    """Write the complete PROP GSG payload to an open binary file object."""
    write_header(file_obj, creator, version)
    records = []
    for number, property_ in enumerate(properties, start=1):
        start = file_obj.tell()
        records.append(("PROP", ENCODING_CODES[property_.encoding], data_position("PROP", start)))
        write_keyword(file_obj, "PROP", "2i", (ENCODING_CODES[property_.encoding], number))
        dtype_code = DTYPE_CODES[property_.dtype]
        write_keyword(file_obj, property_.alias, "4sqi", (b"ca  ", dtype_code, property_.size))
        write_property_values(file_obj, property_)

    case_start = file_obj.tell()
    records.append(("CASE_PROPS", 0, data_position("CASE_PROPS", case_start)))
    write_case_props(file_obj, properties)

    index_pos = file_obj.tell()
    write_index_records(file_obj, records, index_pos)


#---------------------------------------------------------------------------------------------------
def _token4(value, label):
#---------------------------------------------------------------------------------------------------
    """Return a four-byte padded GSG token."""
    raw = str(value).encode("utf-8")
    if len(raw) > 4:
        raise ValueError(f"{label} token must be at most 4 bytes, got {value!r}")
    return raw.ljust(4, b" ")


#---------------------------------------------------------------------------------------------------
def _property_rle_pairs(property_, values):
#---------------------------------------------------------------------------------------------------
    """Return normalized RLE pairs for a property."""
    pair_dtype = np.dtype([("count", "<i4"), ("value", property_.dtype)])
    if values.dtype.names:
        if not {"count", "value"}.issubset(values.dtype.names):
            raise ValueError(f"{property_.alias}: RLE pairs require count and value fields")
        pairs = np.empty(values.size, dtype=pair_dtype)
        pairs["count"] = values["count"]
        pairs["value"] = values["value"]
    else:
        if values.size != property_.size:
            raise ValueError(
                f"{property_.alias}: expected {property_.size} values, got {values.size}"
            )
        pairs = _rle_pairs(np.asarray(values, dtype=property_.dtype)).astype(pair_dtype, copy=False)
    if pairs["count"].sum(dtype=np.int64) != property_.size:
        raise ValueError(f"{property_.alias}: RLE counts do not expand to {property_.size} values")
    return pairs


#---------------------------------------------------------------------------------------------------
def _validate_written_properties(path, expected):
#---------------------------------------------------------------------------------------------------
    """Validate that a written PROP file can be read back."""
    actual = tuple(read_properties(path, _index_entries(path)))
    if len(actual) != len(expected):
        raise GSGFormatError(f"Expected {len(expected)} properties, got {len(actual)}")
    for expected_property, actual_property in zip(expected, actual):
        if (
            actual_property.alias != expected_property.alias
            or actual_property.names != expected_property.names
            or actual_property.roles != expected_property.roles
            or actual_property.dtype != expected_property.dtype
            or actual_property.encoding != expected_property.encoding
            or actual_property.size != expected_property.size
        ):
            raise GSGFormatError(
                f"Written property metadata mismatch for {expected_property.alias}"
            )


#---------------------------------------------------------------------------------------------------
def _index_entries(path):
#---------------------------------------------------------------------------------------------------
    """Return public index entries for a GSG file."""
    with Path(path).open("rb") as file_obj:
        read_header(file_obj)
        return tuple(entry_from_record(record) for record in read_index_records(file_obj))
