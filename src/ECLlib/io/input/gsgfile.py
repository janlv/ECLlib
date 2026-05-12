"""Read Petrel/INTERSECT GSG property and grid files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

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
    write_header,
    write_index_records,
    write_keyword,
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
_DTYPE_CODES = {dtype: code for code, dtype in _GSG_DTYPES.items()}
_ENCODING_CODES = {encoding: code for code, encoding in _ENCODINGS.items()}


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
        names = _as_text_tuple(name, "name")
        roles = _as_text_tuple(role, "role")
        if len(roles) == 1 and len(names) > 1:
            roles = len(names) * roles
        if len(roles) != len(names):
            raise ValueError("role must be one value or match the number of names")

        dtype = _property_dtype(values, dtype)
        array = np.asarray(values, dtype=dtype)
        encoding = _normalize_encoding(encoding)
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
        egrid = _as_egrid_file(source)
        dimensions = tuple(int(value) for value in egrid.nijk())
        coord, zcorn = egrid.coord_zcorn()
        coord = np.asarray(coord, dtype=np.float32).reshape((-1, 6))
        zcorn = np.asarray(zcorn, dtype=np.float32)
        axes = _axes_from_egrid(egrid, coord)
        areal = _simple_areal(dimensions)
        pillars = _simple_pillars_from_egrid(coord, zcorn, dimensions)
        active_values = _active_from_egrid(egrid, dimensions, active)
        grid_name = _egrid_path(egrid).stem if name is None else str(name)
        return cls(grid_name, dimensions, axes, areal, pillars, active=active_values)


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


#---------------------------------------------------------------------------------------------------
def _as_text_tuple(value, label):
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
def _property_dtype(values, dtype=None):
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
    if candidate not in _DTYPE_CODES:
        raise TypeError(f"Unsupported GSG property dtype {candidate}")
    return candidate


#---------------------------------------------------------------------------------------------------
def _normalize_encoding(encoding):
#---------------------------------------------------------------------------------------------------
    """Return a supported property encoding name."""
    encoding = str(encoding).lower()
    if encoding not in _ENCODING_CODES:
        raise ValueError(f"Unsupported GSG property encoding {encoding!r}")
    return encoding


#---------------------------------------------------------------------------------------------------
def _normalize_property(property_):
#---------------------------------------------------------------------------------------------------
    """Return a validated GSGProperty instance."""
    if not isinstance(property_, GSGProperty):
        raise TypeError(f"Expected GSGProperty, got {type(property_).__name__}")
    dtype = _property_dtype(property_.values, property_.dtype)
    encoding = _normalize_encoding(property_.encoding)
    names = _as_text_tuple(property_.names, "names")
    roles = _as_text_tuple(property_.roles, "roles")
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
def _as_egrid_file(source):
#---------------------------------------------------------------------------------------------------
    """Return an ``EGRID_file`` instance for a path or existing reader."""
    from ..output import EGRID_file

    if isinstance(source, EGRID_file):
        return source
    return EGRID_file(source)


#---------------------------------------------------------------------------------------------------
def _egrid_path(egrid):
#---------------------------------------------------------------------------------------------------
    """Return the filesystem path represented by an EGRID reader."""
    for attribute in ("path", "file", "filename"):
        value = getattr(egrid, attribute, None)
        if value is not None:
            return Path(value)
    return Path(str(egrid))


#---------------------------------------------------------------------------------------------------
def _axes_from_egrid(egrid, coord):
#---------------------------------------------------------------------------------------------------
    """Return an AXES payload using EGRID map metadata when available."""
    units = next(egrid.blockdata("MAPUNITS"), None)
    unit_name = str(units).strip().upper() if units is not None else "METRES"
    token = b"f   m   f   " if unit_name == "METRES" else b"f   f   f   "
    mapaxes = next(egrid.blockdata("MAPAXES"), None)
    if mapaxes is not None:
        origin_x = float(mapaxes[2])
        origin_y = float(mapaxes[3])
    else:
        origin_x = float(np.min(coord[:, [0, 3]]))
        origin_y = float(np.min(coord[:, [1, 4]]))
    return (0, token, 6, 6, origin_x, origin_y, 0.0, 0.0, 1.0, 1.0)


#---------------------------------------------------------------------------------------------------
def _simple_areal(dimensions):
#---------------------------------------------------------------------------------------------------
    """Return AREAL rows for a structured grid with regular pillar indexing."""
    nx, ny, _ = dimensions
    rows = np.empty((nx * ny, 6), dtype="<i4")
    row = 0
    for j in range(ny):
        lower = j * (nx + 1)
        upper = lower + nx + 1
        for i in range(nx):
            rows[row] = (row, 4, upper + i, lower + i, lower + i + 1, upper + i + 1)
            row += 1
    return rows


#---------------------------------------------------------------------------------------------------
def _simple_pillars_from_egrid(coord, zcorn, dimensions, tolerance=1.0e-4):
#---------------------------------------------------------------------------------------------------
    """Return simple vertical-pillar GSG rows from EGRID COORD/ZCORN arrays."""
    nx, ny, nz = dimensions
    expected_coord = (nx + 1) * (ny + 1)
    if coord.shape != (expected_coord, 6):
        raise ValueError(f"Expected {expected_coord} COORD rows, got {coord.shape[0]}")
    expected_zcorn = nx * ny * nz * 8
    if zcorn.size != expected_zcorn:
        raise ValueError(f"Expected {expected_zcorn} ZCORN values, got {zcorn.size}")
    if (
        np.max(np.abs(coord[:, 0] - coord[:, 3]), initial=0.0) > tolerance
        or np.max(np.abs(coord[:, 1] - coord[:, 4]), initial=0.0) > tolerance
    ):
        raise ValueError("GSG grid writing currently supports vertical simple pillars only")

    coords = coord.reshape((ny + 1, nx + 1, 6))
    horizons = _pillar_horizons_from_zcorn(zcorn, dimensions)
    values = np.empty((expected_coord, nz + 6), dtype=np.float32)
    flat = values.reshape((ny + 1, nx + 1, nz + 6))
    flat[..., 0] = 1.0
    flat[..., 1] = coords[..., 0]
    flat[..., 2] = coords[..., 1]
    flat[..., 3] = coords[..., 2]
    flat[..., 4] = coords[..., 5]
    flat[..., 5:] = horizons
    header = (0, 0, expected_coord, nz + 1, 0, 0, -1, 0, 4, 4, 0, 1, 2, 3, 0, 0, 0)
    return GSGPillars(header, 0, values=values)


#---------------------------------------------------------------------------------------------------
def _pillar_horizons_from_zcorn(zcorn, dimensions):
#---------------------------------------------------------------------------------------------------
    """Return one averaged horizon depth per pillar and layer boundary."""
    nx, ny, nz = dimensions
    corners = np.asarray(zcorn, dtype=np.float32).reshape((nz, 2, ny, 2, nx, 2))
    horizons = np.empty((ny + 1, nx + 1, nz + 1), dtype=np.float32)
    for pillar_j in range(ny + 1):
        for pillar_i in range(nx + 1):
            for horizon in range(nz + 1):
                horizons[pillar_j, pillar_i, horizon] = _pillar_horizon_depth(
                    corners,
                    dimensions,
                    pillar_i,
                    pillar_j,
                    horizon,
                )
    return horizons


#---------------------------------------------------------------------------------------------------
def _pillar_horizon_depth(corners, dimensions, pillar_i, pillar_j, horizon):
#---------------------------------------------------------------------------------------------------
    """Return one averaged ZCORN horizon value for a structured pillar."""
    nx, ny, nz = dimensions
    if horizon == 0:
        layers = ((0, 0),)
    elif horizon == nz:
        layers = ((nz - 1, 1),)
    else:
        layers = ((horizon - 1, 1), (horizon, 0))

    values = []
    for cell_j in (pillar_j - 1, pillar_j):
        if cell_j < 0 or cell_j >= ny:
            continue
        local_j = 1 if cell_j == pillar_j - 1 else 0
        for cell_i in (pillar_i - 1, pillar_i):
            if cell_i < 0 or cell_i >= nx:
                continue
            local_i = 1 if cell_i == pillar_i - 1 else 0
            for layer, face in layers:
                values.append(corners[layer, face, cell_j, local_j, cell_i, local_i])
    if not values:
        raise ValueError(f"Could not derive horizon {horizon} for pillar {(pillar_i, pillar_j)}")
    return float(np.mean(values, dtype=np.float64))


#---------------------------------------------------------------------------------------------------
def _active_from_egrid(egrid, dimensions, active):
#---------------------------------------------------------------------------------------------------
    """Return an ACTNUM array from EGRID data or ``None`` when disabled."""
    if not active:
        return None
    actnum = next(egrid.blockdata("ACTNUM"), None)
    if actnum is None:
        return np.ones(dimensions, dtype="<i4")
    values = np.asarray(actnum, dtype="<i4")
    expected = int(np.prod(dimensions))
    if values.size != expected:
        raise ValueError(f"Expected {expected} ACTNUM values, got {values.size}")
    return values.reshape(dimensions, order="F")


#---------------------------------------------------------------------------------------------------
def _write_prop_file(path, *properties, creator="PetrelForIx", version="2022.9.0", overwrite=False):
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
def _write_prop_payload(file_obj, properties, creator, version):
#---------------------------------------------------------------------------------------------------
    """Write the complete PROP GSG payload to an open binary file object."""
    write_header(file_obj, creator, version)
    records = []
    for number, property_ in enumerate(properties, start=1):
        start = file_obj.tell()
        records.append(("PROP", _ENCODING_CODES[property_.encoding], _data_position("PROP", start)))
        write_keyword(file_obj, "PROP", "2i", (_ENCODING_CODES[property_.encoding], number))
        dtype_code = _DTYPE_CODES[property_.dtype]
        write_keyword(file_obj, property_.alias, "4sqi", (b"ca  ", dtype_code, property_.size))
        _write_property_values(file_obj, property_)

    case_start = file_obj.tell()
    records.append(("CASE_PROPS", 0, _data_position("CASE_PROPS", case_start)))
    _write_case_props(file_obj, properties)

    index_pos = file_obj.tell()
    write_index_records(file_obj, records, index_pos)


#---------------------------------------------------------------------------------------------------
def _write_grid_file(path, grid, creator="PetrelForIx", version="2022.9.0", overwrite=False):
#---------------------------------------------------------------------------------------------------
    """Write an AXES/grid GSG file and return the output path."""
    output_path = Path(path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(output_path)
    normalized = _normalize_grid(grid)

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
            _write_grid_payload(file_obj, normalized, creator, version)
        _validate_written_grid(temp_path, normalized)
        temp_path.replace(output_path)
    except Exception:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise
    return output_path


#---------------------------------------------------------------------------------------------------
def _write_grid_payload(file_obj, grid, creator, version):
#---------------------------------------------------------------------------------------------------
    """Write the complete AXES/grid GSG payload to an open binary file object."""
    write_header(file_obj, creator, version)
    records = []

    start = file_obj.tell()
    records.append(("AXES", 0, _data_position("AXES", start)))
    write_keyword(file_obj, "AXES", "i12s2i6f", grid.axes)

    start = file_obj.tell()
    records.append(("GRID", 0, _data_position("GRID", start)))
    _write_grid_case(file_obj, grid)

    start = file_obj.tell()
    records.append(("AREAL", 0, _data_position("AREAL", start)))
    _write_areal(file_obj, grid)

    start = file_obj.tell()
    records.append(("PILLARS", 0, _data_position("PILLARS", start)))
    _write_pillars(file_obj, grid)

    start = file_obj.tell()
    records.append(("PROP", 1, _data_position("PROP", start)))
    _write_defined_cells_prop(file_obj, grid)

    active_property = _active_property(grid)
    if active_property is None:
        start = file_obj.tell()
        records.append(("CASE_PROPS", 0, _data_position("CASE_PROPS", start)))
        _write_defined_cells_case_props(file_obj)
    else:
        start = file_obj.tell()
        records.append((
            "PROP",
            _ENCODING_CODES[active_property.encoding],
            _data_position("PROP", start),
        ))
        write_keyword(file_obj, "PROP", "2i", (_ENCODING_CODES[active_property.encoding], 1))
        write_keyword(
            file_obj,
            active_property.alias,
            "4sqi",
            (b"ca  ", _DTYPE_CODES[active_property.dtype], active_property.size),
        )
        _write_property_values(file_obj, active_property)

        start = file_obj.tell()
        records.append(("CASE_PROPS", 0, _data_position("CASE_PROPS", start)))
        _write_case_props(file_obj, (active_property,))

    index_pos = file_obj.tell()
    write_index_records(file_obj, records, index_pos)


#---------------------------------------------------------------------------------------------------
def _normalize_grid(grid):
#---------------------------------------------------------------------------------------------------
    """Return a validated writable ``GSGGrid``."""
    if not isinstance(grid, GSGGrid):
        raise TypeError(f"Expected GSGGrid, got {type(grid).__name__}")
    dimensions = tuple(int(value) for value in grid.dimensions)
    if len(dimensions) != 3 or any(value <= 0 for value in dimensions):
        raise ValueError("GSGGrid.dimensions must contain three positive integers")
    axes = _normalize_axes(grid.axes)
    areal = np.asarray(grid.areal, dtype="<i4") if grid.areal is not None else None
    expected_areal = dimensions[0] * dimensions[1]
    if areal is None or areal.shape != (expected_areal, 6):
        raise ValueError(f"GSGGrid.areal must have shape {(expected_areal, 6)}")
    if grid.pillars is None or grid.pillars.values is None:
        raise ValueError("GSGGrid.pillars with decoded values is required")
    if grid.pillars.grid_type != 0:
        raise ValueError("GSG grid writing currently supports simple pillar grids only")
    pillars = _normalize_pillars(grid.pillars, dimensions)
    active = _normalize_active(grid.active, dimensions)
    return GSGGrid(
        str(grid.name),
        dimensions,
        axes,
        areal,
        pillars,
        tuple(grid.faults),
        tuple(grid.defined_cells),
        active,
    )


#---------------------------------------------------------------------------------------------------
def _normalize_axes(axes):
#---------------------------------------------------------------------------------------------------
    """Return a normalized AXES payload tuple."""
    if len(axes) != 10:
        raise ValueError("GSGGrid.axes must contain ten AXES payload values")
    token = axes[1]
    if isinstance(token, str):
        token = token.encode("utf-8")
    token = token[:12].ljust(12, b" ")
    return (
        int(axes[0]),
        token,
        int(axes[2]),
        int(axes[3]),
        float(axes[4]),
        float(axes[5]),
        float(axes[6]),
        float(axes[7]),
        float(axes[8]),
        float(axes[9]),
    )


#---------------------------------------------------------------------------------------------------
def _normalize_pillars(pillars, dimensions):
#---------------------------------------------------------------------------------------------------
    """Return normalized simple PILLARS data."""
    nx, ny, nz = dimensions
    expected_rows = (nx + 1) * (ny + 1)
    expected_width = nz + 6
    values = np.asarray(pillars.values, dtype=np.float32)
    if values.shape != (expected_rows, expected_width):
        raise ValueError(f"GSGPillars.values must have shape {(expected_rows, expected_width)}")
    header = tuple(int(value) for value in pillars.header)
    if len(header) != 17:
        raise ValueError("GSGPillars.header must contain 17 integers")
    if header[2] != expected_rows or header[3] != nz + 1 or header[14] != 0:
        raise ValueError("GSGPillars.header does not match simple grid dimensions")
    return GSGPillars(header, 0, pillars.span, values)


#---------------------------------------------------------------------------------------------------
def _normalize_active(active, dimensions):
#---------------------------------------------------------------------------------------------------
    """Return normalized ACTNUM values or ``None``."""
    if active is None:
        return None
    values = np.asarray(active, dtype="<i4")
    expected = int(np.prod(dimensions))
    if values.size != expected:
        raise ValueError(f"GSGGrid.active must contain {expected} values")
    return values.reshape(dimensions, order="F")


#---------------------------------------------------------------------------------------------------
def _write_grid_case(file_obj, grid):
#---------------------------------------------------------------------------------------------------
    """Write GRID metadata and the following case keyword."""
    nx, ny, nz = grid.dimensions
    write_keyword(file_obj, "GRID", "2i", (0, 0))
    write_keyword(
        file_obj,
        grid.name,
        "5i4si4si4si",
        (-1, nx, ny, nz, 1, b"scor", 1, b"sp  ", 1, b"vp  ", 0),
    )


#---------------------------------------------------------------------------------------------------
def _write_areal(file_obj, grid):
#---------------------------------------------------------------------------------------------------
    """Write AREAL metadata and rows."""
    areal = np.asarray(grid.areal, dtype="<i4")
    nx, ny, _ = grid.dimensions
    npillars = (nx + 1) * (ny + 1)
    write_keyword(file_obj, "AREAL", "5i", (0, -1, 0, areal.shape[0], npillars - areal.shape[0]))
    file_obj.write(areal.tobytes())


#---------------------------------------------------------------------------------------------------
def _write_pillars(file_obj, grid):
#---------------------------------------------------------------------------------------------------
    """Write simple PILLARS metadata and rows."""
    pillars = grid.pillars
    write_keyword(file_obj, "PILLARS", "17i", pillars.header)
    values = np.asarray(pillars.values, dtype=np.float32)
    row_dtype = np.dtype([("pillar", "<i4"), ("values", "<f4", values.shape[1] - 1)])
    rows = np.empty(values.shape[0], dtype=row_dtype)
    rows["pillar"] = values[:, 0].astype("<i4")
    rows["values"] = values[:, 1:]
    file_obj.write(rows.tobytes())


#---------------------------------------------------------------------------------------------------
def _write_defined_cells_prop(file_obj, grid):
#---------------------------------------------------------------------------------------------------
    """Write the grid DEFINED_CELLS child block used by AXES files."""
    ncells = int(np.prod(grid.dimensions))
    write_keyword(file_obj, "PROP", "2i", (1, 0))
    write_keyword(file_obj, "DEFINED_CELLS", "4s6i", (b"c   ", 0, 0, ncells, 1, ncells, 1))


#---------------------------------------------------------------------------------------------------
def _write_defined_cells_case_props(file_obj):
#---------------------------------------------------------------------------------------------------
    """Write the grid CASE_PROPS/DEFINED_CELLS child block used by AXES files."""
    write_keyword(file_obj, "CASE_PROPS", "i4si8si", (0, b"s   ", 1, b"c   s   ", 0))
    write_keyword(file_obj, "DEFINED_CELLS", "4s3i", (b"g   ", 1, 0, 0))


#---------------------------------------------------------------------------------------------------
def _active_property(grid):
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
def _write_case_props(file_obj, properties):
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
def _data_position(keyword, block_start):
#---------------------------------------------------------------------------------------------------
    """Return the index data-position value for a keyword block."""
    return block_start + 4 + len(keyword.encode("utf-8")) + 4


#---------------------------------------------------------------------------------------------------
def _token4(value, label):
#---------------------------------------------------------------------------------------------------
    """Return a four-byte padded GSG token."""
    raw = str(value).encode("utf-8")
    if len(raw) > 4:
        raise ValueError(f"{label} token must be at most 4 bytes, got {value!r}")
    return raw.ljust(4, b" ")


#---------------------------------------------------------------------------------------------------
def _write_property_values(file_obj, property_):
#---------------------------------------------------------------------------------------------------
    """Write full or RLE property values."""
    values = _flatten_property_values(property_)
    if property_.encoding == "full":
        if values.size != property_.size:
            raise ValueError(f"{property_.alias}: expected {property_.size} values, got {values.size}")
        file_obj.write(np.asarray(values, dtype=property_.dtype).tobytes())
        return

    pairs = _property_rle_pairs(property_, values)
    file_obj.write(np.asarray((1,), dtype="<i4").tobytes())
    file_obj.write(pairs.tobytes())


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
            raise ValueError(f"{property_.alias}: expected {property_.size} values, got {values.size}")
        pairs = _rle_pairs(np.asarray(values, dtype=property_.dtype)).astype(pair_dtype, copy=False)
    if pairs["count"].sum(dtype=np.int64) != property_.size:
        raise ValueError(f"{property_.alias}: RLE counts do not expand to {property_.size} values")
    return pairs


#---------------------------------------------------------------------------------------------------
def _validate_written_properties(path, expected):
#---------------------------------------------------------------------------------------------------
    """Validate that a written PROP file can be read back."""
    actual = tuple(GSGFile(path).properties())
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
            raise GSGFormatError(f"Written property metadata mismatch for {expected_property.alias}")


#---------------------------------------------------------------------------------------------------
def _validate_written_grid(path, expected):
#---------------------------------------------------------------------------------------------------
    """Validate that a written AXES/grid file can be read back."""
    actual = GSGFile(path).grid(read_areal=True, read_pillars=True)
    if actual.name != expected.name or actual.dimensions != expected.dimensions:
        raise GSGFormatError(f"Written grid metadata mismatch for {expected.name}")
    if actual.areal is None or not np.array_equal(actual.areal, expected.areal):
        raise GSGFormatError(f"Written AREAL mismatch for {expected.name}")
    if actual.pillars is None or actual.pillars.values is None:
        raise GSGFormatError(f"Written PILLARS data missing for {expected.name}")
    if actual.pillars.header != expected.pillars.header:
        raise GSGFormatError(f"Written PILLARS metadata mismatch for {expected.name}")
    if not np.allclose(actual.pillars.values, expected.pillars.values):
        raise GSGFormatError(f"Written PILLARS values mismatch for {expected.name}")
    if expected.active is not None:
        if actual.active is None or not np.array_equal(actual.active, expected.active):
            raise GSGFormatError(f"Written ACTNUM mismatch for {expected.name}")


#===================================================================================================
class GSGFile:                                                                             # GSGFile
#===================================================================================================
    """Read Petrel/INTERSECT ``.GSG`` files and write PROP-style GSG files."""

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
    def read(                                                                              # GSGFile
        self,
        *,
        kind=None,
        name=None,
        expand=True,
        shape="auto",
        read_areal=True,
        read_pillars=False,
        read_faults=False,
    ):
    #-----------------------------------------------------------------------------------------------
        """Read GSG content using a format-aware convenience API."""
        read_kind = self.format.casefold() if kind is None else str(kind).casefold()
        if name is not None:
            if read_kind not in {"prop", "properties"}:
                raise GSGFormatError("name can only be used when reading PROP properties")
            if shape == "auto":
                try:
                    shape = self.dim()
                except GSGMetadataError:
                    shape = None
            return self.property(name, expand=expand, shape=shape)
        if read_kind in {"prop", "properties"}:
            if shape == "auto":
                try:
                    shape = self.dim()
                except GSGMetadataError:
                    shape = None
            return tuple(self.properties(expand=expand, shape=shape))
        if read_kind in {"axes", "grid"}:
            return self.grid(
                read_areal=read_areal,
                read_pillars=read_pillars,
                read_faults=read_faults,
            )
        if read_kind == "index":
            return self.index()
        if read_kind == "blocks":
            return tuple(self.blocks())
        raise ValueError(f"Unsupported GSG read kind {kind!r}")

    #-----------------------------------------------------------------------------------------------
    def dim(self):                                                                       # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Return grid dimensions from AXES metadata or the adjacent INTERSECT case."""
        if self.format == "AXES":
            return self.grid(read_areal=False, read_pillars=False).dimensions

        afi_files = tuple(self.path.parent.glob("*.afi"))
        matches = tuple(
            path for path in afi_files
            if self.path.stem == path.stem or self.path.stem.startswith(f"{path.stem}_")
        )
        if matches:
            afi_files = (max(matches, key=lambda path: len(path.stem)),)
        if len(afi_files) != 1:
            raise GSGMetadataError(
                f"Expected one matching AFI file beside {self.path}, found {len(afi_files)}"
            )

        from .intersect import IX_input

        try:
            dimensions = IX_input(afi_files[0]).dim()
        except Exception as exc:
            raise GSGMetadataError(f"Could not read grid dimensions from {afi_files[0]}") from exc
        if not dimensions:
            raise GSGMetadataError(f"Could not read grid dimensions from {afi_files[0]}")
        return tuple(int(value) for value in dimensions)

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

    @staticmethod
    #-----------------------------------------------------------------------------------------------
    def write_prop(                                                                       # GSGFile
        path,
        *properties,
        creator="PetrelForIx",
        version="2022.9.0",
        overwrite=False,
    ):
    #-----------------------------------------------------------------------------------------------
        """Write a PROP GSG file and return the output path."""
        return _write_prop_file(
            path,
            *properties,
            creator=creator,
            version=version,
            overwrite=overwrite,
        )

    @staticmethod
    #-----------------------------------------------------------------------------------------------
    def write_grid(                                                                       # GSGFile
        path,
        grid,
        *,
        creator="PetrelForIx",
        version="2022.9.0",
        overwrite=False,
    ):
    #-----------------------------------------------------------------------------------------------
        """Write an AXES/grid GSG file and return the output path."""
        return _write_grid_file(
            path,
            grid,
            creator=creator,
            version=version,
            overwrite=overwrite,
        )

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
            active = self._read_grid_active(file_obj, entries, dimensions)
        faults = fault_entries if read_faults else ()
        return GSGGrid(name, dimensions, axes, areal, pillars, faults, defined_cells, active)

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
            row_count = data[3]
            remaining = entry.block_end - file_obj.tell()
            if row_count < 0 or row_count * 16 > remaining:
                raise GSGMetadataError("CASE_PROPS block is not PROP-style metadata")
            for _ in range(row_count):
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
    def _read_grid_active(self, file_obj, entries, dimensions):                            # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Read an ACTNUM property embedded in an AXES file, if present."""
        try:
            names_by_prop = self._case_property_names()
        except GSGFormatError:
            return None
        for entry in entries:
            if entry.key != "PROP":
                continue
            try:
                prop = self._read_property(file_obj, entry, names_by_prop, True, None)
            except (GSGFormatError, ValueError):
                continue
            names = {prop.alias.casefold(), *(name.casefold() for name in prop.names)}
            if {"actnum", "active_cell_flag"} & names:
                if prop.values.size != int(np.prod(dimensions)):
                    continue
                return np.asarray(prop.values, dtype="<i4").reshape(dimensions, order="F")
        return None

    #-----------------------------------------------------------------------------------------------
    def _read_defined_cells(self, file_obj, entries):                                     # GSGFile
    #-----------------------------------------------------------------------------------------------
        """Read child DEFINED_CELLS records from compound grid spans."""
        cells = []
        for entry in entries:
            try:
                if entry.key == "PROP":
                    read_keyword(file_obj, "2i", offset=entry.block_start)
                    fmt = "4s6i"
                elif entry.key == "CASE_PROPS":
                    read_keyword(file_obj, "i4si8si", offset=entry.block_start)
                    fmt = "4s3i"
                else:
                    continue
            except GSGFormatError:
                continue
            if file_obj.tell() >= entry.block_end:
                continue
            position = file_obj.tell()
            try:
                key, _, _, _ = read_keyword(file_obj, offset=position)
            except GSGFormatError:
                continue
            if key == "DEFINED_CELLS":
                _, data, _, _ = read_keyword(file_obj, fmt, offset=position)
                cells.append(data)
        return tuple(cells)
