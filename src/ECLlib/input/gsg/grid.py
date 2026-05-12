"""AXES/grid read/write helpers for Petrel/INTERSECT GSG files."""
from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .binary import (
    GSGFormatError,
    GSGMetadataError,
    UnsupportedGSGBlock,
    read_header,
    read_index_records,
    read_keyword,
    write_header,
    write_index_records,
    write_keyword,
)
from .prop import (
    active_property,
    case_property_names,
    data_position,
    read_property,
    write_case_props,
    write_property_values,
)
from .types import GSGGrid, GSGPillars, entry_from_record

__all__ = [
    "grid_from_egrid",
    "read_grid",
    "write_grid_file",
]


#---------------------------------------------------------------------------------------------------
def grid_from_egrid(grid_cls, source, *, name=None, active=True):
#---------------------------------------------------------------------------------------------------
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
    return grid_cls(grid_name, dimensions, axes, areal, pillars, active=active_values)


#---------------------------------------------------------------------------------------------------
def read_grid(path, entries, read_areal=True, read_pillars=False, read_faults=False):
#---------------------------------------------------------------------------------------------------
    """Return decoded AXES grid metadata and optional payload arrays."""
    entry_by_key = {entry.key: entry for entry in entries if entry.key not in {"FAULTS"}}
    fault_entries = tuple(entry for entry in entries if entry.key == "FAULTS")
    with Path(path).open("rb") as file_obj:
        axes = _read_axes(file_obj, entry_by_key["AXES"])
        name, dimensions = _read_grid_case(file_obj, entry_by_key["GRID"])
        areal = None
        if read_areal and "AREAL" in entry_by_key:
            areal = _read_areal(file_obj, entry_by_key["AREAL"])
        pillars = None
        if "PILLARS" in entry_by_key:
            pillars = _read_pillars(file_obj, entry_by_key["PILLARS"], read_pillars)
        defined_cells = _read_defined_cells(file_obj, entries)
        active = _read_grid_active(file_obj, path, entries, dimensions)
    faults = fault_entries if read_faults else ()
    return GSGGrid(name, dimensions, axes, areal, pillars, faults, defined_cells, active)


#---------------------------------------------------------------------------------------------------
def write_grid_file(path, grid, creator="PetrelForIx", version="2022.9.0", overwrite=False):
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
def _as_egrid_file(source):
#---------------------------------------------------------------------------------------------------
    """Return an ``EGRID_file`` instance for a path or existing reader."""
    from ...output import EGRID_file

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
def _write_grid_payload(file_obj, grid, creator, version):
#---------------------------------------------------------------------------------------------------
    """Write the complete AXES/grid GSG payload to an open binary file object."""
    write_header(file_obj, creator, version)
    records = []

    start = file_obj.tell()
    records.append(("AXES", 0, data_position("AXES", start)))
    write_keyword(file_obj, "AXES", "i12s2i6f", grid.axes)

    start = file_obj.tell()
    records.append(("GRID", 0, data_position("GRID", start)))
    _write_grid_case(file_obj, grid)

    start = file_obj.tell()
    records.append(("AREAL", 0, data_position("AREAL", start)))
    _write_areal(file_obj, grid)

    start = file_obj.tell()
    records.append(("PILLARS", 0, data_position("PILLARS", start)))
    _write_pillars(file_obj, grid)

    start = file_obj.tell()
    records.append(("PROP", 1, data_position("PROP", start)))
    _write_defined_cells_prop(file_obj, grid)

    actnum = active_property(grid)
    if actnum is None:
        start = file_obj.tell()
        records.append(("CASE_PROPS", 0, data_position("CASE_PROPS", start)))
        _write_defined_cells_case_props(file_obj)
    else:
        start = file_obj.tell()
        records.append(("PROP", 0, data_position("PROP", start)))
        write_keyword(file_obj, "PROP", "2i", (0, 1))
        write_keyword(file_obj, actnum.alias, "4sqi", (b"ca  ", 0, actnum.size))
        write_property_values(file_obj, actnum)

        start = file_obj.tell()
        records.append(("CASE_PROPS", 0, data_position("CASE_PROPS", start)))
        write_case_props(file_obj, (actnum,))

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
def _read_axes(file_obj, entry):
#---------------------------------------------------------------------------------------------------
    """Read the AXES payload."""
    key, data, _, _ = read_keyword(file_obj, "i12s2i6f", offset=entry.block_start)
    if key != "AXES":
        raise GSGMetadataError(f"Expected AXES, got {key!r}")
    return data


#---------------------------------------------------------------------------------------------------
def _read_grid_case(file_obj, entry):
#---------------------------------------------------------------------------------------------------
    """Read GRID metadata and the following case-name keyword."""
    key, _, _, _ = read_keyword(file_obj, "2i", offset=entry.block_start)
    if key != "GRID":
        raise GSGMetadataError(f"Expected GRID, got {key!r}")
    case_name, data, _, _ = read_keyword(file_obj, "4i")
    dimensions = tuple(int(value) for value in data[1:4])
    return case_name, dimensions


#---------------------------------------------------------------------------------------------------
def _read_areal(file_obj, entry):
#---------------------------------------------------------------------------------------------------
    """Read AREAL rows as an ``int32`` array."""
    key, data, _, _ = read_keyword(file_obj, "5i", offset=entry.block_start)
    if key != "AREAL":
        raise GSGMetadataError(f"Expected AREAL, got {key!r}")
    row_count = data[3]
    values = np.fromfile(file_obj, dtype=np.dtype("<i4"), count=row_count * 6)
    if values.size != row_count * 6:
        raise GSGFormatError(f"Expected {row_count * 6} AREAL integers, got {values.size}")
    return values.reshape((row_count, 6))


#---------------------------------------------------------------------------------------------------
def _read_pillars(file_obj, entry, read_values):
#---------------------------------------------------------------------------------------------------
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


#---------------------------------------------------------------------------------------------------
def _read_grid_active(file_obj, path, entries, dimensions):
#---------------------------------------------------------------------------------------------------
    """Read an ACTNUM property embedded in an AXES file, if present."""
    try:
        names_by_prop = case_property_names(path, entries)
    except GSGFormatError:
        return None
    for entry in entries:
        if entry.key != "PROP":
            continue
        try:
            prop = read_property(file_obj, entry, names_by_prop, True, None)
        except (GSGFormatError, ValueError):
            continue
        names = {prop.alias.casefold(), *(name.casefold() for name in prop.names)}
        if {"actnum", "active_cell_flag"} & names:
            if prop.values.size != int(np.prod(dimensions)):
                continue
            return np.asarray(prop.values, dtype="<i4").reshape(dimensions, order="F")
    return None


#---------------------------------------------------------------------------------------------------
def _read_defined_cells(file_obj, entries):
#---------------------------------------------------------------------------------------------------
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


#---------------------------------------------------------------------------------------------------
def _validate_written_grid(path, expected):
#---------------------------------------------------------------------------------------------------
    """Validate that a written AXES/grid file can be read back."""
    actual = read_grid(path, _index_entries(path), read_areal=True, read_pillars=True)
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


#---------------------------------------------------------------------------------------------------
def _index_entries(path):
#---------------------------------------------------------------------------------------------------
    """Return public index entries for a GSG file."""
    with Path(path).open("rb") as file_obj:
        read_header(file_obj)
        return tuple(entry_from_record(record) for record in read_index_records(file_obj))
