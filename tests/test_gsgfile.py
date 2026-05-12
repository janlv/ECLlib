import os
from dataclasses import replace
from pathlib import Path
from struct import pack

import numpy as np
import pytest

from ECLlib.input.gsg import GSGFile, GSGGrid, GSGProperty, UnsupportedGSGBlock

GSG_SAMPLE_DIR_ENV = "ECLLIB_GSG_SAMPLE_DIR"
EGRID_SAMPLE_FILE_ENV = "ECLLIB_EGRID_SAMPLE_FILE"
OPM_EGRID = Path("tests/Model_From_IXF_OPM/MODEL_FROM_IXF_OPM.EGRID")
_MAGIC = b"GSG000_b\r\n1\n2\r34\x01\x02\x03\x04"
_DTYPES = {
    0: np.dtype("<i4"),
    1: np.dtype("<f4"),
    2: np.dtype("<f8"),
}
_STRUCT_FORMATS = {
    0: "i",
    1: "f",
    2: "d",
}


#---------------------------------------------------------------------------------------------------
def _sample_dir():
#---------------------------------------------------------------------------------------------------
    """Return the optional local GSG sample directory."""
    sample_dir = os.environ.get(GSG_SAMPLE_DIR_ENV)
    if not sample_dir:
        pytest.skip(f"Set {GSG_SAMPLE_DIR_ENV} to run local GSG integration checks")
    path = Path(sample_dir)
    if not path.exists():
        pytest.skip(f"Sample GSG directory not available: {path}")
    return path


#---------------------------------------------------------------------------------------------------
def _sample_files(format_=None):
#---------------------------------------------------------------------------------------------------
    """Return optional local sample GSG files, filtered by first indexed keyword."""
    files = sorted(_sample_dir().glob("*.GSG"))
    if not files:
        pytest.skip(f"No GSG files found in {_sample_dir()}")
    if format_ is None:
        return files

    matched = tuple(path for path in files if GSGFile(path).format == format_)
    if not matched:
        pytest.skip(f"No {format_} GSG files found in {_sample_dir()}")
    return matched


#---------------------------------------------------------------------------------------------------
def _sample_property_payload(encoding):
#---------------------------------------------------------------------------------------------------
    """Return one optional local sample PROP path and properties for the requested encoding."""
    for path in _sample_files("PROP"):
        properties = GSGFile(path).read(shape=None)
        if any(prop.encoding == encoding for prop in properties):
            return path, properties
    pytest.skip(f"No {encoding} sample PROP file found in {_sample_dir()}")


#---------------------------------------------------------------------------------------------------
def _sample_egrid_file():
#---------------------------------------------------------------------------------------------------
    """Return the optional local EGRID sample file."""
    sample_file = os.environ.get(EGRID_SAMPLE_FILE_ENV)
    if not sample_file:
        pytest.skip(f"Set {EGRID_SAMPLE_FILE_ENV} to run local EGRID-to-GSG checks")
    path = Path(sample_file)
    if not path.exists():
        pytest.skip(f"Sample EGRID file not available: {path}")
    return path


#---------------------------------------------------------------------------------------------------
def _keyword(key, fmt="", values=()):
#---------------------------------------------------------------------------------------------------
    """Return one packed GSG keyword record."""
    raw_key = key.encode("utf-8")
    return pack("<i", len(raw_key)) + raw_key + (pack("<" + fmt, *values) if fmt else b"")


#---------------------------------------------------------------------------------------------------
def _header(creator="PetrelForIx", version="2022.9.0", extra=False):
#---------------------------------------------------------------------------------------------------
    """Return a packed GSG file header."""
    values = (
        1,
        1,
        len(creator),
        creator.encode("utf-8"),
        len(version),
        version.encode("utf-8"),
        0,
        0,
        0,
        1,
    )
    header = bytearray(_MAGIC + pack(f"<3i{len(creator)}si{len(version)}s4i", *values))
    if extra:
        header[-16:] = b""
        header.extend(_keyword("Windows (Build 10.0.17134)"))
        header.extend(_keyword("OKhan2"))
        header.extend(_keyword("0000-00-00 00:00:00"))
        header.extend(pack("<i", 1))
    return bytes(header)


#---------------------------------------------------------------------------------------------------
def _build_gsg(blocks, header=None):
#---------------------------------------------------------------------------------------------------
    """Build a complete GSG byte stream from indexed block records."""
    header = _header() if header is None else header
    payload = bytearray()
    records = []
    for index_key, value, block in blocks:
        block_start = len(header) + len(payload)
        data_pos = block_start + 4 + len(index_key.encode("utf-8")) + 4
        records.append((index_key, value, data_pos))
        payload.extend(block)

    index_pos = len(header) + len(payload)
    payload.extend(_keyword("INDEX", "2i", (0, len(records))))
    for index_key, value, data_pos in records:
        payload.extend(_keyword(index_key, "iq", (value, data_pos)))
    payload.extend(pack("<q", index_pos))
    return header + bytes(payload)


#---------------------------------------------------------------------------------------------------
def _case_props_block(items):
#---------------------------------------------------------------------------------------------------
    """Return a CASE_PROPS block for ``(name, role, property_number)`` rows."""
    block = bytearray(_keyword("CASE_PROPS", "i4s2i", (0, b"s   ", 0, len(items))))
    for name, role, property_number in items:
        block.extend(pack("<8si", b"ca  s   ", 0))
        role_bytes = role.encode("utf-8").ljust(4, b" ")
        block.extend(_keyword(name, "4s2i", (role_bytes, 1, property_number)))
    return bytes(block)


#---------------------------------------------------------------------------------------------------
def _prop_block(alias, dtype_code, encoding, values, property_number=1, size=None):
#---------------------------------------------------------------------------------------------------
    """Return one PROP block using either full values or RLE pairs."""
    dtype = _DTYPES[dtype_code]
    if encoding == 0:
        values = np.asarray(values, dtype=dtype)
        size = values.size if size is None else size
        payload = values.tobytes()
    else:
        pairs = list(values)
        size = sum(count for count, _ in pairs) if size is None else size
        value_fmt = _STRUCT_FORMATS[dtype_code]
        payload = bytearray(pack("<i", 1))
        for count, value in pairs:
            payload.extend(pack("<i" + value_fmt, count, value))
        payload = bytes(payload)
    return (
        _keyword("PROP", "2i", (encoding, property_number))
        + _keyword(alias, "4sqi", (b"ca  ", dtype_code, size))
        + payload
    )


#---------------------------------------------------------------------------------------------------
def _write_fixture(tmp_path, blocks, name="fixture.GSG"):
#---------------------------------------------------------------------------------------------------
    """Write a synthetic GSG fixture and return its path."""
    path = tmp_path / name
    path.write_bytes(_build_gsg(blocks))
    return path


#---------------------------------------------------------------------------------------------------
def _write_ix_case(tmp_path, case_name="CASE", dimensions=(2, 3, 1)):
#---------------------------------------------------------------------------------------------------
    """Write a minimal INTERSECT AFI/IXF case and return the AFI path."""
    afi_path = tmp_path / f"{case_name}.afi"
    ixf_path = tmp_path / f"{case_name}_grid.ixf"
    afi_path.write_text(
        f'SIMULATION ix "res1" {{\n  INCLUDE "{ixf_path.name}"\n}}\n',
        encoding="utf-8",
    )
    ixf_path.write_text(
        (
            'StructuredInfo "Grid" {\n'
            f"    NumberCellsInI={dimensions[0]}\n"
            f"    NumberCellsInJ={dimensions[1]}\n"
            f"    NumberCellsInK={dimensions[2]}\n"
            "}\n"
        ),
        encoding="utf-8",
    )
    return afi_path


#---------------------------------------------------------------------------------------------------
def _axes_block():
#---------------------------------------------------------------------------------------------------
    """Return a minimal AXES keyword block."""
    return _keyword("AXES", "i12s2i6f", (0, b"f   f   f   ", 6, 6, 0, 0, 0, 0, 1, 1))


#---------------------------------------------------------------------------------------------------
def _grid_block(name="GridOnly", dimensions=(3, 4, 5)):
#---------------------------------------------------------------------------------------------------
    """Return a minimal GRID keyword plus case-name record."""
    return _keyword("GRID", "2i", (0, 0)) + _keyword(name, "4i", (-1, *dimensions))


#---------------------------------------------------------------------------------------------------
def _assert_properties_match(actual, expected):
#---------------------------------------------------------------------------------------------------
    """Assert that two property sequences have identical metadata and values."""
    assert len(actual) == len(expected)
    for actual_prop, expected_prop in zip(actual, expected):
        assert actual_prop.names == expected_prop.names
        assert actual_prop.roles == expected_prop.roles
        assert actual_prop.alias == expected_prop.alias
        assert actual_prop.dtype == expected_prop.dtype
        assert actual_prop.encoding == expected_prop.encoding
        assert actual_prop.size == expected_prop.size
        np.testing.assert_array_equal(actual_prop.values, expected_prop.values)


#---------------------------------------------------------------------------------------------------
def _assert_generated_grid_matches(actual, expected):
#---------------------------------------------------------------------------------------------------
    """Assert that generated grid metadata and arrays survived a write/read cycle."""
    assert actual.name == expected.name
    assert actual.dimensions == expected.dimensions
    np.testing.assert_array_equal(actual.areal, expected.areal)
    assert actual.pillars.header == expected.pillars.header
    np.testing.assert_allclose(actual.pillars.values, expected.pillars.values)
    if expected.active is None:
        assert actual.active is None
    else:
        np.testing.assert_array_equal(actual.active, expected.active)


#---------------------------------------------------------------------------------------------------
def test_full_float_prop_preserves_float32(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Read full float PROP data without widening the dtype."""
    path = _write_fixture(
        tmp_path,
        (
            ("PROP", 0, _prop_block("PORO", 1, 0, [1.25, 2.5, 3.75])),
            ("CASE_PROPS", 0, _case_props_block((("POROSITY", "g", 1),))),
        ),
    )

    properties = GSGFile(path).read()
    prop = properties[0]

    assert len(properties) == 1
    assert prop.alias == "PORO"
    assert prop.names == ("POROSITY",)
    assert prop.roles == ("g",)
    assert prop.dtype == np.dtype("<f4")
    assert prop.values.dtype == np.dtype("<f4")
    np.testing.assert_allclose(prop.values, np.array([1.25, 2.5, 3.75], dtype="<f4"))


#---------------------------------------------------------------------------------------------------
def test_rle_float_prop_expands(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Expand RLE encoded float PROP data."""
    path = _write_fixture(
        tmp_path,
        (
            ("PROP", 1, _prop_block("NTG", 1, 1, ((3, 1.0), (2, 0.5)))),
            ("CASE_PROPS", 0, _case_props_block((("NET_TO_GROSS_RATIO", "g", 1),))),
        ),
    )

    prop = next(GSGFile(path).properties())

    assert prop.encoding == "rle"
    assert prop.values.dtype == np.dtype("<f4")
    np.testing.assert_allclose(prop.values, np.array([1, 1, 1, 0.5, 0.5], dtype="<f4"))


#---------------------------------------------------------------------------------------------------
def test_rle_int_prop_expands(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Expand RLE encoded integer PROP data."""
    path = _write_fixture(
        tmp_path,
        (
            ("PROP", 1, _prop_block("SATNUM", 0, 1, ((2, 7), (3, 9)))),
            (
                "CASE_PROPS",
                0,
                _case_props_block((("SATURATION_FUNCTION_DRAINAGE_TABLE_NO", "r", 1),)),
            ),
        ),
    )

    prop = next(GSGFile(path).properties())

    assert prop.values.dtype == np.dtype("<i4")
    np.testing.assert_array_equal(prop.values, np.array([7, 7, 9, 9, 9], dtype="<i4"))


#---------------------------------------------------------------------------------------------------
def test_multi_name_case_props_are_preserved(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Preserve all CASE_PROPS names pointing to one PROP payload."""
    path = _write_fixture(
        tmp_path,
        (
            ("PROP", 1, _prop_block("EOSNUM", 0, 1, ((4, 1),))),
            (
                "CASE_PROPS",
                0,
                _case_props_block(
                    (
                        ("EQUATION_OF_STATE_REGION", "r", 1),
                        ("PVT_REGION", "r", 1),
                        ("EQUILIBRATION_REGION", "r", 1),
                    )
                ),
            ),
        ),
    )

    prop = next(GSGFile(path).properties())

    assert prop.names == ("EQUATION_OF_STATE_REGION", "PVT_REGION", "EQUILIBRATION_REGION")
    assert prop.roles == ("r", "r", "r")
    assert GSGFile(path).read(name="PVT_REGION").alias == "EOSNUM"


#---------------------------------------------------------------------------------------------------
def test_ordered_index_preserves_repeated_keys(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Keep repeated index keys as ordered entries instead of overwriting them."""
    path = _write_fixture(
        tmp_path,
        (
            ("FAULTS", 0, _keyword("FAULTS", "i", (1,))),
            ("FAULTS", 0, _keyword("FAULTS", "i", (2,))),
            ("FAULTS", 0, _keyword("FAULTS", "i", (3,))),
        ),
    )

    entries = GSGFile(path).index()

    assert [entry.key for entry in entries] == ["FAULTS", "FAULTS", "FAULTS"]
    assert [entry.value for entry in entries] == [0, 0, 0]
    assert [entry.block_end - entry.block_start for entry in entries] == [14, 14, 14]


#---------------------------------------------------------------------------------------------------
def test_format_comes_from_index_after_extended_header(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Read older GSG headers with extra fields before the first indexed block."""
    path = tmp_path / "extended_header.GSG"
    path.write_bytes(
        _build_gsg(
            (
                ("AXES", 0, _axes_block()),
                ("GRID", 0, _grid_block()),
            ),
            header=_header("PetrelForIx", "2020.1 Preview", extra=True),
        )
    )

    gsg = GSGFile(path)
    grid = gsg.read()

    assert gsg.format == "AXES"
    assert grid.name == "GridOnly"
    assert grid.dimensions == (3, 4, 5)
    assert gsg.dim() == (3, 4, 5)


#---------------------------------------------------------------------------------------------------
def test_axes_grid_allows_missing_areal_and_pillars(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Read AXES files where geometry is represented by other indexed blocks."""
    path = _write_fixture(
        tmp_path,
        (
            ("AXES", 0, _axes_block()),
            ("GRID", 0, _grid_block("SparseGrid", (9, 8, 7))),
            ("PROP", 0, _prop_block("ACTNUM", 0, 0, [1, 1, 0], property_number=1)),
            ("CASE_PROPS", 0, _case_props_block((("ACTIVE_CELL_FLAG", "g", 1),))),
        ),
    )

    grid = GSGFile(path).read()

    assert grid.name == "SparseGrid"
    assert grid.dimensions == (9, 8, 7)
    assert grid.areal is None
    assert grid.pillars is None


#---------------------------------------------------------------------------------------------------
def test_prop_read_auto_shapes_from_adjacent_ix_case(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Automatically reshape PROP values from dimensions in the case AFI/IXF files."""
    _write_ix_case(tmp_path, "CASE", (2, 3, 1))
    _write_ix_case(tmp_path, "CASE_restart_1", (9, 9, 9))
    path = _write_fixture(
        tmp_path,
        (
            ("PROP", 0, _prop_block("PORO", 1, 0, np.arange(6, dtype="<f4"))),
            ("CASE_PROPS", 0, _case_props_block((("POROSITY", "g", 1),))),
        ),
        name="CASE_POROSITY.GSG",
    )

    gsg = GSGFile(path)
    prop = gsg.read()[0]
    raw = gsg.read(shape=None)[0]

    assert gsg.dim() == (2, 3, 1)
    assert prop.values.shape == (2, 3, 1)
    np.testing.assert_array_equal(prop.values.ravel(order="F"), np.arange(6, dtype="<f4"))
    assert raw.values.shape == (6,)


#---------------------------------------------------------------------------------------------------
def test_write_full_float32_property(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Write and read a full float32 PROP file."""
    output = tmp_path / "poro.GSG"
    prop = GSGProperty.from_array("POROSITY", "PORO", [1.25, 2.5, 3.75])

    GSGFile.write_prop(output, prop)
    actual = GSGFile(output).read()

    _assert_properties_match(actual, (prop,))
    assert actual[0].values.dtype == np.dtype("<f4")


#---------------------------------------------------------------------------------------------------
def test_write_full_int32_property(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Write and read a full int32 PROP file."""
    output = tmp_path / "satnum.GSG"
    prop = GSGProperty.from_array("SATURATION_FUNCTION_DRAINAGE_TABLE_NO", "SATNUM", [1, 2, 2, 3])

    GSGFile.write_prop(output, prop)
    actual = GSGFile(output).read()

    _assert_properties_match(actual, (prop,))
    assert actual[0].values.dtype == np.dtype("<i4")


#---------------------------------------------------------------------------------------------------
def test_write_full_float64_property(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Write and read an explicit float64 PROP file."""
    output = tmp_path / "double.GSG"
    values = np.array([1.0, 1.0 + 1e-12], dtype=np.float64)
    prop = GSGProperty.from_array("DOUBLE_PROPERTY", "DBL", values, dtype=np.float64)

    GSGFile.write_prop(output, prop)
    actual = GSGFile(output).read()

    _assert_properties_match(actual, (prop,))
    assert actual[0].values.dtype == np.dtype("<f8")


#---------------------------------------------------------------------------------------------------
def test_write_rle_float_and_int_properties(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Write and read RLE encoded float and integer properties."""
    output = tmp_path / "rle.GSG"
    float_prop = GSGProperty.from_array(
        "NET_TO_GROSS_RATIO",
        "NTG",
        [1.0, 1.0, 1.0, 0.5, 0.5],
        encoding="rle",
    )
    int_prop = GSGProperty.from_array(
        "SATURATION_FUNCTION_DRAINAGE_TABLE_NO",
        "SATNUM",
        [7, 7, 9, 9, 9],
        role="r",
        encoding="rle",
    )

    GSGFile.write_prop(output, float_prop, int_prop)
    actual = GSGFile(output).read()

    _assert_properties_match(actual, (float_prop, int_prop))


#---------------------------------------------------------------------------------------------------
def test_write_multi_name_property_metadata(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Write and read CASE_PROPS metadata with several names for one property."""
    output = tmp_path / "regions.GSG"
    prop = GSGProperty.from_array(
        ("EQUATION_OF_STATE_REGION", "PVT_REGION", "EQUILIBRATION_REGION"),
        "EOSNUM",
        [1, 1, 1, 1],
        role=("r", "r", "r"),
        encoding="rle",
    )

    GSGFile.write_prop(output, prop)
    actual = GSGFile(output).read()

    _assert_properties_match(actual, (prop,))


#---------------------------------------------------------------------------------------------------
def test_write_refuses_existing_file_without_overwrite(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Protect existing files unless overwrite is explicitly requested."""
    output = tmp_path / "exists.GSG"
    output.write_bytes(b"already here")
    prop = GSGProperty.from_array("POROSITY", "PORO", [0.25])

    with pytest.raises(FileExistsError):
        GSGFile.write_prop(output, prop)

    GSGFile.write_prop(output, prop, overwrite=True)
    assert GSGFile(output).property("PORO").values[0] == np.float32(0.25)


#---------------------------------------------------------------------------------------------------
def test_write_prop_refuses_missing_properties(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Require at least one property when writing a PROP file."""
    with pytest.raises(ValueError):
        GSGFile.write_prop(tmp_path / "empty.GSG")


#---------------------------------------------------------------------------------------------------
def test_write_gsg_properties_is_not_public_api():
#---------------------------------------------------------------------------------------------------
    """Keep helper-style write functions out of the public package API."""
    import ECLlib
    import ECLlib.input as input_api
    import ECLlib.input.gsg as gsg_api

    assert not hasattr(ECLlib, "write_gsg_properties")
    assert not hasattr(input_api, "write_gsg_properties")
    assert "write_gsg_properties" not in gsg_api.__all__


#---------------------------------------------------------------------------------------------------
def test_write_prop_round_trips_prop_file(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Round-trip a PROP file through the class writer method."""
    source = tmp_path / "source.GSG"
    target = tmp_path / "target.GSG"
    prop = GSGProperty.from_array("POROSITY", "PORO", [0.2, 0.3, 0.4, 0.5])
    GSGFile.write_prop(source, prop)

    GSGFile.write_prop(target, *GSGFile(source).read())

    _assert_properties_match(GSGFile(target).read(), (prop,))


#---------------------------------------------------------------------------------------------------
def test_grid_from_egrid_fixture_builds_simple_geometry():
#---------------------------------------------------------------------------------------------------
    """Create a writable simple GSG grid from a bundled EGRID fixture."""
    grid = GSGGrid.from_egrid(OPM_EGRID)

    assert grid.name == OPM_EGRID.stem
    assert grid.dimensions == (25, 25, 5)
    assert grid.areal.shape == (25 * 25, 6)
    assert grid.pillars.grid_type == 0
    assert grid.pillars.header == (0, 0, 26 * 26, 6, 0, 0, -1, 0, 4, 4, 0, 1, 2, 3, 0, 0, 0)
    assert grid.pillars.values.shape == (26 * 26, 11)
    assert grid.active.shape == (25, 25, 5)
    assert int(grid.active.sum()) == 25 * 25 * 5


#---------------------------------------------------------------------------------------------------
def test_write_grid_from_egrid_round_trips(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Write an AXES/grid GSG file from EGRID-derived geometry."""
    output = tmp_path / "grid.GSG"
    expected = GSGGrid.from_egrid(OPM_EGRID, name="GeneratedGrid")

    GSGFile.write_grid(output, expected)
    actual = GSGFile(output).read(read_pillars=True)

    assert GSGFile(output).format == "AXES"
    _assert_generated_grid_matches(actual, expected)
    assert [entry.key for entry in GSGFile(output).index()] == [
        "AXES",
        "GRID",
        "AREAL",
        "PILLARS",
        "PROP",
        "PROP",
        "CASE_PROPS",
    ]
    assert actual.defined_cells == ((b"c   ", 0, 0, 25 * 25 * 5, 1, 25 * 25 * 5, 1),)


#---------------------------------------------------------------------------------------------------
def test_write_grid_preserves_inactive_actnum(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Preserve inactive ACTNUM cells as an embedded grid property."""
    output = tmp_path / "grid_actnum.GSG"
    active = GSGGrid.from_egrid(OPM_EGRID).active.copy()
    active.ravel(order="F")[::7] = 0
    expected = replace(GSGGrid.from_egrid(OPM_EGRID), active=active)

    GSGFile.write_grid(output, expected)
    actual = GSGFile(output).read(read_pillars=True)

    _assert_generated_grid_matches(actual, expected)
    assert int(actual.active.sum()) == int(active.sum())


#---------------------------------------------------------------------------------------------------
def test_write_grid_refuses_existing_file_without_overwrite(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Protect existing grid files unless overwrite is explicitly requested."""
    output = tmp_path / "exists.GSG"
    output.write_bytes(b"already here")
    grid = GSGGrid.from_egrid(OPM_EGRID)

    with pytest.raises(FileExistsError):
        GSGFile.write_grid(output, grid)

    GSGFile.write_grid(output, grid, overwrite=True)
    assert GSGFile(output).read().dimensions == grid.dimensions


#---------------------------------------------------------------------------------------------------
def test_write_grid_refuses_incomplete_grid(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Require generated AREAL and simple PILLARS arrays when writing grids."""
    grid = GSGGrid("Incomplete", (1, 1, 1), (0, b"f   m   f   ", 6, 6, 0, 0, 0, 0, 1, 1))

    with pytest.raises(ValueError, match="areal"):
        GSGFile.write_grid(tmp_path / "bad.GSG", grid)


#---------------------------------------------------------------------------------------------------
def test_sample_property_files_have_expected_cell_count():
#---------------------------------------------------------------------------------------------------
    """Read all available local sample property files."""
    for path in _sample_files("PROP"):
        for prop in GSGFile(path).read(shape=None):
            assert prop.size > 0
            assert prop.values.size == prop.size
            assert prop.values.dtype in {np.dtype("<i4"), np.dtype("<f4"), np.dtype("<f8")}


#---------------------------------------------------------------------------------------------------
def test_sample_axes_grid_metadata_and_fault_spans():
#---------------------------------------------------------------------------------------------------
    """Read local sample AXES metadata without requiring bundled case data."""
    for path in _sample_files("AXES"):
        gsg = GSGFile(path)
        grid = gsg.grid(read_areal=True, read_pillars=False, read_faults=True)

        assert gsg.format == "AXES"
        assert grid.name
        assert len(grid.dimensions) == 3
        assert all(value > 0 for value in grid.dimensions)
        if grid.areal is not None:
            assert grid.areal.ndim == 2
            assert grid.areal.shape[1] == 6
        if grid.pillars is not None:
            assert grid.pillars.values is None
            if grid.pillars.grid_type != 0:
                with pytest.raises(UnsupportedGSGBlock):
                    gsg.grid(read_areal=False, read_pillars=True)
        assert len(grid.faults) == sum(entry.key == "FAULTS" for entry in gsg.index())


#---------------------------------------------------------------------------------------------------
def test_sample_full_property_round_trips_through_writer(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Round-trip a real local full-array property file through the PROP writer."""
    source, expected = _sample_property_payload("full")
    target = tmp_path / source.name

    GSGFile.write_prop(target, *expected)
    actual = GSGFile(target).read(shape=None)

    _assert_properties_match(actual, expected)


#---------------------------------------------------------------------------------------------------
def test_sample_rle_property_round_trips_through_writer(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Round-trip a real local RLE property file through the PROP writer."""
    source, expected = _sample_property_payload("rle")
    target = tmp_path / source.name

    GSGFile.write_prop(target, *expected)
    actual = GSGFile(target).read(shape=None)

    _assert_properties_match(actual, expected)


#---------------------------------------------------------------------------------------------------
def test_sample_egrid_file_writes_grid_gsg(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Optionally generate a GSG grid from a local EGRID file."""
    source = _sample_egrid_file()
    try:
        expected = GSGGrid.from_egrid(source)
    except ValueError as exc:
        pytest.skip(f"Local EGRID geometry is not supported by the simple grid writer: {exc}")
    target = tmp_path / f"{source.stem}.GSG"

    GSGFile.write_grid(target, expected)
    actual = GSGFile(target).read(read_pillars=True)

    assert actual.dimensions == expected.dimensions
    assert actual.pillars.values.shape == expected.pillars.values.shape
    if expected.active is not None:
        assert int(actual.active.sum()) == int(expected.active.sum())
