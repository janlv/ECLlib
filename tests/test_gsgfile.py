from pathlib import Path
from struct import pack

import numpy as np
import pytest

from ECLlib.io.input.gsgfile import GSGFile, UnsupportedGSGBlock

SAMPLE_DIR = Path("/home/AD.NORCERESEARCH.NO/javi/ekofisk/input/IORSIM_BASECASE_SURF_RES")
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

    prop = next(GSGFile(path).properties())

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
    assert GSGFile(path).property("PVT_REGION").alias == "EOSNUM"


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
    grid = gsg.grid()

    assert gsg.format == "AXES"
    assert grid.name == "GridOnly"
    assert grid.dimensions == (3, 4, 5)


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

    grid = GSGFile(path).grid()

    assert grid.name == "SparseGrid"
    assert grid.dimensions == (9, 8, 7)
    assert grid.areal is None
    assert grid.pillars is None


#---------------------------------------------------------------------------------------------------
def test_sample_property_files_have_expected_cell_count():
#---------------------------------------------------------------------------------------------------
    """Read all available Ekofisk sample property files."""
    if not SAMPLE_DIR.exists():
        pytest.skip(f"Sample GSG directory not available: {SAMPLE_DIR}")

    for path in sorted(SAMPLE_DIR.glob("*.GSG")):
        if path.name == "IORSIM_BASECASE_INPUT.GSG":
            continue
        for prop in GSGFile(path).properties():
            assert prop.size == 1284780
            assert prop.values.size == 1284780
            assert prop.values.dtype in {np.dtype("<i4"), np.dtype("<f4"), np.dtype("<f8")}


#---------------------------------------------------------------------------------------------------
def test_sample_axes_grid_metadata_and_fault_spans():
#---------------------------------------------------------------------------------------------------
    """Read Ekofisk AXES metadata without decoding unsupported complex pillars."""
    path = SAMPLE_DIR / "IORSIM_BASECASE_INPUT.GSG"
    if not path.exists():
        pytest.skip(f"Sample AXES GSG file not available: {path}")

    gsg = GSGFile(path)
    grid = gsg.grid(read_areal=True, read_pillars=False, read_faults=True)

    assert gsg.format == "AXES"
    assert grid.name == "FlankMidCase"
    assert grid.dimensions == (210, 266, 23)
    assert grid.areal.shape == (55860, 6)
    assert grid.pillars is not None
    assert grid.pillars.grid_type == 1
    assert grid.pillars.values is None
    assert len(grid.faults) == 42
    assert sum(entry.key == "FAULTS" for entry in gsg.index()) == 42
    with pytest.raises(UnsupportedGSGBlock):
        gsg.grid(read_areal=False, read_pillars=True)
