from __future__ import annotations

import numpy as np
import pytest

from ECLlib import unfmt_block, unfmt_file


#---------------------------------------------------------------------------------------------------
def _write_blocks(path, blocks):
#---------------------------------------------------------------------------------------------------
    """Write serialized unformatted blocks to a binary file."""
    with open(path, "wb") as file:
        for key, data, dtype in blocks:
            file.write(unfmt_block.from_data(key, data, dtype).as_bytes())


#---------------------------------------------------------------------------------------------------
def _first_block(path, *, use_mmap=True):
#---------------------------------------------------------------------------------------------------
    """Return the first block together with its live iterator."""
    blocks = unfmt_file(path).blocks(use_mmap=use_mmap)
    return blocks, next(blocks)


#---------------------------------------------------------------------------------------------------
def test_unfmt_block_from_data_normalizes_fortran_order():
#---------------------------------------------------------------------------------------------------
    """Flatten multidimensional payloads using Eclipse/Fortran ordering."""
    raw = unfmt_block.from_data("SATNUM", [[1, 2], [3, 4]], "int").as_bytes()

    assert np.frombuffer(raw[28:-4], dtype=">i4").tolist() == [1, 3, 2, 4]


#---------------------------------------------------------------------------------------------------
def test_unfmt_block_from_data_truncates_long_names():
#---------------------------------------------------------------------------------------------------
    """Truncate long block keywords to Eclipse's 8-character limit."""
    raw = unfmt_block.from_data("anhydrite", [1], "int").as_bytes()

    assert raw[4:12] == b"anhydrit"


#---------------------------------------------------------------------------------------------------
def test_unfmt_block_from_data_validates_keyword_and_payload_constraints():
#---------------------------------------------------------------------------------------------------
    """Reject unsupported payloads."""
    with pytest.raises(ValueError, match="<= 8 bytes"):
        unfmt_block.from_data("NAME", ["123456789"], "char")

    with pytest.raises(ValueError, match="MESS blocks must use an empty payload"):
        unfmt_block.from_data("STARTSOL", [1], "mess")


#---------------------------------------------------------------------------------------------------
def test_unfmt_block_bool_payload_uses_4_byte_logicals():
#---------------------------------------------------------------------------------------------------
    """Encode boolean payloads using 4-byte Eclipse logical values."""
    raw = unfmt_block.from_data("FLAG", [True, False, True], "bool").as_bytes()

    assert len(raw) == 44
    assert int.from_bytes(raw[24:28], "big", signed=True) == 12
    assert np.frombuffer(raw[28:-4], dtype=">i4").tolist() == [1, 0, 1]
    assert int.from_bytes(raw[-4:], "big", signed=True) == 12


#---------------------------------------------------------------------------------------------------
def test_unfmt_block_from_data_roundtrip_for_supported_dtypes(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Round-trip all supported block dtypes through an unformatted file."""
    path = tmp_path / "blocks.bin"
    blocks = [
        ("INTVAL", [1, -2], "int"),
        ("REALS", [[1.0, 2.0], [3.0, 4.0]], "float"),
        ("DOUBLES", [1.5, 2.5], "double"),
        ("FLAG", [True, False, True], "bool"),
        ("WORDS", ["AA", "BB"], "char"),
        ("EMPTY", [], "mess"),
        ("TAIL", [99], "int"),
    ]
    _write_blocks(path, blocks)

    data = {}
    for block in unfmt_file(path).blocks(use_mmap=False):
        if block.key() == "WORDS":
            data[block.key()] = block.data(strip=True).tolist()
        else:
            data[block.key()] = block.data().tolist()

    assert data["INTVAL"] == [1, -2]
    assert data["REALS"] == pytest.approx([1.0, 3.0, 2.0, 4.0])
    assert data["DOUBLES"] == pytest.approx([1.5, 2.5])
    assert data["FLAG"] == [True, False, True]
    assert data["WORDS"] == ["AA", "BB"]
    assert data["EMPTY"] == []
    assert data["TAIL"] == [99]


#---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_mmap", [True, False])
def test_unfmt_block_read_into_matches_numeric_data(tmp_path, use_mmap):
#---------------------------------------------------------------------------------------------------
    """Decode numeric payloads directly into caller-owned buffers."""
    path = tmp_path / "numeric.bin"
    _write_blocks(path, [("SATNUM", np.arange(6, dtype=np.int32), "int")])

    blocks, block = _first_block(path, use_mmap=use_mmap)
    out = np.empty(6, dtype=np.int32)
    block.read_into(out)

    assert np.array_equal(out, block.data())
    assert next(blocks, None) is None


#---------------------------------------------------------------------------------------------------
def test_unfmt_block_read_into_handles_multi_payload_blocks(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Decode blocks spanning multiple Eclipse payload chunks without joining all bytes first."""
    path = tmp_path / "long.bin"
    values = np.arange(1505, dtype=np.float32)
    _write_blocks(path, [("LONGREAL", values, "float")])

    blocks, block = _first_block(path)
    out = np.empty((5, 7, 43), dtype=np.float32, order="F")
    block.read_into(out)

    assert np.array_equal(out.ravel(order="F"), block.data())
    assert next(blocks, None) is None


#---------------------------------------------------------------------------------------------------
def test_unfmt_block_read_into_decodes_logical_payloads_to_bool(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Decode logical payloads directly into boolean output arrays."""
    path = tmp_path / "logical.bin"
    values = np.asarray([True, False, True, True], dtype=bool)
    _write_blocks(path, [("LOGFLAG", values, "bool")])

    blocks, block = _first_block(path)
    out = np.empty((2, 2), dtype=bool, order="F")
    block.read_into(out)

    assert out.dtype == bool
    assert np.array_equal(out.ravel(order="F"), block.data())
    assert next(blocks, None) is None
