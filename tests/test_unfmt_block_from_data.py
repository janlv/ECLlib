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
