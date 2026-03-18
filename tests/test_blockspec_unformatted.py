from __future__ import annotations

import numpy as np
import pytest

from ECLlib import BlockSpec, unfmt_block, unfmt_file


def _write_specs(path, specs):
    with open(path, "wb") as file:
        for spec in specs:
            file.write(unfmt_block.from_spec(spec).as_bytes())


def test_blockspec_normalizes_fortran_order():
    spec = BlockSpec("SATNUM", [[1, 2], [3, 4]], "int")

    assert spec.array().tolist() == [1, 3, 2, 4]


def test_blockspec_validates_keyword_and_payload_constraints():
    with pytest.raises(ValueError, match="1-8 characters"):
        BlockSpec("TOO_LONG_KEY", [1], "int")

    with pytest.raises(ValueError, match="<= 8 bytes"):
        BlockSpec("NAME", ["123456789"], "char")

    with pytest.raises(ValueError, match="MESS blocks must use an empty payload"):
        BlockSpec("STARTSOL", [1], "mess")


def test_unfmt_block_bool_payload_uses_4_byte_logicals():
    raw = unfmt_block.from_spec(BlockSpec("FLAG", [True, False, True], "bool")).as_bytes()

    assert len(raw) == 44
    assert int.from_bytes(raw[24:28], "big", signed=True) == 12
    assert np.frombuffer(raw[28:-4], dtype=">i4").tolist() == [1, 0, 1]
    assert int.from_bytes(raw[-4:], "big", signed=True) == 12


def test_unfmt_block_roundtrip_for_supported_dtypes(tmp_path):
    path = tmp_path / "blocks.bin"
    specs = [
        BlockSpec("INTVAL", [1, -2], "int"),
        BlockSpec("REALS", [[1.0, 2.0], [3.0, 4.0]], "float"),
        BlockSpec("DOUBLES", [1.5, 2.5], "double"),
        BlockSpec("FLAG", [True, False, True], "bool"),
        BlockSpec("WORDS", ["AA", "BB"], "char"),
        BlockSpec("EMPTY", [], "mess"),
        BlockSpec("TAIL", [99], "int"),
    ]
    _write_specs(path, specs)

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
