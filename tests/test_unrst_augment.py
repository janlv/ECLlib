from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ECLlib import UNRST_file, unfmt_block


#---------------------------------------------------------------------------------------------------
def _write_unrst(path: Path, steps=(0, 1, 2), endkey="ENDSOL", header_specs=None, solution_specs=None):
#---------------------------------------------------------------------------------------------------
    """Write a small synthetic UNRST file for append and merge tests."""
    header_specs = header_specs or {}
    solution_specs = solution_specs or {}
    with open(path, "wb") as file:
        for step in steps:
            specs = (
                ("SEQNUM", [step], "int"),
                ("INTEHEAD", [1000 + step, 2000 + step], "int"),
                *tuple(header_specs.get(step, ())),
                ("STARTSOL", [], "mess"),
                ("TEMP", [float(step), float(step) + 0.25], "float"),
                ("PRESSURE", [100.0 + step], "float"),
                *tuple(solution_specs.get(step, ())),
                (endkey, [], "mess"),
            )
            for key, data, dtype in specs:
                file.write(unfmt_block.from_data(key, data, dtype).as_bytes())


#---------------------------------------------------------------------------------------------------
def _section_bytes(unrst: UNRST_file):
#---------------------------------------------------------------------------------------------------
    return [b"".join(block.binarydata() for block in section) for section in unrst.section_blocks()]


#---------------------------------------------------------------------------------------------------
def _section_snapshot(unrst: UNRST_file, index: int):
#---------------------------------------------------------------------------------------------------
    """Return ordered keys and decoded solution data for one section."""
    for i, section in enumerate(unrst.section_blocks(use_mmap=False)):
        if i != index:
            continue
        keys = [block.key() for block in section]
        data = {}
        for block in section:
            if block.key() in {"SEQNUM", "INTEHEAD", "STARTSOL", "ENDSOL"}:
                continue
            data[block.key()] = block.data().tolist()
        return keys, data
    raise IndexError(index)


#---------------------------------------------------------------------------------------------------
def _section_block_bytes(unrst: UNRST_file, index: int):
#---------------------------------------------------------------------------------------------------
    """Return serialized block bytes for one section."""
    for i, section in enumerate(unrst.section_blocks(use_mmap=False)):
        if i == index:
            return [block.binarydata() for block in section]
    raise IndexError(index)


#---------------------------------------------------------------------------------------------------
def _solution_block_bytes(unrst: UNRST_file, index: int, *keys: str):
#---------------------------------------------------------------------------------------------------
    """Return serialized solution blocks for the selected keys in one section."""
    keyset = set(keys)
    for i, section in enumerate(unrst.section_blocks(use_mmap=False)):
        if i != index:
            continue
        in_solution = False
        selected = []
        for block in section:
            if block.key() == "STARTSOL":
                in_solution = True
                continue
            if in_solution and block.key() in keyset:
                selected.append(block.binarydata())
        return selected
    raise IndexError(index)


#---------------------------------------------------------------------------------------------------
def test_unrst_file_public_imports_are_stable():
#---------------------------------------------------------------------------------------------------
    """Keep the public UNRST_file imports stable across the module split."""
    from ECLlib.io.output import UNRST_file as output_unrst_file

    assert output_unrst_file is UNRST_file


#---------------------------------------------------------------------------------------------------
def test_unfmt_block_renamed_bytes_preserves_payload_and_dtype():
#---------------------------------------------------------------------------------------------------
    """Renaming block bytes should preserve payload validation and serialized dtype."""
    block = unfmt_block.from_data("TEMP", [1.5, 2.5], "float")

    renamed = block.renamed_bytes("TNEW")
    expected = unfmt_block.from_data("TNEW", [1.5, 2.5], "float").as_bytes()

    assert renamed == expected

    renamed_long = block.renamed_bytes("TOO_LONG9")
    expected_long = unfmt_block.from_data("TOO_LONG9", [1.5, 2.5], "float").as_bytes()

    assert renamed_long == expected_long


#---------------------------------------------------------------------------------------------------
def test_merge_keys_from_copies_selected_donor_keys_into_all_sections(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Merge selected donor keys into each matching host section."""
    host_path = tmp_path / "host.UNRST"
    donor_path = tmp_path / "donor.UNRST"
    out = tmp_path / "merged.UNRST"
    _write_unrst(host_path)
    _write_unrst(
        donor_path,
        solution_specs={
            0: (("KEY1", [0.1, 0.2], "float"), ("KEY2", [10], "int")),
            1: (("KEY1", [1.1, 1.2], "float"), ("KEY2", [11], "int")),
            2: (("KEY1", [2.1, 2.2], "float"), ("KEY2", [12], "int")),
        },
    )

    merged = UNRST_file(host_path).merge_keys_from(donor_path, keys=("KEY1", "KEY2"), name=out)

    for index, values in enumerate((((0.1, 0.2), [10]), ((1.1, 1.2), [11]), ((2.1, 2.2), [12]))):
        keys, data = _section_snapshot(merged, index)
        assert keys == [
            "SEQNUM",
            "INTEHEAD",
            "STARTSOL",
            "TEMP",
            "PRESSURE",
            "KEY1",
            "KEY2",
            "ENDSOL",
        ]
        assert data["KEY1"] == pytest.approx(values[0])
        assert data["KEY2"] == values[1]


#---------------------------------------------------------------------------------------------------
def test_merge_keys_from_uses_default_output_name(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Default merged output names should follow the host stem plus _MERGED."""
    host_path = tmp_path / "host.UNRST"
    donor_path = tmp_path / "donor.UNRST"
    _write_unrst(host_path)
    _write_unrst(donor_path, solution_specs={step: (("KEY1", [step], "int"),) for step in (0, 1, 2)})

    merged = UNRST_file(host_path).merge_keys_from(donor_path, keys=("KEY1",))

    assert merged.path == tmp_path / "host_MERGED.UNRST"
    assert merged.exists()


#---------------------------------------------------------------------------------------------------
def test_merge_keys_from_allows_explicit_output_name(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Explicit output paths should override the default merged naming."""
    host_path = tmp_path / "host.UNRST"
    donor_path = tmp_path / "donor.UNRST"
    out = tmp_path / "custom.UNRST"
    _write_unrst(host_path)
    _write_unrst(donor_path, solution_specs={step: (("KEY1", [step], "int"),) for step in (0, 1, 2)})

    merged = UNRST_file(host_path).merge_keys_from(donor_path, keys=("KEY1",), name=out)

    assert merged.path == out


#---------------------------------------------------------------------------------------------------
def test_merge_keys_from_preserves_host_bytes_before_the_end_marker(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Merged sections should equal host bytes plus donor payloads before ENDSOL."""
    host_path = tmp_path / "host.UNRST"
    donor_path = tmp_path / "donor.UNRST"
    _write_unrst(host_path)
    _write_unrst(
        donor_path,
        solution_specs={step: (("KEY1", [step + 10], "int"), ("KEY2", [step + 20], "int"))
                        for step in (0, 1, 2)},
    )

    host = UNRST_file(host_path)
    merged = host.merge_keys_from(donor_path, keys=("KEY1", "KEY2"), name=tmp_path / "merged.UNRST")

    host_blocks = _section_block_bytes(host, 1)
    donor_blocks = _solution_block_bytes(UNRST_file(donor_path), 1, "KEY1", "KEY2")
    merged_blocks = _section_block_bytes(merged, 1)
    expected = host_blocks[:-1] + donor_blocks + host_blocks[-1:]
    assert b"".join(merged_blocks) == b"".join(expected)


#---------------------------------------------------------------------------------------------------
def test_merge_keys_from_uses_only_donor_solution_blocks(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Requested donor keys must be copied from the donor solution region only."""
    host_path = tmp_path / "host.UNRST"
    donor_path = tmp_path / "donor.UNRST"
    _write_unrst(host_path, steps=(7,))
    _write_unrst(
        donor_path,
        steps=(7,),
        header_specs={7: (("KEY1", [999.0], "float"),)},
        solution_specs={7: (("KEY1", [7.5, 7.75], "float"),)},
    )

    merged = UNRST_file(host_path).merge_keys_from(donor_path, keys=("KEY1",), name=tmp_path / "merged.UNRST")
    _, data = _section_snapshot(merged, 0)
    assert data["KEY1"] == pytest.approx([7.5, 7.75])


#---------------------------------------------------------------------------------------------------
def test_merge_keys_from_supports_renaming_donor_keys(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Selected donor keys can be renamed before they are appended to the host sections."""
    host_path = tmp_path / "host.UNRST"
    donor_path = tmp_path / "donor.UNRST"
    _write_unrst(host_path)
    _write_unrst(
        donor_path,
        solution_specs={step: (("KEY1", [float(step)], "float"),) for step in (0, 1, 2)},
    )

    merged = UNRST_file(host_path).merge_keys_from(
        UNRST_file(donor_path),
        keys=("KEY1",),
        rename={"KEY1": "RENAMED"},
        name=tmp_path / "renamed.UNRST",
    )

    keys, data = _section_snapshot(merged, 2)
    assert "RENAMED" in keys
    assert "KEY1" not in keys
    assert data["RENAMED"] == pytest.approx([2.0])


#---------------------------------------------------------------------------------------------------
def test_merge_keys_from_does_not_use_section_blocks(tmp_path, monkeypatch):
#---------------------------------------------------------------------------------------------------
    """The optimized merge path should not fall back to section tuple materialization."""
    host_path = tmp_path / "host.UNRST"
    donor_path = tmp_path / "donor.UNRST"
    out = tmp_path / "merged.UNRST"
    _write_unrst(host_path)
    _write_unrst(donor_path, solution_specs={step: (("KEY1", [step], "int"),) for step in (0, 1, 2)})

    def fail(*args, **kwargs):
        """Fail if the old section-based merge path is used."""
        raise AssertionError("section_blocks should not be used by merge_keys_from()")

    with monkeypatch.context() as patcher:
        patcher.setattr(UNRST_file, "section_blocks", fail)
        merged = UNRST_file(host_path).merge_keys_from(donor_path, keys=("KEY1",), name=out)

    _, data = _section_snapshot(merged, 1)
    assert data["KEY1"] == [1]


#---------------------------------------------------------------------------------------------------
def test_merge_keys_from_deletes_partial_output_on_failure(tmp_path, monkeypatch):
#---------------------------------------------------------------------------------------------------
    """Failed merges should remove partially written output files."""
    host_path = tmp_path / "host.UNRST"
    donor_path = tmp_path / "donor.UNRST"
    out = tmp_path / "merged.UNRST"
    _write_unrst(host_path)
    _write_unrst(
        donor_path,
        solution_specs={step: (("KEY1", [step], "int"),) for step in (0, 1, 2)},
    )

    renamed_calls = {"count": 0}
    original = unfmt_block.renamed_bytes

    def fail_on_second_rename(self, key):
        """Raise during the second rename to simulate a mid-merge failure."""
        renamed_calls["count"] += 1
        if renamed_calls["count"] == 2:
            raise RuntimeError("boom")
        return original(self, key)

    monkeypatch.setattr(unfmt_block, "renamed_bytes", fail_on_second_rename)

    with pytest.raises(RuntimeError):
        UNRST_file(host_path).merge_keys_from(
            donor_path,
            keys=("KEY1",),
            rename={"KEY1": "RENAMED"},
            name=out,
        )

    assert not out.exists()


#---------------------------------------------------------------------------------------------------
def test_merge_keys_from_large_synthetic_smoke(tmp_path):
#---------------------------------------------------------------------------------------------------
    """A larger synthetic merge should preserve section count and selected donor values."""
    steps = tuple(range(40))
    host_path = tmp_path / "host.UNRST"
    donor_path = tmp_path / "donor.UNRST"
    _write_unrst(host_path, steps=steps)
    _write_unrst(
        donor_path,
        steps=steps,
        solution_specs={
            step: (
                ("KEY1", [step + 0.1, step + 0.2], "float"),
                ("KEY2", [step + 10], "int"),
                ("KEY3", [step % 2 == 0], "bool"),
            )
            for step in steps
        },
    )

    merged = UNRST_file(host_path).merge_keys_from(
        donor_path,
        keys=("KEY1", "KEY2", "KEY3"),
        name=tmp_path / "smoke.UNRST",
    )

    assert merged.count_sections() == len(steps)
    for index in (0, 17, 39):
        _, data = _section_snapshot(merged, index)
        assert data["KEY1"] == pytest.approx([index + 0.1, index + 0.2])
        assert data["KEY2"] == [index + 10]
        assert data["KEY3"] == [index % 2 == 0]


#---------------------------------------------------------------------------------------------------
def test_append_blocks_updates_last_section_in_place(tmp_path):
#---------------------------------------------------------------------------------------------------
    src = tmp_path / "source.UNRST"
    _write_unrst(src)

    unrst = UNRST_file(src)
    before = _section_bytes(unrst)
    appended = unrst.append_blocks(
        step=2,
        keys=("XAPP", "FLAG"),
        blocks=(
            np.array([2.5, 3.5], dtype=np.float32),
            np.array([True, False], dtype=bool),
        ),
    )

    after = _section_bytes(appended)
    assert before[0] == after[0]
    assert before[1] == after[1]
    assert before[2] != after[2]

    keys, data = _section_snapshot(appended, 2)
    assert keys == [
        "SEQNUM",
        "INTEHEAD",
        "STARTSOL",
        "TEMP",
        "PRESSURE",
        "XAPP",
        "FLAG",
        "ENDSOL",
    ]
    assert data["XAPP"] == pytest.approx([2.5, 3.5])
    assert data["FLAG"] == [True, False]


#---------------------------------------------------------------------------------------------------
def test_append_blocks_normalizes_long_keys_in_one_batch(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Append-time long keys should be written with plain 8-character truncation."""
    src = tmp_path / "source.UNRST"
    _write_unrst(src)

    appended = UNRST_file(src).append_blocks(
        step=2,
        keys=("anhydrite", "anhydrit", "anhydrite2"),
        blocks=(
            np.array([1.5], dtype=np.float32),
            np.array([2.5], dtype=np.float32),
            np.array([3.5], dtype=np.float32),
        ),
    )

    keys, data = _section_snapshot(appended, 2)

    assert keys[-4:-1] == ["anhydrit", "anhydrit", "anhydrit"]
    assert data["anhydrit"] == pytest.approx([3.5])


#---------------------------------------------------------------------------------------------------
def test_unrst_file_check_duplicate_keys_returns_empty_tuple(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Duplicate-key checker should report no conflicts for unique section keys."""
    src = tmp_path / "source.UNRST"
    _write_unrst(src)

    assert UNRST_file(src).check_duplicate_keys(sec=0, raise_error=False) == ()


def test_unrst_file_check_duplicate_keys_reports_truncated_duplicates(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Duplicate-key checker should report truncated duplicates once in first-seen order."""
    src = tmp_path / "source.UNRST"
    _write_unrst(
        src,
        steps=(0,),
        solution_specs={0: (("anhydrite", [1.5], "float"), ("anhydrit", [2.5], "float"))},
    )

    assert UNRST_file(src).check_duplicate_keys(sec=0, raise_error=False) == ("anhydrit",)


#---------------------------------------------------------------------------------------------------
def test_unrst_file_check_duplicate_keys_raises_on_duplicates(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Duplicate-key checker should raise when requested."""
    src = tmp_path / "source.UNRST"
    _write_unrst(
        src,
        steps=(0,),
        solution_specs={0: (("anhydrite", [1.5], "float"), ("anhydrit", [2.5], "float"))},
    )

    with pytest.raises(ValueError, match="anhydrit"):
        UNRST_file(src).check_duplicate_keys(sec=0, raise_error=True)


#---------------------------------------------------------------------------------------------------
def test_append_blocks_uses_unfmt_block_from_data(tmp_path, monkeypatch):
#---------------------------------------------------------------------------------------------------
    """The append path should use the canonical low-level block constructor."""
    src = tmp_path / "source.UNRST"
    _write_unrst(src)

    calls = {"count": 0}
    original = unfmt_block.from_data

    def counting_from_data(cls, key, data, dtype):
        """Count append-time block construction calls."""
        calls["count"] += 1
        return original.__func__(cls, key, data, dtype)

    with monkeypatch.context() as patcher:
        patcher.setattr(unfmt_block, "from_data", classmethod(counting_from_data))
        appended = UNRST_file(src).append_blocks(
            step=2,
            keys=("XFAST",),
            blocks=(np.array([8, 9], dtype=np.int32),),
        )

    keys, data = _section_snapshot(appended, 2)
    assert calls["count"] == 1
    assert keys[-2:] == ["XFAST", "ENDSOL"]
    assert data["XFAST"] == [8, 9]


#---------------------------------------------------------------------------------------------------
def test_append_blocks_requires_last_step(tmp_path):
#---------------------------------------------------------------------------------------------------
    src = tmp_path / "source.UNRST"
    _write_unrst(src)

    with pytest.raises(ValueError, match="last section"):
        UNRST_file(src).append_blocks(
            step=1,
            keys=("XFAIL",),
            blocks=(np.array([1], dtype=np.int32),),
        )


#---------------------------------------------------------------------------------------------------
def test_append_blocks_validates_endblock(tmp_path):
#---------------------------------------------------------------------------------------------------
    src = tmp_path / "source.UNRST"
    _write_unrst(src)

    with pytest.raises(ValueError, match="does not end with STOP"):
        UNRST_file(src).append_blocks(
            step=2,
            keys=("XFAIL",),
            blocks=(np.array([1], dtype=np.int32),),
            endblock="STOP",
        )


#---------------------------------------------------------------------------------------------------
def test_append_blocks_supports_custom_endblock(tmp_path):
#---------------------------------------------------------------------------------------------------
    src = tmp_path / "source.UNRST"
    _write_unrst(src, endkey="STOP")

    appended = UNRST_file(src).append_blocks(
        step=2,
        keys=("XSTOP",),
        blocks=(np.array([7], dtype=np.int32),),
        endblock="STOP",
    )

    keys, data = _section_snapshot(appended, 2)
    assert keys[-2:] == ["XSTOP", "STOP"]
    assert data["XSTOP"] == [7]
