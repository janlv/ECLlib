from __future__ import annotations

from pathlib import Path
from itertools import islice

import pytest

from ECLlib import BlockSpec, UNRST_file, unfmt_block


def _write_unrst(path: Path, steps=(0, 1, 2)):
    with open(path, "wb") as file:
        for step in steps:
            specs = (
                BlockSpec("SEQNUM", [step], "int"),
                BlockSpec("INTEHEAD", [1000 + step, 2000 + step], "int"),
                BlockSpec("STARTSOL", [], "mess"),
                BlockSpec("TEMP", [float(step), float(step) + 0.25], "float"),
                BlockSpec("PRESSURE", [100.0 + step], "float"),
                BlockSpec("ENDSOL", [], "mess"),
            )
            for spec in specs:
                file.write(unfmt_block.from_spec(spec).as_bytes())


def _section_bytes(unrst: UNRST_file):
    return [b"".join(block.binarydata() for block in section) for section in unrst.section_blocks()]


def _section_snapshot(unrst: UNRST_file, index: int):
    section = next(islice(unrst.section_blocks(), index, None))
    keys = [block.key() for block in section]
    data = {}
    for block in section:
        if block.key() in {"SEQNUM", "INTEHEAD", "STARTSOL", "ENDSOL"}:
            continue
        data[block.key()] = block.data().tolist()
    return keys, data


def test_unrst_augment_selective_append_preserves_other_sections(tmp_path):
    src = tmp_path / "source.UNRST"
    out = tmp_path / "augmented.UNRST"
    _write_unrst(src)

    original = UNRST_file(src)
    augmented = original.augment(
        out,
        lambda step, _section: [BlockSpec("XTEST", [float(step), float(step) + 0.5], "float")],
        steps=(1,),
    )

    original_sections = _section_bytes(original)
    augmented_sections = _section_bytes(augmented)
    assert original_sections[0] == augmented_sections[0]
    assert original_sections[2] == augmented_sections[2]
    assert original_sections[1] != augmented_sections[1]

    keys, data = _section_snapshot(augmented, 1)
    assert keys == [
        "SEQNUM",
        "INTEHEAD",
        "STARTSOL",
        "TEMP",
        "PRESSURE",
        "XTEST",
        "ENDSOL",
    ]
    assert data["XTEST"] == pytest.approx([1.0, 1.5])


def test_unrst_augment_can_replace_existing_solution_keys(tmp_path):
    src = tmp_path / "source.UNRST"
    out = tmp_path / "replaced.UNRST"
    _write_unrst(src)

    augmented = UNRST_file(src).augment(
        out,
        {
            2: (
                BlockSpec("TEMP", [77.0, 88.0], "float"),
                BlockSpec("FLAG", [True, False], "bool"),
            )
        },
        steps=(2,),
        replace_keys=("TEMP",),
    )

    keys, data = _section_snapshot(augmented, 2)
    assert keys == [
        "SEQNUM",
        "INTEHEAD",
        "STARTSOL",
        "PRESSURE",
        "TEMP",
        "FLAG",
        "ENDSOL",
    ]
    assert data["TEMP"] == pytest.approx([77.0, 88.0])
    assert data["FLAG"] == [True, False]


def test_insert_blocks_last_modifies_only_last_section(tmp_path):
    src = tmp_path / "source.UNRST"
    out = tmp_path / "last.UNRST"
    _write_unrst(src)

    original = UNRST_file(src)
    inserted = original.insert_blocks(
        out,
        BlockSpec("XLAST", [9.0], "float"),
        target="last",
    )

    original_sections = _section_bytes(original)
    inserted_sections = _section_bytes(inserted)
    assert original_sections[0] == inserted_sections[0]
    assert original_sections[1] == inserted_sections[1]
    assert original_sections[2] != inserted_sections[2]

    keys, data = _section_snapshot(inserted, 2)
    assert keys == [
        "SEQNUM",
        "INTEHEAD",
        "STARTSOL",
        "TEMP",
        "PRESSURE",
        "XLAST",
        "ENDSOL",
    ]
    assert data["XLAST"] == pytest.approx([9.0])


def test_insert_blocks_all_broadcasts_same_blocks_to_all_sections(tmp_path):
    src = tmp_path / "source.UNRST"
    out = tmp_path / "all.UNRST"
    _write_unrst(src)

    inserted = UNRST_file(src).insert_blocks(
        out,
        {"key": "FLAG", "data": [True, False], "dtype": "bool"},
        target="all",
    )

    for index in range(3):
        keys, data = _section_snapshot(inserted, index)
        assert keys == [
            "SEQNUM",
            "INTEHEAD",
            "STARTSOL",
            "TEMP",
            "PRESSURE",
            "FLAG",
            "ENDSOL",
        ]
        assert data["FLAG"] == [True, False]


@pytest.mark.parametrize("target", [2, (2,)])
def test_insert_blocks_specific_targets_only_selected_steps(tmp_path, target):
    src = tmp_path / "source.UNRST"
    out = tmp_path / "selected.UNRST"
    _write_unrst(src)

    original = UNRST_file(src)
    inserted = original.insert_blocks(
        out,
        ("XSTEP", [20], "int"),
        target=target,
    )

    original_sections = _section_bytes(original)
    inserted_sections = _section_bytes(inserted)
    assert original_sections[0] == inserted_sections[0]
    assert original_sections[1] == inserted_sections[1]
    assert original_sections[2] != inserted_sections[2]

    keys, data = _section_snapshot(inserted, 2)
    assert keys == [
        "SEQNUM",
        "INTEHEAD",
        "STARTSOL",
        "TEMP",
        "PRESSURE",
        "XSTEP",
        "ENDSOL",
    ]
    assert data["XSTEP"] == [20]


def test_insert_blocks_replace_keys_removes_existing_solution_keys(tmp_path):
    src = tmp_path / "source.UNRST"
    out = tmp_path / "replace.UNRST"
    _write_unrst(src)

    inserted = UNRST_file(src).insert_blocks(
        out,
        ("TEMP", [77.0, 88.0], "float"),
        target=1,
        replace_keys=("TEMP",),
    )

    keys, data = _section_snapshot(inserted, 1)
    assert keys == [
        "SEQNUM",
        "INTEHEAD",
        "STARTSOL",
        "PRESSURE",
        "TEMP",
        "ENDSOL",
    ]
    assert data["TEMP"] == pytest.approx([77.0, 88.0])


@pytest.mark.parametrize(
    ("blocks", "expected_key", "expected_data"),
    [
        (BlockSpec("XFLOAT", [3.5], "float"), "XFLOAT", [3.5]),
        ({"key": "XDICT", "data": [7], "dtype": "int"}, "XDICT", [7]),
        (("XTUP", [True, False], "bool"), "XTUP", [True, False]),
    ],
)
def test_insert_blocks_normalizes_blockspec_dict_and_tuple_inputs(
    tmp_path, blocks, expected_key, expected_data
):
    src = tmp_path / "source.UNRST"
    out = tmp_path / f"{expected_key}.UNRST"
    _write_unrst(src)

    inserted = UNRST_file(src).insert_blocks(
        out,
        blocks,
        target="last",
    )

    keys, data = _section_snapshot(inserted, 2)
    assert keys[-2] == expected_key
    assert data[expected_key] == expected_data


def test_insert_blocks_requires_target_keyword(tmp_path):
    src = tmp_path / "source.UNRST"
    out = tmp_path / "missing_target.UNRST"
    _write_unrst(src)

    with pytest.raises(TypeError):
        UNRST_file(src).insert_blocks(out, BlockSpec("X", [1], "int"))
