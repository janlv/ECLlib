from __future__ import annotations

from pathlib import Path

from ECLlib import SMSPEC_file, unfmt_block

BlockSpec = tuple[str, list[object], str]


#---------------------------------------------------------------------------------------------------
def _write_smspec(path: Path, *, leading_blocks: tuple[BlockSpec, ...], names_key: str):
#---------------------------------------------------------------------------------------------------
    """Write a minimal SMSPEC file with the requested leading layout."""
    blocks = (
        *leading_blocks,
        ("DIMENS", [1, 1, 1], "int"),
        ("KEYWORDS", ["TIME", "WOPR", "WWPR"], "char"),
        (names_key, [":+:", "P-F-12", "P-F-12"], "char"),
        ("NUMS", [0, 0, 0], "int"),
        ("MEASRMNT", ["TIME", "RATE", "RATE"], "char"),
        ("UNITS", ["DAYS", "SM3/DAY", "SM3/DAY"], "char"),
        ("STARTDAT", [31, 12, 2007, 0, 0, 0], "int"),
    )
    with path.open("wb") as handle:
        for key, data, dtype in blocks:
            handle.write(unfmt_block.from_data(key, data, dtype).as_bytes())


#---------------------------------------------------------------------------------------------------
def _assert_well_vectors_are_readable(path: Path):
#---------------------------------------------------------------------------------------------------
    """Assert that well vectors can be selected from one SMSPEC layout."""
    spec = SMSPEC_file(path)

    assert {"WOPR", "WWPR"} <= set(spec.list_keys())
    wopr_vectors = spec.select_vectors(keys=("WOPR",))
    wwpr_vectors = spec.select_vectors(keys=("WWPR",))
    assert len(wopr_vectors) == 1
    assert len(wwpr_vectors) == 1
    assert wopr_vectors[0].key == "WOPR"
    assert wopr_vectors[0].name == "P-F-12"
    assert wwpr_vectors[0].key == "WWPR"
    assert wwpr_vectors[0].name == "P-F-12"
    assert spec.welldata(keys=("WOPR",), wells=("P-F-12",))
    assert spec.well_pos() == (1,)


#---------------------------------------------------------------------------------------------------
def test_smspec_reads_eclipse_layout_without_intehead(tmp_path: Path):
#---------------------------------------------------------------------------------------------------
    """Read Eclipse-style SMSPEC metadata that starts before DIMENS without INTEHEAD."""
    path = tmp_path / "ECLIPSE.SMSPEC"
    _write_smspec(path, leading_blocks=(("RESTART", ["F"], "char"),), names_key="WGNAMES")

    _assert_well_vectors_are_readable(path)


#---------------------------------------------------------------------------------------------------
def test_smspec_reads_intersect_layout_with_intehead(tmp_path: Path):
#---------------------------------------------------------------------------------------------------
    """Read INTERSECT-style SMSPEC metadata with INTEHEAD before DIMENS."""
    path = tmp_path / "INTERSECT.SMSPEC"
    _write_smspec(
        path,
        leading_blocks=(
            ("INTEHEAD", [1, 2, 3], "int"),
            ("RESTART", ["F"], "char"),
        ),
        names_key="NAMES",
    )

    _assert_well_vectors_are_readable(path)
