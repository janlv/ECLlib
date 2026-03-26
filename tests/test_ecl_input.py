from __future__ import annotations

import pytest

from ECLlib.io.input import DATA_file, ECL_input


#---------------------------------------------------------------------------------------------------
def test_ecl_input_accepts_case_root(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Resolve a case root to the matching DATA path."""
    path = tmp_path / "CASE.DATA"
    path.write_text("RUNSPEC\nDIMENS\n  2 3 4 /\nEND\n", encoding="utf-8")

    case = ECL_input(tmp_path / "CASE")

    assert case.path == path.resolve()


#---------------------------------------------------------------------------------------------------
def test_ecl_input_accepts_direct_data_path(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Keep the supplied DATA path when it is explicit."""
    path = tmp_path / "CASE.DATA"
    path.write_text("RUNSPEC\nDIMENS\n  2 3 4 /\nEND\n", encoding="utf-8")

    case = ECL_input(path)

    assert case.path == path.resolve()


#---------------------------------------------------------------------------------------------------
def test_ecl_input_dim_delegates_to_data_file(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Expose the DATA-file dimensions through the wrapper."""
    path = tmp_path / "CASE.DATA"
    path.write_text("RUNSPEC\nDIMENS\n  5 6 7 /\nEND\n", encoding="utf-8")

    assert ECL_input(path).dim() == (5, 6, 7)


#---------------------------------------------------------------------------------------------------
def test_ecl_input_wells_by_type_delegates_to_data_file(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Expose typed well metadata through the wrapper."""
    path = tmp_path / "CASE.DATA"
    path.write_text(
        "SCHEDULE\n"
        "WCONPROD\n"
        " 'P1' OPEN ORAT 100 4* /\n"
        "/\n"
        "WCONINJE\n"
        " 'I1' WAT OPEN RATE 200 1* /\n"
        "/\n"
        "END\n",
        encoding="utf-8",
    )

    assert ECL_input(path).wells_by_type() == DATA_file(path).wells_by_type()


#---------------------------------------------------------------------------------------------------
def test_ecl_input_wellpos_by_name_delegates_to_data_file(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Expose COMPDAT well positions through the wrapper."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nCOMPDAT\n 'P1' 2 3 4 4 OPEN 1* /\n/\nEND\n", encoding="utf-8")

    assert ECL_input(path).wellpos_by_name(wellnames=("P1",)) == DATA_file(path).wellpos_by_name(
        wellnames=("P1",)
    )


#---------------------------------------------------------------------------------------------------
def test_data_file_wells_by_type_reads_wconprod(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Classify producer wells from WCONPROD."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nWCONPROD\n 'P1' OPEN ORAT 100 4* /\n/\nEND\n", encoding="utf-8")

    assert DATA_file(path).wells_by_type() == {"PRODUCER": ["P1"]}


#---------------------------------------------------------------------------------------------------
def test_data_file_wells_by_type_reads_wconhist(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Classify producer wells from WCONHIST."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nWCONHIST\n 'P1' OPEN ORAT 100 4* /\n/\nEND\n", encoding="utf-8")

    assert DATA_file(path).wells_by_type() == {"PRODUCER": ["P1"]}


#---------------------------------------------------------------------------------------------------
def test_data_file_wells_by_type_reads_wconinje(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Classify injector wells from WCONINJE."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nWCONINJE\n 'I1' WAT OPEN RATE 200 1* /\n/\nEND\n", encoding="utf-8")

    assert DATA_file(path).wells_by_type() == {"INJECTOR": ["I1"]}


#---------------------------------------------------------------------------------------------------
def test_data_file_wells_by_type_reads_wconinjh(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Classify injector wells from WCONINJH."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nWCONINJH\n 'I1' WAT OPEN 200 /\n/\nEND\n", encoding="utf-8")

    assert DATA_file(path).wells_by_type() == {"INJECTOR": ["I1"]}


#---------------------------------------------------------------------------------------------------
def test_data_file_wells_by_type_preserves_quoted_well_names(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Keep quoted well names intact when classifying controls."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nWCONPROD\n 'PROD A' OPEN ORAT 100 4* /\n/\nEND\n", encoding="utf-8")

    assert DATA_file(path).wells_by_type() == {"PRODUCER": ["PROD A"]}


#---------------------------------------------------------------------------------------------------
def test_data_file_wells_by_type_reads_included_schedule(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Read typed well controls from included schedule files."""
    path = tmp_path / "CASE.DATA"
    inc = tmp_path / "SCHEDULE.INC"
    path.write_text("SCHEDULE\nINCLUDE\n 'SCHEDULE.INC' /\nEND\n", encoding="utf-8")
    inc.write_text("WCONPROD\n 'P1' OPEN ORAT 100 4* /\n/\n", encoding="utf-8")

    assert DATA_file(path).wells_by_type() == {"PRODUCER": ["P1"]}


#---------------------------------------------------------------------------------------------------
def test_data_file_wells_by_type_reads_sch_fallback(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Read typed well controls from a matching SCH file when needed."""
    path = tmp_path / "CASE.DATA"
    sch = tmp_path / "CASE.SCH"
    path.write_text("RUNSPEC\nREADDATA\nEND\n", encoding="utf-8")
    sch.write_text("WCONINJE\n 'I1' WAT OPEN RATE 200 1* /\n/\n", encoding="utf-8")

    assert DATA_file(path).wells_by_type() == {"INJECTOR": ["I1"]}


#---------------------------------------------------------------------------------------------------
def test_data_file_wells_by_type_allows_compatible_repeats(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Allow repeated controls that normalize to the same well type."""
    path = tmp_path / "CASE.DATA"
    path.write_text(
        "SCHEDULE\n"
        "WCONPROD\n"
        " 'P1' OPEN ORAT 100 4* /\n"
        "/\n"
        "WCONHIST\n"
        " 'P1' OPEN ORAT 90 4* /\n"
        "/\n"
        "END\n",
        encoding="utf-8",
    )

    assert DATA_file(path).wells_by_type() == {"PRODUCER": ["P1"]}


#---------------------------------------------------------------------------------------------------
def test_data_file_wells_by_type_raises_on_templates_and_lists(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Reject well templates and well lists until they are expanded properly."""
    path = tmp_path / "CASE.DATA"
    path.write_text(
        "SCHEDULE\n"
        "WCONINJE\n"
        " 'INJ*' WAT OPEN RATE 200 1* /\n"
        "/\n"
        "END\n",
        encoding="utf-8",
    )

    with pytest.raises(SystemError, match="templates/lists are not supported"):
        DATA_file(path).wells_by_type()


#---------------------------------------------------------------------------------------------------
def test_data_file_wells_by_type_omits_wellspecs_only_wells(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Ignore wells that never receive a typed schedule control."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nWELSPECS\n 'W1' G1 1 1 1* OIL /\n/\nEND\n", encoding="utf-8")

    assert DATA_file(path).wells_by_type() == {}


#---------------------------------------------------------------------------------------------------
def test_data_file_wellpos_by_name_reads_compdat(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Read a single explicit COMPDAT completion."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nCOMPDAT\n 'P1' 2 3 4 4 OPEN 1* /\n/\nEND\n", encoding="utf-8")

    assert DATA_file(path).wellpos_by_name(wellnames=("P1",)) == {"P1": ((1, 2, 3),)}


#---------------------------------------------------------------------------------------------------
def test_data_file_wellpos_by_name_expands_k_ranges(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Expand K1..K2 into explicit zero-based cells."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nCOMPDAT\n 'P1' 2 3 4 6 OPEN 1* /\n/\nEND\n", encoding="utf-8")

    assert DATA_file(path).wellpos_by_name(wellnames=("P1",)) == {
        "P1": ((1, 2, 3), (1, 2, 4), (1, 2, 5))
    }


#---------------------------------------------------------------------------------------------------
def test_data_file_wellpos_by_name_preserves_quoted_names(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Keep quoted well names intact when reading COMPDAT."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nCOMPDAT\n 'PROD A' 1 1 1 2 OPEN 1* /\n/\nEND\n", encoding="utf-8")

    assert DATA_file(path).wellpos_by_name(wellnames=("PROD A",)) == {
        "PROD A": ((0, 0, 0), (0, 0, 1))
    }


#---------------------------------------------------------------------------------------------------
def test_data_file_wellpos_by_name_reads_included_schedule(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Read COMPDAT records from included schedule files."""
    path = tmp_path / "CASE.DATA"
    inc = tmp_path / "SCHEDULE.INC"
    path.write_text("SCHEDULE\nINCLUDE\n 'SCHEDULE.INC' /\nEND\n", encoding="utf-8")
    inc.write_text("COMPDAT\n 'P1' 2 3 4 4 OPEN 1* /\n/\n", encoding="utf-8")

    assert DATA_file(path).wellpos_by_name(wellnames=("P1",)) == {"P1": ((1, 2, 3),)}


#---------------------------------------------------------------------------------------------------
def test_data_file_wellpos_by_name_reads_sch_fallback(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Read COMPDAT from a matching SCH file when needed."""
    path = tmp_path / "CASE.DATA"
    sch = tmp_path / "CASE.SCH"
    path.write_text("RUNSPEC\nREADDATA\nEND\n", encoding="utf-8")
    sch.write_text("COMPDAT\n 'P1' 2 3 4 4 OPEN 1* /\n/\n", encoding="utf-8")

    assert DATA_file(path).wellpos_by_name(wellnames=("P1",)) == {"P1": ((1, 2, 3),)}


#---------------------------------------------------------------------------------------------------
def test_data_file_wellpos_by_name_filters_and_preserves_requested_order(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Return only requested wells and keep their order."""
    path = tmp_path / "CASE.DATA"
    path.write_text(
        "SCHEDULE\n"
        "COMPDAT\n"
        " 'P1' 1 1 1 1 OPEN 1* /\n"
        " 'P2' 2 2 2 2 OPEN 1* /\n"
        "/\n"
        "END\n",
        encoding="utf-8",
    )

    result = DATA_file(path).wellpos_by_name(wellnames=("P2", "P1"))

    assert list(result) == ["P2", "P1"]
    assert result == {"P2": ((1, 1, 1),), "P1": ((0, 0, 0),)}


#---------------------------------------------------------------------------------------------------
def test_data_file_wellpos_by_name_returns_ijk_columns(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Return transposed I/J/K arrays when requested."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nCOMPDAT\n 'P1' 2 3 4 5 OPEN 1* /\n/\nEND\n", encoding="utf-8")

    assert DATA_file(path).wellpos_by_name(ijk=True, wellnames=("P1",)) == {
        "P1": ((1, 1), (2, 2), (3, 4))
    }


#---------------------------------------------------------------------------------------------------
def test_data_file_wellpos_by_name_keeps_missing_wells(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Return requested wells without COMPDAT as empty tuples."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nCOMPDAT\n 'P1' 2 3 4 4 OPEN 1* /\n/\nEND\n", encoding="utf-8")

    assert DATA_file(path).wellpos_by_name(wellnames=("P1", "P2")) == {
        "P1": ((1, 2, 3),),
        "P2": (),
    }


#---------------------------------------------------------------------------------------------------
def test_data_file_wellpos_by_name_defaults_to_welspecs(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Use WELSPECS as the default well list."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nWELSPECS\n 'W1' G1 1 1 1* OIL /\n/\nEND\n", encoding="utf-8")

    assert DATA_file(path).wellpos_by_name() == {"W1": ()}


#---------------------------------------------------------------------------------------------------
def test_data_file_wellpos_by_name_raises_on_templates_and_lists(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Reject COMPDAT well templates and well lists."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nCOMPDAT\n 'INJ*' 2 3 4 4 OPEN 1* /\n/\nEND\n", encoding="utf-8")

    with pytest.raises(SystemError, match="templates/lists are not supported"):
        DATA_file(path).wellpos_by_name(wellnames=("P1",))


#---------------------------------------------------------------------------------------------------
def test_data_file_wellpos_by_name_raises_on_defaulted_coordinates(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Reject COMPDAT records without explicit I/J/K values."""
    path = tmp_path / "CASE.DATA"
    path.write_text("SCHEDULE\nCOMPDAT\n 'P1' 1* 3 4 4 OPEN 1* /\n/\nEND\n", encoding="utf-8")

    with pytest.raises(SystemError, match="Explicit COMPDAT I/J/K1/K2 are required"):
        DATA_file(path).wellpos_by_name(wellnames=("P1",))


#---------------------------------------------------------------------------------------------------
def test_data_file_get_matches_get_old(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Keep the cached get path aligned with the legacy implementation."""
    path = tmp_path / "CASE.DATA"
    path.write_text(
        "RUNSPEC\n"
        "START\n"
        " 1 JAN 2020 /\n"
        "DIMENS\n"
        " 1 2 3 /\n"
        "SCHEDULE\n"
        "WELSPECS\n"
        " 'PROD A' G1 1 1 1* OIL /\n"
        "/\n"
        "TSTEP\n"
        " 5 10 /\n"
        "/\n"
        "END\n",
        encoding="utf-8",
    )

    new_deck = DATA_file(path)
    old_deck = DATA_file(path)

    assert new_deck.get("START", "DIMENS", "WELSPECS", "TSTEP") == old_deck.get_old(
        "START", "DIMENS", "WELSPECS", "TSTEP"
    )
    assert DATA_file(path).get("TSTEP", pos=True) == DATA_file(path).get_old("TSTEP", pos=True)


#---------------------------------------------------------------------------------------------------
def test_data_file_including_invalidates_get_cache(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Refresh cached section text when additional include files are injected."""
    path = tmp_path / "CASE.DATA"
    include_path = tmp_path / "SCHEDULE.INC"
    path.write_text("SCHEDULE\nEND\n", encoding="utf-8")
    include_path.write_text("TSTEP\n 3 /\n/\n", encoding="utf-8")

    deck = DATA_file(path)

    assert deck.get("TSTEP") == ()
    assert deck.including(include_path.name).get("TSTEP") == (3.0,)
