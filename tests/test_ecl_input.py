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
