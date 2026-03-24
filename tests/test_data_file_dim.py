from __future__ import annotations

from ECLlib.io.input import DATA_file


#---------------------------------------------------------------------------------------------------
def test_data_file_dim_reads_dimens(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Return dimensions from the DIMENS keyword."""
    path = tmp_path / "CASE.DATA"
    path.write_text("RUNSPEC\nDIMENS\n  10 20 30 /\nEND\n", encoding="utf-8")

    assert DATA_file(path).dim() == (10, 20, 30)


#---------------------------------------------------------------------------------------------------
def test_data_file_dim_falls_back_to_specgrid(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Return dimensions from SPECGRID when DIMENS is absent."""
    path = tmp_path / "CASE.DATA"
    path.write_text("GRID\nSPECGRID\n  11 22 33 1 F /\nEND\n", encoding="utf-8")

    assert DATA_file(path).dim() == (11, 22, 33)


#---------------------------------------------------------------------------------------------------
def test_data_file_dim_reads_include(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Return dimensions from an included file."""
    path = tmp_path / "CASE.DATA"
    inc = tmp_path / "DIMENS.INC"
    path.write_text("RUNSPEC\nINCLUDE\n 'DIMENS.INC' /\nEND\n", encoding="utf-8")
    inc.write_text("DIMENS\n  3 4 5 /\n", encoding="utf-8")

    assert DATA_file(path).dim() == (3, 4, 5)


#---------------------------------------------------------------------------------------------------
def test_data_file_dim_returns_none_when_missing(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Return None when the deck does not define dimensions."""
    path = tmp_path / "CASE.DATA"
    path.write_text("RUNSPEC\nSTART\n  1 JAN 2000 /\nEND\n", encoding="utf-8")

    assert DATA_file(path).dim() is None
