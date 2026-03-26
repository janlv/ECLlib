#---------------------------------------------------------------------------------------------------
from pathlib import Path
#---------------------------------------------------------------------------------------------------

from ECLlib.io.input import ECL_input, IX_input


#---------------------------------------------------------------------------------------------------
def test_ecl_input_regression_on_model_from_ixf_opm():
#---------------------------------------------------------------------------------------------------
    """Validate the shared metadata API against the OPM/Eclipse regression case."""
    case = ECL_input(Path("tests/Model_From_IXF_OPM/Model_From_IXF_OPM.DATA"))

    assert case.dim() == (25, 25, 5)
    assert case.wells_by_type() == {"PRODUCER": ["PRODUCER"], "INJECTOR": ["INJECTOR"]}
    assert case.wellpos_by_name() == {
        "INJECTOR": ((0, 0, 2), (0, 0, 3), (0, 0, 4)),
        "PRODUCER": ((24, 24, 0), (24, 24, 1), (24, 24, 2)),
    }


#---------------------------------------------------------------------------------------------------
def test_ix_input_regression_on_model_from_ixf_ix():
#---------------------------------------------------------------------------------------------------
    """Validate the shared metadata API against the INTERSECT regression case."""
    case = IX_input(Path("tests/Model_From_IXF_IX/Model_From_IXF.afi"))

    assert case.dim() == (25, 25, 5)
    assert case.wells_by_type() == {"WATER_INJECTOR": ["INJECTOR"], "PRODUCER": ["PRODUCER"]}
    assert case.wellpos_by_name() == {
        "INJECTOR": ((0, 0, 2), (0, 0, 3), (0, 0, 4)),
        "PRODUCER": ((24, 24, 0), (24, 24, 1), (24, 24, 2)),
    }
