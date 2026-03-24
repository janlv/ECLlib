from __future__ import annotations

import subprocess
import sys
from itertools import islice
from pathlib import Path

from ECLlib import UNRST_file, unfmt_block


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "unrst_tool.py"


#---------------------------------------------------------------------------------------------------
def _write_unrst(path: Path, solution_specs=None):
#---------------------------------------------------------------------------------------------------
    """Write a small UNRST file used by the example CLI tests."""
    solution_specs = solution_specs or {}
    with open(path, "wb") as file:
        for step in (0, 1):
            specs = (
                ("SEQNUM", [step], "int"),
                ("STARTSOL", [], "mess"),
                ("TEMP", [10.0 + step], "float"),
                ("PRESSURE", [100.0 + step], "float"),
                *tuple(solution_specs.get(step, ())),
                ("ENDSOL", [], "mess"),
            )
            for key, data, dtype in specs:
                file.write(unfmt_block.from_data(key, data, dtype).as_bytes())


#---------------------------------------------------------------------------------------------------
def _section_keys(unrst: UNRST_file, index: int):
#---------------------------------------------------------------------------------------------------
    """Return ordered block keys for one section."""
    return [block.key() for block in next(islice(unrst.section_blocks(), index, None))]


#---------------------------------------------------------------------------------------------------
def test_unrst_tool_inspect_and_merge(tmp_path):
#---------------------------------------------------------------------------------------------------
    """The CLI script should inspect files and merge donor keys into a new output."""
    src = tmp_path / "source.UNRST"
    donor = tmp_path / "donor.UNRST"
    out = tmp_path / "out.UNRST"
    _write_unrst(src)
    _write_unrst(
        donor,
        solution_specs={
            0: (("KEY1", [1.0], "float"), ("KEY2", [10], "int")),
            1: (("KEY1", [2.0], "float"), ("KEY2", [11], "int")),
        },
    )

    inspect = subprocess.run(
        [sys.executable, str(SCRIPT), "inspect", str(src), "--steps", "1"],
        capture_output=True,
        check=True,
        cwd=SCRIPT.parents[1],
        text=True,
    )
    assert "sections=2" in inspect.stdout
    assert "step=1" in inspect.stdout
    assert "TEMP" in inspect.stdout

    merge = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "merge",
            str(src),
            str(donor),
            "KEY1",
            "KEY2",
            "--output",
            str(out),
            "--overwrite",
        ],
        capture_output=True,
        check=True,
        cwd=SCRIPT.parents[1],
        text=True,
    )
    assert "wrote=" in merge.stdout

    assert _section_keys(UNRST_file(out), 1) == [
        "SEQNUM",
        "STARTSOL",
        "TEMP",
        "PRESSURE",
        "KEY1",
        "KEY2",
        "ENDSOL",
    ]
