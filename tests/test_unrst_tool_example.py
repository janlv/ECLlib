from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from itertools import islice

from ECLlib import BlockSpec, UNRST_file, unfmt_block


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "unrst_tool.py"


def _write_unrst(path: Path):
    with open(path, "wb") as file:
        for step in (0, 1):
            specs = (
                BlockSpec("SEQNUM", [step], "int"),
                BlockSpec("STARTSOL", [], "mess"),
                BlockSpec("TEMP", [10.0 + step], "float"),
                BlockSpec("PRESSURE", [100.0 + step], "float"),
                BlockSpec("ENDSOL", [], "mess"),
            )
            for spec in specs:
                file.write(unfmt_block.from_spec(spec).as_bytes())


def _section_keys(unrst: UNRST_file, index: int):
    return [block.key() for block in next(islice(unrst.section_blocks(), index, None))]


def test_unrst_tool_inspect_and_augment(tmp_path):
    src = tmp_path / "source.UNRST"
    out = tmp_path / "out.UNRST"
    spec_module = tmp_path / "spec_module.py"
    _write_unrst(src)
    spec_module.write_text(
        "from ECLlib import BlockSpec\n"
        "REPLACE_KEYS = ('TEMP',)\n"
        "def build_blocks(step, section):\n"
        "    return [BlockSpec('TEMP', [200.0 + step], 'float'), BlockSpec('XTRA', [step], 'int')]\n",
        encoding="utf8",
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

    augment = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "augment",
            str(src),
            str(out),
            str(spec_module),
            "--steps",
            "1",
            "--overwrite",
        ],
        capture_output=True,
        check=True,
        cwd=SCRIPT.parents[1],
        text=True,
    )
    assert "wrote=" in augment.stdout

    assert _section_keys(UNRST_file(out), 1) == [
        "SEQNUM",
        "STARTSOL",
        "PRESSURE",
        "TEMP",
        "XTRA",
        "ENDSOL",
    ]
