from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from struct import unpack

import pytest

from ECLlib.config import ENDIAN
from ECLlib import UNRST_file, unfmt_block

ROOT = Path(__file__).resolve().parents[2]
RECORDS_TXT = ROOT / "examples" / "format_templates" / "records.txt"

from ecllib_tools import unrst_records_txt
from ecllib_tools import unrst_tool


#---------------------------------------------------------------------------------------------------
def test_ecl_unrst_entry_point_target_exists():
#---------------------------------------------------------------------------------------------------
    """The installed ecl-unrst command should target the CLI main function."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'ecl-unrst = "ecllib_tools.unrst_tool:main"' in pyproject
    assert callable(unrst_tool.main)


#---------------------------------------------------------------------------------------------------
def _write_unrst(path: Path, *, steps=(0, 1, 2), days=None, solution_specs=None):
#---------------------------------------------------------------------------------------------------
    """Write a small UNRST file with DOUBHEAD days for records-tool tests."""
    days = tuple(steps if days is None else days)
    solution_specs = solution_specs or {}
    with open(path, "wb") as file:
        for step, day in zip(steps, days, strict=True):
            specs = (
                ("SEQNUM", [step], "int"),
                ("INTEHEAD", [0, 0, 2], "int"),
                ("DOUBHEAD", [float(day)], "double"),
                ("STARTSOL", [], "mess"),
                ("TEMP", [float(step), float(step) + 0.25], "float"),
                ("PRESSURE", [100.0 + step], "float"),
                *tuple(solution_specs.get(step, ())),
                ("ENDSOL", [], "mess"),
            )
            for key, data, dtype in specs:
                file.write(unfmt_block.from_data(key, data, dtype).as_bytes())


#---------------------------------------------------------------------------------------------------
def _write_records(path: Path, text: str):
#---------------------------------------------------------------------------------------------------
    """Write records text with normalized newlines."""
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


#---------------------------------------------------------------------------------------------------
def _records_text(*times):
#---------------------------------------------------------------------------------------------------
    """Return records text with two values for each requested time."""
    chunks = ["NBLOCK 2"]
    for time in times:
        chunks.extend(
            (
                "",
                f"TIME = {time} d",
                "",
                "WATER SATURATIONS",
                "",
                "0.1 0.2",
                "",
                "OIL SATURATIONS",
                "",
                "0.9 0.8",
            )
        )
    return "\n".join(chunks)


#---------------------------------------------------------------------------------------------------
def _section_bytes(unrst: UNRST_file):
#---------------------------------------------------------------------------------------------------
    """Return serialized bytes for each section."""
    return [b"".join(block.binarydata() for block in section) for section in unrst.section_blocks()]


#---------------------------------------------------------------------------------------------------
def _section_snapshot(unrst: UNRST_file, index: int):
#---------------------------------------------------------------------------------------------------
    """Return ordered keys and decoded data for one section."""
    for section_index, section in enumerate(unrst.section_blocks(use_mmap=False)):
        if section_index != index:
            continue
        keys = [block.key() for block in section]
        data = {
            block.key(): block.data().tolist()
            for block in section
            if block.key() not in {"SEQNUM", "INTEHEAD", "DOUBHEAD", "STARTSOL", "ENDSOL"}
        }
        return keys, data
    raise IndexError(index)


#---------------------------------------------------------------------------------------------------
def _cli_env():
#---------------------------------------------------------------------------------------------------
    """Return an environment that imports ECLlib from this checkout."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), env.get("PYTHONPATH", "")))
    return env


#---------------------------------------------------------------------------------------------------
def _cli_args(*args):
#---------------------------------------------------------------------------------------------------
    """Return command arguments for running the installed UNRST CLI module."""
    return [sys.executable, "-m", "ecllib_tools.unrst_tool", *map(str, args)]


#---------------------------------------------------------------------------------------------------
def _block_header(data):
#---------------------------------------------------------------------------------------------------
    """Return ``(key, length, type)`` from serialized unformatted block bytes."""
    key, length, dtype = unpack(ENDIAN + "8si4s", data[4:20])
    return key.decode().strip(), length, dtype.decode()


#---------------------------------------------------------------------------------------------------
def _stdout_pairs(result):
#---------------------------------------------------------------------------------------------------
    """Return CLI summary output as a key-value dictionary."""
    return dict(line.split("=", 1) for line in result.stdout.strip().splitlines())


#---------------------------------------------------------------------------------------------------
def test_parse_real_records_file():
#---------------------------------------------------------------------------------------------------
    """Parse the checked records.txt format template."""
    keys, rows = unrst_records_txt.parse_records_txt(RECORDS_TXT)
    row_tuple = tuple(rows)
    first_time, first_blocks = row_tuple[0]

    assert len(row_tuple) == 6
    assert first_time == 0
    assert keys == ("WATER_SA", "OIL_SATU", "GAS_SATU")
    assert [_block_header(block) for block in first_blocks] == [
        ("WATER_SA", 20, "REAL"),
        ("OIL_SATU", 20, "REAL"),
        ("GAS_SATU", 20, "REAL"),
    ]


#---------------------------------------------------------------------------------------------------
def test_records_txt_wrapper_returns_keys_and_row_generator(tmp_path):
#---------------------------------------------------------------------------------------------------
    """The built-in records-txt wrapper should return keys and a streaming row iterable."""
    records = _write_records(tmp_path / "records.txt", _records_text(0, 2))

    keys, rows = unrst_records_txt.parse_records_txt(records)

    assert keys == ("WATER_SA", "OIL_SATU")
    assert iter(rows) is rows
    assert [(time, tuple(_block_header(block) for block in row)) for time, row in rows] == [
        (0.0, (("WATER_SA", 2, "REAL"), ("OIL_SATU", 2, "REAL"))),
        (2.0, (("WATER_SA", 2, "REAL"), ("OIL_SATU", 2, "REAL"))),
    ]


#---------------------------------------------------------------------------------------------------
def test_records_txt_wrapper_streams_without_read_text(tmp_path, monkeypatch):
#---------------------------------------------------------------------------------------------------
    """The records-txt parser should stream from the file instead of reading it all at once."""
    records = _write_records(tmp_path / "records.txt", _records_text(0, 2))

    def fail_read_text(*args, **kwargs):
        """Fail if the parser uses whole-file text loading."""
        raise AssertionError("read_text should not be used by records-txt")

    monkeypatch.setattr(Path, "read_text", fail_read_text)
    keys, rows = unrst_records_txt.parse_records_txt(records)

    assert keys == ("WATER_SA", "OIL_SATU")
    assert [time for time, _ in rows] == [0.0, 2.0]


#---------------------------------------------------------------------------------------------------
def test_records_txt_accepts_case_insensitive_nblock_and_time(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Accept mixed-case NBLOCK and TIME markers while preserving records data."""
    records = _write_records(
        tmp_path / "records.txt",
        """
        nblock 2
        tImE = 0 D
        WATER SATURATIONS
        0.1 0.2
        OIL SATURATIONS
        0.9 0.8
        TiMe = 1 d
        WATER SATURATIONS
        0.3 0.4
        OIL SATURATIONS
        0.7 0.6
        """,
    )

    keys, rows = unrst_records_txt.parse_records_txt(records)

    assert keys == ("WATER_SA", "OIL_SATU")
    assert [time for time, _ in rows] == [0.0, 1.0]


#---------------------------------------------------------------------------------------------------
def test_records_txt_rejects_malformed_nblock(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Reject NBLOCK lines whose count is not an integer."""
    records = _write_records(
        tmp_path / "records.txt",
        """
        NBLOCK two
        TIME = 0 d
        WATER SATURATIONS
        0.1
        """,
    )

    with pytest.raises(ValueError):
        unrst_records_txt.parse_records_txt(records)


#---------------------------------------------------------------------------------------------------
def test_records_txt_rejects_time_with_wrong_unit(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Reject TIME markers that do not use the strict days unit."""
    records = _write_records(
        tmp_path / "records.txt",
        """
        NBLOCK 1
        TIME = 0 days
        WATER SATURATIONS
        0.1
        """,
    )

    with pytest.raises(ValueError, match="Expected TIME unit 'd'"):
        unrst_records_txt.parse_records_txt(records)


#---------------------------------------------------------------------------------------------------
def test_records_txt_detector_uses_content_not_name(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Detect the records text format from the first markers, regardless of filename."""
    records = _write_records(tmp_path / "case_export.out", _records_text(0))
    unrelated = _write_records(tmp_path / "records.txt", "not a records text file")

    assert unrst_records_txt.is_records_txt(records)
    assert not unrst_records_txt.is_records_txt(unrelated)


#---------------------------------------------------------------------------------------------------
def test_merge_records_inserts_blocks_in_matching_sections(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Insert parsed records before ENDSOL and leave unmatched sections unchanged."""
    source = tmp_path / "source.UNRST"
    records = _write_records(tmp_path / "records.txt", _records_text(0, 2))
    output = tmp_path / "out.UNRST"
    _write_unrst(source)
    before_file = source.read_bytes()
    before_sections = _section_bytes(UNRST_file(source))

    written, keys, sections = unrst_tool.merge_records_to_unrst(source, records, output=output)

    assert written == output
    assert keys == ("WATER_SA", "OIL_SATU")
    assert sections == 2
    assert output.is_file()
    assert source.read_bytes() == before_file

    after = UNRST_file(output)
    after_sections = _section_bytes(after)
    assert after_sections[1] == before_sections[1]
    assert after_sections[0] != before_sections[0]
    assert after_sections[2] != before_sections[2]

    keys, data = _section_snapshot(after, 0)
    assert keys == [
        "SEQNUM",
        "INTEHEAD",
        "DOUBHEAD",
        "STARTSOL",
        "TEMP",
        "PRESSURE",
        "WATER_SA",
        "OIL_SATU",
        "ENDSOL",
    ]
    assert data["WATER_SA"] == pytest.approx([0.1, 0.2])
    assert data["OIL_SATU"] == pytest.approx([0.9, 0.8])


#---------------------------------------------------------------------------------------------------
def test_merge_records_delegates_to_merge_keys_from_blocks(tmp_path, monkeypatch):
#---------------------------------------------------------------------------------------------------
    """The records helper should pass parsed keys and rows to the UNRST block merger."""
    source = tmp_path / "source.UNRST"
    records = _write_records(tmp_path / "records.txt", _records_text(0))
    output = tmp_path / "out.UNRST"
    _write_unrst(source, steps=(0,))
    calls = {}

    def fake_merge(self, *, keys, rows, name=None, overwrite=False, tolerance=1e-6, endblock=None):
        """Capture merge arguments without writing an output file."""
        calls["keys"] = keys
        calls["rows"] = tuple(rows)
        calls["name"] = name
        calls["overwrite"] = overwrite
        calls["tolerance"] = tolerance
        return UNRST_file(output)

    monkeypatch.setattr(UNRST_file, "merge_keys_from_blocks", fake_merge)

    written, keys, sections = unrst_tool.merge_records_to_unrst(
        source,
        records,
        output=output,
        overwrite=True,
        time_tolerance=1e-5,
    )

    assert written == output
    assert keys == ("WATER_SA", "OIL_SATU")
    assert sections == 1
    assert calls["keys"] == ("WATER_SA", "OIL_SATU")
    assert [time for time, _ in calls["rows"]] == [0.0]
    assert calls["name"] == output
    assert calls["overwrite"] is True
    assert calls["tolerance"] == pytest.approx(1e-5)


#---------------------------------------------------------------------------------------------------
def test_merge_records_rejects_existing_target_keys(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Reject appending a key that already exists in the matched section."""
    source = tmp_path / "source.UNRST"
    records = _write_records(tmp_path / "records.txt", _records_text(0))
    output = tmp_path / "out.UNRST"
    _write_unrst(source, steps=(0,), solution_specs={0: (("WATER_SA", [9.0], "float"),)})

    with pytest.raises(ValueError, match="already contains"):
        unrst_tool.merge_records_to_unrst(source, records, output=output)

    assert not output.exists()


#---------------------------------------------------------------------------------------------------
def test_merge_records_rejects_missing_time_match(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Reject records whose TIME value does not match a UNRST DOUBHEAD day."""
    source = tmp_path / "source.UNRST"
    records = _write_records(tmp_path / "records.txt", _records_text(10))
    output = tmp_path / "out.UNRST"
    _write_unrst(source, steps=(0,))

    with pytest.raises(ValueError, match="No UNRST section matches"):
        unrst_tool.merge_records_to_unrst(source, records, output=output)

    assert not output.exists()


#---------------------------------------------------------------------------------------------------
def test_merge_records_matches_with_time_tolerance(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Match record and UNRST times within the configured tolerance."""
    source = tmp_path / "source.UNRST"
    records = _write_records(tmp_path / "records.txt", _records_text("1.0000004"))
    output = tmp_path / "out.UNRST"
    _write_unrst(source, steps=(7,), days=(1.0,))

    written, keys, sections = unrst_tool.merge_records_to_unrst(
        source,
        records,
        output=output,
        time_tolerance=1e-5,
    )

    assert keys == ("WATER_SA", "OIL_SATU")
    assert sections == 1
    keys, data = _section_snapshot(UNRST_file(written), 0)
    assert keys[-3:] == ["WATER_SA", "OIL_SATU", "ENDSOL"]
    assert data["WATER_SA"] == pytest.approx([0.1, 0.2])


#---------------------------------------------------------------------------------------------------
def test_merge_records_does_not_use_section_blocks(tmp_path, monkeypatch):
#---------------------------------------------------------------------------------------------------
    """The optimized records path should stream blocks directly."""
    source = tmp_path / "source.UNRST"
    records = _write_records(tmp_path / "records.txt", _records_text(0))
    output = tmp_path / "out.UNRST"
    _write_unrst(source, steps=(0,))

    def fail(*args, **kwargs):
        """Fail if merge falls back to section tuple materialization."""
        raise AssertionError("section_blocks should not be used by merge")

    with monkeypatch.context() as patcher:
        patcher.setattr(UNRST_file, "section_blocks", fail)
        written, keys, sections = unrst_tool.merge_records_to_unrst(source, records, output=output)

    assert keys == ("WATER_SA", "OIL_SATU")
    assert sections == 1
    keys, data = _section_snapshot(UNRST_file(written), 0)
    assert keys[-3:] == ["WATER_SA", "OIL_SATU", "ENDSOL"]
    assert data["OIL_SATU"] == pytest.approx([0.9, 0.8])


#---------------------------------------------------------------------------------------------------
def test_merge_records_rejects_nblock_mismatch(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Reject record arrays with lengths different from NBLOCK."""
    records = _write_records(
        tmp_path / "records.txt",
        """
        NBLOCK 2
        TIME = 0 d
        WATER SATURATIONS
        0.1
        """,
    )

    with pytest.raises(ValueError, match="expected 2"):
        unrst_records_txt.parse_records_txt(records)


#---------------------------------------------------------------------------------------------------
def test_merge_records_cli_default_output(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Write the default records output path from the command line."""
    source = tmp_path / "cli.UNRST"
    records = _write_records(tmp_path / "records.txt", _records_text(0))
    expected = tmp_path / "cli_RECORDS.UNRST"
    _write_unrst(source, steps=(0,))

    result = subprocess.run(
        _cli_args("merge", source, records),
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        check=True,
        text=True,
    )

    summary = _stdout_pairs(result)
    assert expected.is_file()
    assert summary == {
        "wrote": str(expected),
        "mode": "records",
        "source": str(records),
        "keys": "WATER_SA,OIL_SATU",
        "sections": "1",
    }


#---------------------------------------------------------------------------------------------------
def test_merge_records_cli_explicit_output(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Write an explicit records output path from the command line."""
    source = tmp_path / "cli.UNRST"
    records = _write_records(tmp_path / "records.txt", _records_text(0))
    output = tmp_path / "custom.UNRST"
    _write_unrst(source, steps=(0,))

    result = subprocess.run(
        _cli_args("merge", source, records, "-o", output),
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        check=True,
        text=True,
    )

    summary = _stdout_pairs(result)
    assert output.is_file()
    assert summary == {
        "wrote": str(output),
        "mode": "records",
        "source": str(records),
        "keys": "WATER_SA,OIL_SATU",
        "sections": "1",
    }


#---------------------------------------------------------------------------------------------------
def test_merge_records_cli_accepts_records_content_with_any_source_name(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Accept records text content even when the file is not named records.txt."""
    source = tmp_path / "cli.UNRST"
    records = _write_records(tmp_path / "values.txt", _records_text(0))
    expected = tmp_path / "cli_VALUES.UNRST"
    _write_unrst(source, steps=(0,))

    result = subprocess.run(
        _cli_args("merge", source, records),
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        check=True,
        text=True,
    )

    summary = _stdout_pairs(result)
    assert expected.is_file()
    assert summary == {
        "wrote": str(expected),
        "mode": "records",
        "source": str(records),
        "keys": "WATER_SA,OIL_SATU",
        "sections": "1",
    }


#---------------------------------------------------------------------------------------------------
def test_merge_records_rejects_unsupported_source_format(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Reject unknown merge sources when no donor keys are provided."""
    source = tmp_path / "cli.UNRST"
    records = _write_records(tmp_path / "values.txt", "not a records text file")
    _write_unrst(source, steps=(0,))

    result = subprocess.run(
        _cli_args("merge", source, records),
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "donor UNRST merge requires KEY" in result.stderr


#---------------------------------------------------------------------------------------------------
def test_merge_records_cli_rejects_extra_keys_for_records_source(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Reject donor-key arguments when the source is records text."""
    source = tmp_path / "cli.UNRST"
    records = _write_records(tmp_path / "records.txt", _records_text(0))
    _write_unrst(source, steps=(0,))

    result = subprocess.run(
        _cli_args("merge", source, records, "SWAT"),
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "Records text merge does not take key arguments" in result.stderr


#---------------------------------------------------------------------------------------------------
def test_merge_records_cli_command_is_removed(tmp_path):
#---------------------------------------------------------------------------------------------------
    """The old format-specific CLI command should no longer be accepted."""
    source = tmp_path / "cli.UNRST"
    records = _write_records(tmp_path / "records.txt", _records_text(0))
    _write_unrst(source, steps=(0,))

    result = subprocess.run(
        _cli_args("merge-format", source, records),
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


#---------------------------------------------------------------------------------------------------
def test_append_records_cli_command_is_removed(tmp_path):
#---------------------------------------------------------------------------------------------------
    """The old records-specific CLI command should no longer be accepted."""
    source = tmp_path / "cli.UNRST"
    records = _write_records(tmp_path / "records.txt", _records_text(0))
    _write_unrst(source, steps=(0,))

    result = subprocess.run(
        _cli_args("append-records", source, records),
        cwd=ROOT,
        env=_cli_env(),
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode != 0
    assert "invalid choice" in result.stderr
