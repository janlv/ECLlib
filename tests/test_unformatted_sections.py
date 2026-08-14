from __future__ import annotations

import numpy as np
import pytest

from ECLlib import AutoRefreshIterator, unfmt_block, unfmt_file
from ECLlib.core import unformatted as unformatted_module


#---------------------------------------------------------------------------------------------------
def _serialized_block(key, value=None):
#---------------------------------------------------------------------------------------------------
    """Return one integer or message block serialized for a test file."""
    if value is None:
        return unfmt_block.from_data(key, [], "mess").as_bytes()
    return unfmt_block.from_data(key, [value], "int").as_bytes()


#---------------------------------------------------------------------------------------------------
def _write_section_file(path, sections, *, partial=()):
#---------------------------------------------------------------------------------------------------
    """Write complete start/end-delimited sections and an optional partial section."""
    with open(path, "wb") as file:
        for step, key, value in sections:
            file.write(_serialized_block("SEQNUM", step))
            file.write(_serialized_block(key, value))
            file.write(_serialized_block("ENDSOL"))
        for key, value in partial:
            file.write(_serialized_block(key, value))


#---------------------------------------------------------------------------------------------------
def _section_reader(path):
#---------------------------------------------------------------------------------------------------
    """Return a generic unformatted reader configured with restart section markers."""
    reader = unfmt_file(path)
    reader.start = "SEQNUM"
    reader.end = "ENDSOL"
    return reader


#---------------------------------------------------------------------------------------------------
def _section_keys(reader, **kwargs):
#---------------------------------------------------------------------------------------------------
    """Materialize section keys while their borrowed backing source remains open."""
    return [[block.key() for block in section] for section in reader.section_blocks(**kwargs)]


#---------------------------------------------------------------------------------------------------
def test_section_blocks_preserves_forward_and_tail_order(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Preserve section order and reverse each tuple when reading from the tail."""
    path = tmp_path / "ordered.UNRST"
    _write_section_file(path, [(1, "PRESSURE", 10), (2, "SWAT", 20)])
    reader = _section_reader(path)

    assert _section_keys(reader) == [
        ["SEQNUM", "PRESSURE", "ENDSOL"],
        ["SEQNUM", "SWAT", "ENDSOL"],
    ]
    assert _section_keys(reader, tail=True) == [
        ["ENDSOL", "SWAT", "SEQNUM"],
        ["ENDSOL", "PRESSURE", "SEQNUM"],
    ]
    assert list(reader.section_start_indices()) == [0, 3, 6]
    assert list(reader.section_start_indices(tail=True)) == [0, 3, 6]


#---------------------------------------------------------------------------------------------------
def test_section_blocks_reads_source_and_headers_once(tmp_path, monkeypatch):
#---------------------------------------------------------------------------------------------------
    """Group sections using one source iterator and one visit per block header."""
    path = tmp_path / "single-pass.UNRST"
    _write_section_file(path, [(1, "PRESSURE", 10), (2, "SWAT", 20)])
    reader = _section_reader(path)
    source_calls = 0
    header_calls = 0
    original_blocks = reader.blocks
    original_read_header = reader.read_header

    def counted_blocks(**kwargs):
        """Count construction of the underlying forward block source."""
        nonlocal source_calls
        source_calls += 1
        yield from original_blocks(**kwargs)

    def counted_read_header(data, startpos):
        """Count each parsed unformatted block header."""
        nonlocal header_calls
        header_calls += 1
        return original_read_header(data, startpos)

    monkeypatch.setattr(reader, "blocks", counted_blocks)
    monkeypatch.setattr(reader, "read_header", counted_read_header)

    assert len(_section_keys(reader)) == 2
    assert source_calls == 1
    assert header_calls == 6


#---------------------------------------------------------------------------------------------------
def test_use_mmap_false_avoids_mmap_for_forward_and_tail(tmp_path, monkeypatch):
#---------------------------------------------------------------------------------------------------
    """Use seek-based sources in both directions when memory mapping is disabled."""
    path = tmp_path / "without-mmap.UNRST"
    _write_section_file(path, [(1, "PRESSURE", 10), (2, "SWAT", 20)])
    reader = _section_reader(path)

    def fail_mmap(*args, **kwargs):
        """Fail if a disabled mmap path attempts to construct a mapping."""
        raise AssertionError("mmap must not be used")

    monkeypatch.setattr(unformatted_module, "mmap", fail_mmap)

    assert len(_section_keys(reader, use_mmap=False)) == 2
    assert len(_section_keys(reader, tail=True, use_mmap=False)) == 2


#---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_mmap", [True, False])
def test_tail_blocks_empty_sources_and_zero_start_are_iterators(tmp_path, use_mmap):
#---------------------------------------------------------------------------------------------------
    """Return exhausted iterators for unavailable data and an explicit zero boundary."""
    missing = tmp_path / "missing.UNRST"
    empty = tmp_path / "empty.UNRST"
    short = tmp_path / "short.UNRST"
    valid = tmp_path / "valid.UNRST"
    empty.touch()
    short.write_bytes(b"short")
    _write_section_file(valid, [(1, "PRESSURE", 10)])

    for path in (missing, empty, short):
        assert next(_section_reader(path).tail_blocks(use_mmap=use_mmap), None) is None
    assert next(_section_reader(valid).tail_blocks(start=0, use_mmap=use_mmap), None) is None


#---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_mmap", [True, False])
def test_tail_blocks_rejects_truncated_multi_record_payload(tmp_path, use_mmap):
#---------------------------------------------------------------------------------------------------
    """Fail closed on a truncated tail block and accept it after all payload records arrive."""
    path = tmp_path / "truncated-tail.UNRST"
    values = np.arange(1505, dtype=np.int32)
    terminal_bytes = unfmt_block.from_data("CONNXT", values, "int").as_bytes()
    with open(path, "wb") as file:
        file.write(_serialized_block("ENDSOL"))
        file.write(terminal_bytes[:24])

    reader = _section_reader(path)
    assert next(reader.tail_blocks(use_mmap=use_mmap), None) is None

    with open(path, "ab") as file:
        file.write(terminal_bytes[24:])

    blocks = reader.tail_blocks(use_mmap=use_mmap)
    block = next(blocks)
    assert block.key() == "CONNXT"
    assert np.array_equal(block.data(), values)
    blocks.close()


#---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_mmap", [True, False])
def test_final_section_payload_source_lives_while_yielded(tmp_path, use_mmap):
#---------------------------------------------------------------------------------------------------
    """Keep the final section's borrowed payload source open until iteration resumes."""
    path = tmp_path / "payload-lifetime.UNRST"
    _write_section_file(path, [(1, "PRESSURE", 123)])
    sections = _section_reader(path).section_blocks(use_mmap=use_mmap)

    section = next(sections)
    payload = section[1]
    source = payload._data if use_mmap else payload._file_obj
    assert source.closed is False
    assert payload.data().tolist() == [123]

    sections.close()
    assert source.closed is True


#---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_mmap", [True, False])
def test_section_blocks_early_close_closes_borrowed_source(tmp_path, use_mmap):
#---------------------------------------------------------------------------------------------------
    """Close the block source deterministically when a section consumer stops early."""
    path = tmp_path / "early-close.UNRST"
    _write_section_file(path, [(1, "PRESSURE", 10), (2, "SWAT", 20)])
    sections = _section_reader(path).section_blocks(use_mmap=use_mmap)

    section = next(sections)
    payload = section[1]
    source = payload._data if use_mmap else payload._file_obj
    assert source.closed is False

    sections.close()
    assert source.closed is True


#---------------------------------------------------------------------------------------------------
def test_static_section_blocks_still_yields_partial_final_section(tmp_path):
#---------------------------------------------------------------------------------------------------
    """Retain static grouping semantics for a final section without its end marker."""
    path = tmp_path / "static-partial.UNRST"
    _write_section_file(path, [], partial=[("SEQNUM", 1), ("PRESSURE", 10)])

    assert _section_keys(_section_reader(path), use_mmap=False) == [
        ["SEQNUM", "PRESSURE"]
    ]


#---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_mmap", [True, False])
@pytest.mark.parametrize("only_new", [False, True])
def test_no_end_final_section_with_trailing_bytes_keeps_payload_source_open(
        tmp_path, use_mmap, only_new):
#---------------------------------------------------------------------------------------------------
    """Reborrow a final section when invalid trailing bytes exhaust its original source."""
    path = tmp_path / "no-end-trailing-bytes.UNRST"
    with open(path, "wb") as file:
        file.write(_serialized_block("START", 1))
        file.write(_serialized_block("VALUE", 10))
        complete_end = file.tell()
        file.write(b"short")

    reader = unfmt_file(path)
    reader.start = "START"
    sections = reader.section_blocks(only_new=only_new, use_mmap=use_mmap)
    section = next(sections)
    payload = section[1]
    assert payload.data().tolist() == [10]
    assert payload._data is None
    source = payload._file_obj
    assert source.closed is False
    assert reader._endpos == (complete_end if only_new else 0)

    sections.close()
    assert source.closed is True


#---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_mmap", [True, False])
def test_only_new_withholds_partial_append_and_emits_once_when_complete(tmp_path, use_mmap):
#---------------------------------------------------------------------------------------------------
    """Commit the only-new cursor only after an appended section receives its end marker."""
    path = tmp_path / "appending.UNRST"
    _write_section_file(
        path,
        [(1, "PRESSURE", 10)],
        partial=[("SEQNUM", 2), ("SWAT", 20)],
    )
    reader = _section_reader(path)
    complete_size = sum(
        len(_serialized_block(key, value))
        for key, value in (("SEQNUM", 1), ("PRESSURE", 10), ("ENDSOL", None))
    )

    assert _section_keys(reader, only_new=True, use_mmap=use_mmap) == [
        ["SEQNUM", "PRESSURE", "ENDSOL"]
    ]
    assert reader._endpos == complete_size

    assert _section_keys(reader, only_new=True, use_mmap=use_mmap) == []
    assert reader._endpos == complete_size

    with open(path, "ab") as file:
        file.write(_serialized_block("ENDSOL"))

    assert _section_keys(reader, only_new=True, use_mmap=use_mmap) == [
        ["SEQNUM", "SWAT", "ENDSOL"]
    ]
    assert reader._endpos == path.stat().st_size
    assert _section_keys(reader, only_new=True, use_mmap=use_mmap) == []


#---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_mmap", [True, False])
def test_only_new_complete_section_early_close_resumes_at_next_section(tmp_path, use_mmap):
#---------------------------------------------------------------------------------------------------
    """Commit a yielded complete section so reopening resumes at the following section."""
    path = tmp_path / "complete-early-close.UNRST"
    _write_section_file(path, [(1, "PRESSURE", 10), (2, "SWAT", 20)])
    reader = _section_reader(path)

    sections = reader.section_blocks(only_new=True, use_mmap=use_mmap)
    first = next(sections)
    first_end = first[-1].endpos
    assert [block.key() for block in first] == ["SEQNUM", "PRESSURE", "ENDSOL"]
    assert reader._endpos == first_end
    sections.close()

    assert _section_keys(reader, only_new=True, use_mmap=use_mmap) == [
        ["SEQNUM", "SWAT", "ENDSOL"]
    ]
    assert reader._endpos == path.stat().st_size


#---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_mmap", [True, False])
def test_only_new_nested_start_before_end_leaves_cursor_unchanged(tmp_path, use_mmap):
#---------------------------------------------------------------------------------------------------
    """Withhold malformed nested sections without committing past the first start marker."""
    path = tmp_path / "nested-start.UNRST"
    with open(path, "wb") as file:
        for key, value in (
            ("SEQNUM", 1),
            ("PRESSURE", 10),
            ("SEQNUM", 2),
            ("SWAT", 20),
            ("ENDSOL", None),
        ):
            file.write(_serialized_block(key, value))

    reader = _section_reader(path)
    assert _section_keys(reader, only_new=True, use_mmap=use_mmap) == []
    assert reader._endpos == 0


#---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_mmap", [True, False])
def test_only_new_withholds_terminal_block_with_truncated_payload(tmp_path, use_mmap):
#---------------------------------------------------------------------------------------------------
    """Do not commit a declared terminal block until its nonempty payload is complete."""
    path = tmp_path / "truncated-terminal.UNRST"
    start_bytes = _serialized_block("TIME", 1)
    terminal_bytes = _serialized_block("CONNXT", 99)
    with open(path, "wb") as file:
        file.write(start_bytes)
        file.write(terminal_bytes[:24])

    reader = unfmt_file(path)
    reader.start = "TIME"
    reader.end = "CONNXT"

    assert _section_keys(reader, only_new=True, use_mmap=use_mmap) == []
    assert reader._endpos == 0

    with open(path, "ab") as file:
        file.write(terminal_bytes[24:])

    assert _section_keys(reader, only_new=True, use_mmap=use_mmap) == [
        ["TIME", "CONNXT"]
    ]
    assert reader._endpos == path.stat().st_size
    assert _section_keys(reader, only_new=True, use_mmap=use_mmap) == []


#---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_mmap", [True, False])
def test_only_new_without_end_rereads_next_start_after_early_close(tmp_path, use_mmap):
#---------------------------------------------------------------------------------------------------
    """Commit at the next start marker so closing early cannot discard the following section."""
    path = tmp_path / "no-end-early-close.UNRST"
    with open(path, "wb") as file:
        for key, value in (("START", 1), ("VALUE", 10), ("START", 2), ("VALUE", 20)):
            file.write(_serialized_block(key, value))

    reader = unfmt_file(path)
    reader.start = "START"
    sections = reader.section_blocks(only_new=True, use_mmap=use_mmap)
    first = next(sections)
    second_start = first[-1].endpos
    assert [block.key() for block in first] == ["START", "VALUE"]
    assert reader._endpos == second_start
    sections.close()

    assert _section_keys(reader, only_new=True, use_mmap=use_mmap) == [
        ["START", "VALUE"]
    ]
    assert reader._endpos == path.stat().st_size


#---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("use_mmap", [True, False])
def test_only_new_without_end_does_not_commit_truncated_next_start(tmp_path, use_mmap):
#---------------------------------------------------------------------------------------------------
    """Commit only complete blocks when the next start marker has a truncated payload."""
    path = tmp_path / "no-end-truncated-start.UNRST"
    second_start = _serialized_block("START", 2)
    with open(path, "wb") as file:
        file.write(_serialized_block("START", 1))
        file.write(_serialized_block("VALUE", 10))
        expected_cursor = file.tell()
        file.write(second_start[:24])

    reader = unfmt_file(path)
    reader.start = "START"
    assert _section_keys(reader, only_new=True, use_mmap=use_mmap) == [
        ["START", "VALUE"]
    ]
    assert reader._endpos == expected_cursor

    with open(path, "ab") as file:
        file.write(second_start[24:])
        file.write(_serialized_block("VALUE", 20))

    assert _section_keys(reader, only_new=True, use_mmap=use_mmap) == [
        ["START", "VALUE"]
    ]
    assert reader._endpos == path.stat().st_size


#---------------------------------------------------------------------------------------------------
def test_auto_refresh_iterator_closes_exhausted_sources_and_stays_closed():
#---------------------------------------------------------------------------------------------------
    """Close refreshed generators exactly once and never reopen after explicit close."""
    events = []
    factory_calls = 0

    def factory(*, only_new=False):
        """Create one populated source followed by empty refreshed sources."""
        nonlocal factory_calls
        index = factory_calls
        factory_calls += 1
        if index:
            assert events == [f"closed-{previous}" for previous in range(index)]

        def source():
            """Yield the initial value and record deterministic generator closure."""
            try:
                if index == 0:
                    yield 7
            finally:
                events.append(f"closed-{index}")

        return source()

    iterator = AutoRefreshIterator(factory)
    assert next(iterator) == 7
    with pytest.raises(StopIteration):
        next(iterator)
    assert events == ["closed-0", "closed-1"]
    assert factory_calls == 2

    with pytest.raises(StopIteration):
        next(iterator)
    assert events == ["closed-0", "closed-1", "closed-2"]
    assert factory_calls == 3

    iterator.close()
    iterator.close()
    iterator._refresh()
    with pytest.raises(StopIteration):
        next(iterator)
    assert events == ["closed-0", "closed-1", "closed-2"]
    assert factory_calls == 3
