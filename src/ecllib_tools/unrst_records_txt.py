"""Parse records text files into UNRST merge rows."""
from __future__ import annotations

import re

from ECLlib import File, unfmt_block


#---------------------------------------------------------------------------------------------------
def normalize_record_key(heading):
#---------------------------------------------------------------------------------------------------
    """Return the UNRST keyword derived from a records heading."""
    key = re.sub(r"\s+", "_", heading.strip())[:8]
    if not key:
        raise ValueError("Empty record heading")
    return key


#---------------------------------------------------------------------------------------------------
def _lines(path):
#---------------------------------------------------------------------------------------------------
    """Yield non-empty stripped text lines."""
    for line in File(path).lines():
        if stripped := line.strip():
            yield stripped


#---------------------------------------------------------------------------------------------------
def _nblock(line):
#---------------------------------------------------------------------------------------------------
    """Return the NBLOCK count from the first records text line."""
    words = line.split()
    if len(words) != 2 or words[0].upper() != "NBLOCK":
        raise ValueError("records text must start with 'NBLOCK <count>'")
    return int(words[1])


#---------------------------------------------------------------------------------------------------
def _time(line):
#---------------------------------------------------------------------------------------------------
    """Return the day from a TIME line, or ``None`` when ``line`` is not a TIME marker."""
    key, separator, value = line.partition("=")
    if key.strip().upper() != "TIME" or not separator:
        return None
    value, unit = value.split()
    if unit.lower() != "d":
        raise ValueError(f"Expected TIME unit 'd', got {unit!r}")
    return float(value)


#---------------------------------------------------------------------------------------------------
def _block_bytes(keys, values):
#---------------------------------------------------------------------------------------------------
    """Return serialized float blocks for one records time group."""
    return tuple(
        unfmt_block.from_data(key, value, "float").as_bytes()
        for key, value in zip(keys, values, strict=True)
    )


#---------------------------------------------------------------------------------------------------
def _values(lines, heading, nblock):
#---------------------------------------------------------------------------------------------------
    """Read one NBLOCK-sized value array after ``heading``."""
    values = []
    while len(values) < nblock:
        try:
            values.extend(float(value) for value in next(lines).split())
        except StopIteration as error:
            raise ValueError(f"{heading!r} has {len(values)} values; expected {nblock}") from error
    if len(values) != nblock:
        raise ValueError(f"{heading!r} has {len(values)} values; expected {nblock}")
    return tuple(values)


#---------------------------------------------------------------------------------------------------
def _record_group(lines, time, nblock):
#---------------------------------------------------------------------------------------------------
    """Read one TIME group and return its keys, values, and the next TIME value."""
    keys = []
    arrays = []
    for line in lines:
        if (next_time := _time(line)) is not None:
            return tuple(keys), tuple(arrays), next_time
        keys.append(normalize_record_key(line))
        arrays.append(_values(lines, line, nblock))
    return tuple(keys), tuple(arrays), None


#---------------------------------------------------------------------------------------------------
def _first_time(path, lines):
#---------------------------------------------------------------------------------------------------
    """Return the first TIME value in the records stream."""
    for line in lines:
        if (time := _time(line)) is not None:
            return time
    raise ValueError(f"{path} contains no TIME records")


#---------------------------------------------------------------------------------------------------
def _first_nblock(lines):
#---------------------------------------------------------------------------------------------------
    """Return the NBLOCK value from the records stream."""
    try:
        return _nblock(next(lines))
    except StopIteration as error:
        raise ValueError("records text must start with 'NBLOCK <count>'") from error


#---------------------------------------------------------------------------------------------------
def is_records_txt(path):
#---------------------------------------------------------------------------------------------------
    """Return whether ``path`` starts like the supported records text format."""
    lines = _lines(path)
    try:
        _nblock(next(lines))
        return _time(next(lines)) is not None
    except (StopIteration, UnicodeDecodeError, ValueError):
        return False


#---------------------------------------------------------------------------------------------------
def parse_records_txt(path):
#---------------------------------------------------------------------------------------------------
    """Return keys and streaming serialized rows for the NBLOCK/TIME records text format."""
    lines = _lines(path)
    nblock = _first_nblock(lines)
    first_time = _first_time(path, lines)
    keys, first_values, next_time = _record_group(lines, first_time, nblock)

    def value_rows():
        """Yield serialized records rows."""
        nonlocal next_time
        yield first_time, _block_bytes(keys, first_values)
        while next_time is not None:
            time = next_time
            _keys, values, next_time = _record_group(lines, time, nblock)
            yield time, _block_bytes(keys, values)

    return keys, value_rows()
