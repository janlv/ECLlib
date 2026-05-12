"""Private binary helpers for Petrel/INTERSECT GSG files."""
from __future__ import annotations

from os import SEEK_END
from struct import calcsize, unpack

GSG_MAGIC = b"GSG000"
_HEADER_START = 28
_MAX_KEYWORD_SIZE = 1000


#===================================================================================================
class GSGFormatError(Exception):                                                    # GSGFormatError
#===================================================================================================
    """Raised when a GSG file cannot be decoded as the expected binary format."""


#===================================================================================================
class GSGMetadataError(GSGFormatError):                                           # GSGMetadataError
#===================================================================================================
    """Raised when GSG metadata is inconsistent with the indexed blocks."""


#===================================================================================================
class UnsupportedGSGBlock(GSGFormatError):                                     # UnsupportedGSGBlock
#===================================================================================================
    """Raised when a valid GSG block uses an unsupported encoding."""


#---------------------------------------------------------------------------------------------------
def read_exact(file_obj, size):
#---------------------------------------------------------------------------------------------------
    """Read exactly ``size`` bytes or raise a format error."""
    data = file_obj.read(size)
    if len(data) != size:
        raise GSGFormatError(f"Expected {size} bytes, got {len(data)}")
    return data


#---------------------------------------------------------------------------------------------------
def read_struct(file_obj, fmt):
#---------------------------------------------------------------------------------------------------
    """Read and unpack a little-endian struct payload."""
    if not fmt:
        return ()
    return unpack("<" + fmt, read_exact(file_obj, calcsize("<" + fmt)))


#---------------------------------------------------------------------------------------------------
def read_keyword(file_obj, fmt="", offset=None):
#---------------------------------------------------------------------------------------------------
    """Read a GSG keyword record and return key, payload, block start, and payload start."""
    if offset is not None:
        file_obj.seek(offset)
    block_start = file_obj.tell()
    key_size = read_struct(file_obj, "i")[0]
    if key_size < 0 or key_size > _MAX_KEYWORD_SIZE:
        raise GSGFormatError(f"Invalid keyword length {key_size} at byte {block_start}")
    raw_key = read_exact(file_obj, key_size)
    try:
        key = raw_key.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GSGFormatError(f"Invalid keyword bytes at byte {block_start}: {raw_key!r}") from error
    payload_start = file_obj.tell()
    payload = read_struct(file_obj, fmt)
    return key, payload, block_start, payload_start


#---------------------------------------------------------------------------------------------------
def read_header(file_obj):
#---------------------------------------------------------------------------------------------------
    """Read the optional GSG header and return creator, version, and first data offset."""
    file_obj.seek(0)
    prefix = file_obj.read(len(GSG_MAGIC))
    if not prefix.startswith(GSG_MAGIC):
        file_obj.seek(0)
        return "", "", 0

    creator, _, _, _ = read_keyword(file_obj, offset=_HEADER_START)
    version, _, _, _ = read_keyword(file_obj, "4i")
    return creator, version, file_obj.tell()


#---------------------------------------------------------------------------------------------------
def file_size(file_obj):
#---------------------------------------------------------------------------------------------------
    """Return the current file size while preserving the stream position."""
    current = file_obj.tell()
    file_obj.seek(0, SEEK_END)
    size = file_obj.tell()
    file_obj.seek(current)
    return size


#---------------------------------------------------------------------------------------------------
def read_index_records(file_obj):
#---------------------------------------------------------------------------------------------------
    """Read ordered index records as tuples preserving repeated keys."""
    size = file_size(file_obj)
    if size < 8:
        raise GSGMetadataError("GSG file is too small to contain an INDEX footer")

    file_obj.seek(size - 8)
    index_pos = read_struct(file_obj, "q")[0]
    if index_pos <= 0 or index_pos >= size:
        raise GSGMetadataError(f"Invalid INDEX position {index_pos}")

    key, data, _, _ = read_keyword(file_obj, "2i", offset=index_pos)
    if key != "INDEX":
        raise GSGMetadataError(f"Expected INDEX at byte {index_pos}, got {key!r}")

    count = data[1]
    records = []
    for _ in range(count):
        entry_key, entry_data, _, _ = read_keyword(file_obj, "iq")
        value, data_pos = entry_data
        block_start = data_pos - 4 - len(entry_key.encode("utf-8")) - 4
        records.append((entry_key, value, data_pos, block_start))

    indexed = []
    for i, (entry_key, value, data_pos, block_start) in enumerate(records):
        block_end = records[i + 1][3] if i + 1 < len(records) else index_pos
        if block_start < 0 or block_end <= block_start:
            raise GSGMetadataError(
                f"Invalid span for {entry_key!r}: start={block_start}, end={block_end}"
            )
        indexed.append((entry_key, value, data_pos, block_start, block_end))
    return tuple(indexed)
