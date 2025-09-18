"""String handling helpers."""
from __future__ import annotations

from itertools import chain
from pathlib import Path
from typing import Iterable, Iterator

from numpy import array, sum as npsum
from re import IGNORECASE, MULTILINE, sub as re_sub, compile as re_compile
from fnmatch import fnmatch

from .iterables import flat_list, pairwise


def ensure_bytestring(astring: str | bytes) -> bytes:
    """Return ``astring`` as bytes."""
    if isinstance(astring, bytes):
        return astring
    return astring.encode()


def to_letter(num: int, base: int = 26, case: str = "lower") -> str:
    """Convert ``num`` to a base-``base`` alphabetic representation."""
    if num < 1:
        return ""
    num -= 1
    quotient = num // base
    shift = {"upper": 65, "lower": 97}
    letter = chr(shift[case] + num - quotient * base)
    return to_letter(quotient, base, case) + letter


def letter_range(length: int, base: int = 26, case: str = "lower") -> Iterator[str]:
    """Yield letters in sequence using :func:`to_letter`."""
    for i in range(length):
        yield to_letter(i + 1, base, case)


def match_in_wildlist(string: str, wildlist: Iterable[str]) -> str | None:
    """Return the first pattern in ``wildlist`` that matches ``string``."""
    return next((pattern for pattern in wildlist if fnmatch(string, pattern)), None)


def expand_pattern(patterns: Iterable[str], strings: Iterable[str], invert: bool = False) -> list[str]:
    """Return expanded patterns that match ``strings``."""
    if invert:
        return [s for s in strings if not any(fnmatch(s, pat) for pat in patterns)]
    return [s for pat in patterns for s in strings if fnmatch(s, pat)]


def split_in_lines(text: str) -> Iterator[str]:
    """Yield stripped non-empty lines from ``text``."""
    return (line for t in text.split("\n") if (line := t.strip()))


def removeprefix(prefix: str, string: str) -> str:
    """Remove ``prefix`` from ``string`` if present."""
    if string.startswith(prefix):
        return string[len(prefix) :]
    return string


def string_split(value: str, length: int, strip: bool = False) -> Iterator[str]:
    """Split ``value`` into chunks of ``length``."""
    strings = (value[i : i + length] for i in range(0, len(value), length))
    if strip:
        return (s.strip() for s in strings)
    return strings


def split_by_words(string: str | bytes, words: Iterable[str]):
    """Split ``string`` into sections starting with any of ``words``."""
    regex = r"^\s*\b(" + "|".join(words) + r")\b"
    if isinstance(string, bytes):
        regex = regex.encode()
    matches_ = re_compile(regex, flags=IGNORECASE | MULTILINE).finditer(string)
    tag_pos = chain(((m.group(1), m.start()) for m in matches_), [("", len(string))])
    return ((tag, a, b) for (tag, a), (_, b) in pairwise(tag_pos))


def convert_float_or_str(words: Iterable[str]) -> Iterator[float | str]:
    """Yield floats if possible, otherwise stripped strings."""
    for word in words:
        try:
            value = float(word)
        except ValueError:
            value = str(word).strip()
        yield value


def string_in_file(string: str, file: str | Path) -> bool:
    """Return ``True`` if ``string`` exists in ``file``."""
    with open(file, "rb") as f:
        output = f.read()
    return string.encode() in output


def remove_chars(chars: Iterable[str], text: str) -> str:
    """Remove any characters in ``chars`` from ``text``."""
    for char in chars:
        if char in text:
            text = text.replace(char, "")
    return text


def remove_leading_nondigits(txt: str) -> str:
    """Remove non-digit characters at the beginning of ``txt``."""
    return re_sub(r"^[a-zA-Z-+._]*", "", txt)


def strip_zero(numbers: Iterable[float]) -> list[str]:
    """Strip trailing zeros from formatted floats."""
    return [f"{num:.3f}".rstrip("0").rstrip(".") for num in numbers]


def decode(data: bytes) -> str:
    """Decode ``data`` using UTF-8 or Latin-1."""
    for encoding in ("utf-8", "latin1"):
        try:
            return data.decode(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise SystemError("ERROR decode with utf-8/latin1 encoding failed!")


def upper_and_lower(values: Iterable[str]) -> list[str]:
    """Return both upper- and lower-case variants of ``values``."""
    return list(flat_list([[item.upper(), item.lower()] for item in values]))


def return_matching_string(str_list: Iterable[str], string: str) -> str | None:
    """Return the first string in ``str_list`` that occurs in ``string``."""
    for candidate in str_list:
        if candidate in string:
            return candidate
    return None


def get_substrings(string: str, length: int) -> list[str]:
    """Return substrings of ``string`` of size ``length``."""
    length = int(length)
    return [string[i : i + length].strip() for i in range(0, len(string), length)]


def list2str(alist: Iterable, start: str = "", end: str = "", sep: str = "", count: bool = False) -> str:
    """Format ``alist`` as a comma separated string."""
    if count:
        return ", ".join([f"{v} (n={npsum(array(alist) == v)})" for v in set(alist)])
    return f"{start}{', '.join(f'{sep}{i}{sep}' for i in alist)}{end}"


def list2text(alist: Iterable) -> str:
    """Return ``alist`` joined with commas and an ``and`` before the last item."""
    text = ", ".join(str(item) for item in alist)
    return " and".join(text.rsplit(",", 1))


def float_or_str(words: Iterable) -> list[float | str]:
    """Return a list containing floats where possible, otherwise strings."""
    if not isinstance(words, (list, tuple)):
        words = (words,)
    values: list[float | str] = []
    for word in words:
        try:
            value: float | str = float(word)
        except ValueError:
            value = str(word)
        values.append(value)
    return values


def bytes_string(size: float, digits: int = 0) -> str:
    """Return ``size`` formatted using binary unit prefixes."""
    unit = {0: "", 1: "K", 2: "M", 3: "G", 4: "T"}
    index = 0
    while size > 1024:
        index += 1
        size /= 1024
    return f"{size:.{digits}f} {unit[index]}"


__all__ = [
    "bytes_string",
    "convert_float_or_str",
    "decode",
    "ensure_bytestring",
    "expand_pattern",
    "float_or_str",
    "get_substrings",
    "letter_range",
    "list2str",
    "list2text",
    "match_in_wildlist",
    "remove_chars",
    "remove_leading_nondigits",
    "removeprefix",
    "return_matching_string",
    "split_by_words",
    "split_in_lines",
    "string_in_file",
    "string_split",
    "strip_zero",
    "to_letter",
    "upper_and_lower",
]
