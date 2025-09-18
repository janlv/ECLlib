"""Utilities for working with iterables and containers."""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Sequence
from itertools import chain, groupby, islice, tee, zip_longest
from operator import itemgetter, sub
from typing import Any, Callable


def slice_range(start: int = 0, stop: int | None = None, step: int | None = None) -> Iterator[tuple[int, int]]:
    """Yield ``(start, stop)`` pairs from ``range``."""
    range_with_stop = chain(range(start, stop, step), [stop])
    return (pair for pair in pairwise(range_with_stop))


def split_number(input_number: int, base: int) -> list[int]:
    """Split an integer into chunks of ``base`` with a remainder."""
    num_full_units = input_number // base
    remainder = input_number % base
    result = [base] * num_full_units
    if remainder > 0:
        result.append(remainder)
    return result


def ensure_list(values: Any) -> list:
    """Return ``values`` as a list."""
    try:
        return values[:]
    except TypeError:
        return [values]


def unique_key(key: str, keylist: list[str], symbol: str = "#") -> str:
    """Return a unique key by appending ``symbol`` and an index if needed."""
    if (count := keylist.count(key)):
        key += f"{symbol}{count}"
    return key


def list_prec(iterable: Iterable, fmt: str = ".2e") -> str:
    """Format each entry of ``iterable`` using ``fmt``."""
    return "[" + ", ".join(f"{i:{fmt}}" for i in iterable) + "]"


def batched_as_list(iterable: Iterable, n: int) -> Iterator[list]:
    """Yield lists of length ``n`` from ``iterable``."""
    for batch in batched(iterable, n):
        yield list(batch)


def missing_elements(values: Sequence[int]) -> set[int]:
    """Return the missing integer values in ``values``."""
    ordered = sorted(values)
    return set(range(ordered[0], ordered[-1] + 1)).difference(ordered)


def first_index(cond: Callable[[Any], bool], items: Sequence, fail: Any | None = None) -> Any:
    """Return the index of the first entry in ``items`` satisfying ``cond``."""
    return next((i for i, value in enumerate(items) if cond(value)), fail)


def batched_when(items: Sequence, cond: Callable[[Any], bool]) -> Iterator[Sequence]:
    """Yield slices of ``items`` ending whenever ``cond`` is true."""
    positions = chain((i for i, value in enumerate(items) if cond(value)), [len(items)])
    return (items[a:b] for a, b in pairwise(positions))


def pad(values: Sequence, length: int, fill: Any | None = None):
    """Pad ``values`` with ``fill`` until ``length``."""
    filler = (fill,) if isinstance(values, tuple) else [fill]
    return values + filler * (length - len(values))


def ordered_intersect(values: Iterable, other: Iterable) -> list:
    """Return items from ``values`` that are present in ``other`` preserving order."""
    other_set = frozenset(other)
    return [value for value in values if value in other_set]


def ordered_intersect_index(values: Sequence, other: Iterable) -> list[int]:
    """Return indices of ``values`` whose elements appear in ``other``."""
    other_set = frozenset(other)
    return [i for i, value in enumerate(values) if value in other_set]


def group_indices(indices: Sequence[int]) -> Iterator[tuple[int, int]]:
    """Group consecutive indices into ``(start, end)`` pairs."""
    jumps = (pair for pair in pairwise(indices) if -sub(*pair) > 1)
    return (tuple(pair) for pair in batched(chain([indices[0]], *jumps, [indices[-1]]), 2))


def index_limits(index: Sequence[int]) -> list[tuple[int, int] | tuple[()]]:
    """Group consecutive indices into ``(start, end)`` pairs."""
    jumps = (index[0],) + flat_list((a, b) for a, b in pairwise(index) if b - a > 1) + (index[-1],)
    return [() if a < 0 else (a, b + 1) for a, b in grouper(jumps, 2)]


def sliding_window(iterable: Iterable, n: int) -> Iterator[tuple]:
    """Yield overlapping windows of length ``n``."""
    iterator = iter(iterable)
    window = deque(islice(iterator, n - 1), maxlen=n)
    for value in iterator:
        window.append(value)
        yield tuple(window)


def pairwise(iterable: Iterable) -> Iterator[tuple]:
    """Return consecutive pairs from ``iterable``."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def triplewise(iterable: Iterable) -> Iterator[tuple]:
    """Return consecutive triplets from ``iterable``."""
    for (a, _), (b, c) in pairwise(pairwise(iterable)):
        yield a, b, c


def nth(iterable: Iterable, n: int, default: Any | None = None) -> Any:
    """Return the ``n``\ th item or ``default`` if exhausted."""
    return next(islice(iterable, n, None), default)


def take(n: int, iterable: Iterable) -> tuple:
    """Return the first ``n`` items of ``iterable`` as a tuple."""
    return tuple(islice(iterable, n))


def tail(n: int, iterable: Iterable) -> Iterator:
    """Return an iterator over the last ``n`` items."""
    return iter(deque(iterable, maxlen=n))


def prepend(value: Any, iterator: Iterable) -> Iterator:
    """Prepend ``value`` in front of ``iterator``."""
    return chain([value], iterator)


def flatten(list_of_lists: Iterable[Iterable]) -> Iterator:
    """Flatten a single nesting level from ``list_of_lists``."""
    return chain.from_iterable(list_of_lists)


def flat_list(list_or_tuple: Iterable) -> Iterable:
    """Return ``list_or_tuple`` flattened by one level."""
    try:
        flat = chain.from_iterable(list_or_tuple)
        if isinstance(list_or_tuple, list):
            return list(flat)
        return tuple(flat)
    except TypeError:
        return list_or_tuple


def flatten_all(list_of_lists: Iterable) -> Iterator:
    """Flatten arbitrarily nested iterables."""
    for entry in list_of_lists:
        if isinstance(entry, Iterable) and not isinstance(entry, (str, bytes)):
            yield from flatten_all(entry)
        else:
            yield entry


def grouper(iterable: Iterable, n: int, *, incomplete: str = "fill", fillvalue: Any | None = None):
    """Collect data into non-overlapping fixed-length chunks."""
    args = [iter(iterable)] * n
    if incomplete == "fill":
        return zip_longest(*args, fillvalue=fillvalue)
    if incomplete == "strict":
        return zip(*args, strict=True)
    if incomplete == "ignore":
        return zip(*args)
    msg = "Expected fill, strict, or ignore"
    raise ValueError(msg)


def batched(iterable: Iterable, n: int) -> Iterator[tuple]:
    """Batch data into tuples of length ``n``."""
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := tuple(islice(iterator, n)):
        yield batch


def consume(iterator: Iterable, n: int | None = None) -> None:
    """Advance ``iterator`` ``n`` steps ahead, consuming it entirely by default."""
    if n is None:
        deque(iterator, maxlen=0)
    else:
        next(islice(iterator, n, n), None)


def iter_index(iterable: Sequence, value: Any, start: int = 0) -> Iterator[int]:
    """Yield indices where ``value`` occurs in ``iterable``."""
    seq_index = iterable.index
    i = start - 1
    try:
        while True:
            i = seq_index(value, i + 1)
            yield i
    except ValueError:
        return


def groupby_sorted(iterable: Iterable, key: Callable | None = None, reverse: bool = False):
    """Sort ``iterable`` before applying :func:`itertools.groupby`."""
    key = key or itemgetter(0)
    ordered = sorted(iterable, key=key, reverse=reverse)
    for tag, groups in groupby(ordered, key):
        yield tag, [[g for g in group if g != tag] for group in groups]


def get_tuple(tuple_list_or_val: Any) -> tuple:
    """Return ``tuple_list_or_val`` as a tuple."""
    if isinstance(tuple_list_or_val, (tuple, list)):
        return tuple(tuple_list_or_val)
    return (tuple_list_or_val,)


def unique_names(names: Sequence[str], sep: str = "-") -> list[str]:
    """Append a counter to duplicate names."""
    new_names: list[str] = []
    for i, name in enumerate(names):
        total = names.count(name)
        count = names[: i + 1].count(name)
        if total > 1 and count > 1:
            new_names.append(f"{name}{sep}{count-1}")
        else:
            new_names.append(name)
    return new_names


def safeindex(values: Sequence, value: Any) -> int | None:
    """Return ``value`` index if present else ``None``."""
    return values.index(value) if value in values else None


def same_length(*lists: Sequence) -> bool:
    """Return ``True`` if all iterables have the same length."""
    iterator = iter(lists)
    first_len = len(next(iterator))
    return all(len(current) == first_len for current in iterator)


def clear_dict(adict: dict) -> None:
    """Clear nested dictionaries and lists in ``adict`` in-place."""
    for value in adict.values():
        if isinstance(value, list):
            value.clear()
        elif isinstance(value, dict):
            clear_dict(value)


__all__ = [
    "batched",
    "batched_as_list",
    "batched_when",
    "clear_dict",
    "consume",
    "ensure_list",
    "first_index",
    "flat_list",
    "flatten",
    "flatten_all",
    "get_tuple",
    "group_indices",
    "groupby_sorted",
    "grouper",
    "index_limits",
    "iter_index",
    "list_prec",
    "missing_elements",
    "nth",
    "ordered_intersect",
    "ordered_intersect_index",
    "pairwise",
    "pad",
    "prepend",
    "safeindex",
    "same_length",
    "slice_range",
    "sliding_window",
    "split_number",
    "take",
    "tail",
    "triplewise",
    "unique_key",
    "unique_names",
]
