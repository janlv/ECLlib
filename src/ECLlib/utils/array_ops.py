"""Array and geometry utility functions."""
from __future__ import annotations

from itertools import chain
from typing import Iterable, Iterator, Sequence

from numpy import (
    append as npappend,
    arange,
    array,
    concatenate,
    diff as npdiff,
    int32,
    meshgrid,
    roll as nproll,
    stack,
    trapz,
    where,
)


def neighbour_connections(dim: Sequence[int]) -> array:
    """Generate connection indices for the six block faces in a 3D grid."""
    ind = index_array(dim)
    kwargs = {"as_scalar": True, "wrapped": -1}
    pos_conn = roll_xyz(ind, -1, **kwargs).swapaxes(-2, -1)
    neg_conn = roll_xyz(ind, 1, **kwargs).swapaxes(-2, -1)
    return concatenate((pos_conn, neg_conn), axis=-2)


def roll_xyz(src: array, shift: int = 1, as_scalar: bool = False, wrapped: int = 0) -> array:
    """Roll the values of a source array along all axes."""
    end = 0 if shift > 0 else -1
    roll_list = []
    arr = src
    for axis in range(3):
        ind = [slice(None)] * 3
        ind[axis] = end
        if src.ndim > 3 and not as_scalar:
            arr = src[..., axis]
        rolled = nproll(arr, shift, axis=axis)
        rolled[tuple(ind)] = wrapped
        roll_list.append(rolled)
    return stack(roll_list, axis=-1)


def index_array(shape: Sequence[int]) -> array:
    """Return an index array for the given shape."""
    i, j, k = meshgrid(
        arange(shape[0], dtype=int32),
        arange(shape[1], dtype=int32),
        arange(shape[2], dtype=int32),
        indexing="ij",
    )
    return stack((i, j, k), axis=-1)


def run_length_encode(data: array) -> list:
    """Run-length encode the supplied 1D array."""
    mask = concatenate(([True], data[:-1] != data[1:]))
    counts = npdiff(where(npappend(mask, True))[0])
    from .iterables import flatten  # Imported lazily to avoid circular import.

    return list(flatten(zip(counts, data[mask])))


def any_cell_in_box(cells: Iterable[Sequence[int]], box: Sequence[Sequence[int]]) -> bool:
    """Return ``True`` if any of ``cells`` is inside ``box``."""
    return any(all(box[n][0] <= cell[n] < box[n][1] for n in range(3)) for cell in cells)


def bounding_box(pos: Iterable[Sequence[int]]) -> list[tuple[int, int]]:
    """Return the bounding box of the supplied positions."""
    return [(p[0], p[-1]) for p in map(sorted, zip(*pos))]


def cumtrapz(y: array, x: array, *args, **kwargs) -> array:
    """A NumPy based cumulative trapezoidal integration."""
    return array([trapz(y[:i], x[:i], *args, **kwargs) for i in range(1, len(x) + 1)])


def pad_zero(lists: Iterable[Sequence[str]]) -> list[list[str]]:
    """Pad nested sequences with zeros to equal length."""
    items = [list(a) for a in lists]
    length = max(len(a) for a in items)
    return [a + ["0"] * (length - len(a)) for a in items]
__all__ = [
    "any_cell_in_box",
    "bounding_box",
    "cumtrapz",
    "index_array",
    "neighbour_connections",
    "pad_zero",
    "roll_xyz",
    "run_length_encode",
]
