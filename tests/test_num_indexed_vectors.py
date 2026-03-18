from __future__ import annotations

from fnmatch import fnmatch
from types import SimpleNamespace

import numpy as np
import pytest

import ECLlib.io.output.unformatted_files as uf


class _DummySpec:
    def __init__(self, vectors):
        self._vectors = tuple(vectors)

    @staticmethod
    def _normalize_filter_input(values):
        if values in (None, (), [], ""):
            return ()
        if isinstance(values, str):
            return (values,)
        return tuple(values)

    def select_vectors(self, keys=()):
        if not keys:
            return self._vectors
        return tuple(
            vec for vec in self._vectors if any(fnmatch(vec.key, pattern) for pattern in keys)
        )


def _vec(index, key, name, num):
    return SimpleNamespace(index=index, key=key, name=name, num=num)


def _make_unsmry(monkeypatch, vectors, records, *, init_exists=True, ijk_map=None):
    ijk_map = ijk_map or {}

    class DummyINIT:
        def __init__(self, _path):
            pass

        def is_file(self):
            return init_exists

        def cell_ijk(self, *nums):
            return np.asarray([ijk_map[num] for num in nums], dtype=int)

    monkeypatch.setattr(uf, "INIT_file", DummyINIT)

    unsmry = uf.UNSMRY_file.__new__(uf.UNSMRY_file)
    unsmry.path = "CASE.UNSMRY"
    unsmry.spec = _DummySpec(vectors)
    unsmry.block_calls = []

    def blockdata(*args, singleton=False, only_new=False, **kwargs):
        unsmry.block_calls.append(
            {
                "args": args,
                "singleton": singleton,
                "only_new": only_new,
                "kwargs": kwargs,
            }
        )
        for rec in records:
            yield rec

    unsmry.blockdata = blockdata
    return unsmry


def test_num_indexed_vectors_single_key_grouped(monkeypatch):
    vectors = (
        _vec(3, "CWVFR", "INJ", 11),
        _vec(4, "CWVFR", "INJ", 12),
        _vec(5, "CWVFR", "PROD", 21),
    )
    ijk_map = {11: (0, 0, 0), 12: (0, 1, 0), 21: (1, 1, 1)}
    records = (
        (
            np.asarray([1.25]),
            np.asarray([-1.0, -2.0, 3.0]),
        ),
    )
    unsmry = _make_unsmry(monkeypatch, vectors, records, ijk_map=ijk_map)

    (day, key_groups), = tuple(unsmry.num_indexed_vectors(keys=("CWVFR",)))

    assert day == pytest.approx(1.25)
    assert len(key_groups) == 1
    key_group = key_groups[0]
    assert isinstance(key_group, uf.KeyIndexedValues)
    assert key_group.key == "CWVFR"
    assert [g.name for g in key_group.groups] == ["INJ", "PROD"]

    inj = key_group.groups[0]
    prod = key_group.groups[1]
    assert isinstance(inj, uf.NameIndexedValues)
    assert tuple(inj.pos[0]) == (0, 0)
    assert tuple(inj.pos[1]) == (0, 1)
    assert tuple(inj.pos[2]) == (0, 0)
    assert tuple(inj.values) == pytest.approx((-1.0, -2.0))
    assert tuple(prod.pos[0]) == (1,)
    assert tuple(prod.pos[1]) == (1,)
    assert tuple(prod.pos[2]) == (1,)
    assert tuple(prod.values) == pytest.approx((3.0,))


def test_num_indexed_vectors_multi_key_order(monkeypatch):
    vectors = (
        _vec(2, "CWBHP", "W1", 100),
        _vec(3, "CWBHP", "W2", 101),
        _vec(10, "CWVFR", "W1", 100),
        _vec(11, "CWVFR", "W2", 101),
    )
    ijk_map = {100: (0, 0, 0), 101: (1, 0, 0)}
    records = (
        (
            np.asarray([2.0]),
            np.asarray([10.0, 20.0]), # CWBHP chunk [2:4] (monotonic read order)
            np.asarray([1.0, 2.0]),   # CWVFR chunk [10:12]
        ),
    )
    unsmry = _make_unsmry(monkeypatch, vectors, records, ijk_map=ijk_map)

    (day, key_groups), = tuple(unsmry.num_indexed_vectors(keys=("CWVFR", "CWBHP")))

    assert day == pytest.approx(2.0)
    assert [g.key for g in key_groups] == ["CWVFR", "CWBHP"]
    assert [g.name for g in key_groups[0].groups] == ["W1", "W2"]
    assert [g.name for g in key_groups[1].groups] == ["W1", "W2"]
    assert tuple(key_groups[0].groups[0].values) == pytest.approx((1.0,))
    assert tuple(key_groups[1].groups[0].values) == pytest.approx((10.0,))
    assert unsmry.block_calls
    assert unsmry.block_calls[0]["args"] == ("PARAMS", 0, "PARAMS", 2, 4, "PARAMS", 10, 12)


def test_num_indexed_vectors_keys_empty_returns_all_num_positive(monkeypatch):
    vectors = (
        _vec(0, "FOPT", "FIELD", 0),
        _vec(1, "CWVFR", "W1", 7),
        _vec(2, "CWBHP", "W1", 8),
    )
    ijk_map = {7: (0, 0, 0), 8: (1, 1, 1)}
    records = (
        (
            np.asarray([3.0]),
            np.asarray([5.0, 6.0]),
        ),
    )
    unsmry = _make_unsmry(monkeypatch, vectors, records, ijk_map=ijk_map)

    (day, key_groups), = tuple(unsmry.num_indexed_vectors(keys=()))

    assert day == pytest.approx(3.0)
    assert [group.key for group in key_groups] == ["CWVFR", "CWBHP"]


def test_num_indexed_vectors_start_stop_step_only_new(monkeypatch):
    vectors = (_vec(1, "CWVFR", "W1", 10),)
    ijk_map = {10: (0, 0, 0)}
    records = (
        (np.asarray([1.0]), np.asarray([10.0])),
        (np.asarray([2.0]), np.asarray([20.0])),
        (np.asarray([3.0]), np.asarray([30.0])),
        (np.asarray([4.0]), np.asarray([40.0])),
    )
    unsmry = _make_unsmry(monkeypatch, vectors, records, ijk_map=ijk_map)

    out = tuple(unsmry.num_indexed_vectors(keys=("CWVFR",), start=1, stop=4, step=2, only_new=True))

    assert [day for day, _ in out] == pytest.approx([2.0, 4.0])
    assert unsmry.block_calls
    assert unsmry.block_calls[0]["only_new"] is True


def test_num_indexed_vectors_no_match_raises(monkeypatch):
    vectors = (_vec(1, "WBHP", "W1", 10),)
    unsmry = _make_unsmry(monkeypatch, vectors, records=(), ijk_map={10: (0, 0, 0)})

    with pytest.raises(ValueError, match="No summary vectors matched"):
        next(unsmry.num_indexed_vectors(keys=("CWVFR",)))


def test_num_indexed_vectors_invalid_num_raises(monkeypatch):
    vectors = (_vec(1, "CWVFR", "W1", 0),)
    unsmry = _make_unsmry(monkeypatch, vectors, records=())

    with pytest.raises(ValueError, match="NUM > 0"):
        next(unsmry.num_indexed_vectors(keys=("CWVFR",)))


def test_num_indexed_vectors_missing_init_raises(monkeypatch):
    vectors = (_vec(1, "CWVFR", "W1", 10),)
    unsmry = _make_unsmry(
        monkeypatch,
        vectors,
        records=((np.asarray([1.0]), np.asarray([1.0])),),
        init_exists=False,
        ijk_map={10: (0, 0, 0)},
    )

    with pytest.raises(FileNotFoundError, match="INIT"):
        next(unsmry.num_indexed_vectors(keys=("CWVFR",)))
