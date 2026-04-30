from __future__ import annotations

from itertools import product, repeat
from math import prod
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ECLlib import EGRID_file
from ECLlib.utils import batched, flatten


ROOT = Path(__file__).resolve().parents[1]
OPM_EGRID = Path("tests/Model_From_IXF_OPM/MODEL_FROM_IXF_OPM.EGRID")
IX_EGRID = Path("tests/Model_From_IXF_IX/Model_From_IXF.EGRID")
EKOFISK_EGRID = Path("tests/EKOFISK_HAUKAAS_2014/EKOFISK_HAUKAAS_2014.EGRID")


#---------------------------------------------------------------------------------------------------
def _legacy_pyvista_grid(egrid, i=None, j=None, k=None, scale=(1, 1, 1)):
#---------------------------------------------------------------------------------------------------
    """Build the previous PyVista grid representation for comparison."""
    pyvista = pytest.importorskip("pyvista")
    nijk = egrid.nijk()
    i = i or (0, nijk[0])
    j = j or (0, nijk[1])
    k = k or (0, nijk[2])
    dim = [b-a for a, b in (i, j, k)]
    ijk = product(range(*i), range(*j), range(*k))
    corners = list(egrid.cell_corners(ijk))
    points = np.asarray(corners)[:, [1, 0, 2, 3, 5, 4, 6, 7], :] * np.asarray(scale)
    ncells = prod(dim)
    cells = np.asarray(
        list(flatten((a, *b) for a, b in zip(repeat(8), batched(range(ncells * 8), 8))))
    )
    cell_type = pyvista.CellType.HEXAHEDRON * np.ones(ncells, dtype=int)
    return pyvista.UnstructuredGrid(cells, cell_type, points.reshape(-1, 3)), dim


#---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("path", (OPM_EGRID, IX_EGRID))
def test_egrid_grid_returns_corner_array(path):
#---------------------------------------------------------------------------------------------------
    """Return a NumPy corner-coordinate array for bundled EGRID fixtures."""
    grid = EGRID_file(path).grid()

    assert grid.shape == (25, 25, 5, 8, 3)


#---------------------------------------------------------------------------------------------------
def test_egrid_grid_supports_subsets():
#---------------------------------------------------------------------------------------------------
    """Return only the requested I/J/K ranges."""
    grid = EGRID_file(OPM_EGRID).grid(i=(1, 3), j=(2, 5), k=(0, 2))

    assert grid.shape == (2, 3, 2, 8, 3)


#---------------------------------------------------------------------------------------------------
def test_egrid_grid_applies_scale():
#---------------------------------------------------------------------------------------------------
    """Scale each coordinate axis in the returned corner array."""
    egrid = EGRID_file(OPM_EGRID)

    grid = egrid.grid(i=(0, 1), j=(0, 1), k=(0, 1))
    scaled = egrid.grid(i=(0, 1), j=(0, 1), k=(0, 1), scale=(2, 3, 4))

    np.testing.assert_allclose(scaled, grid * np.asarray((2, 3, 4)))


#---------------------------------------------------------------------------------------------------
def test_egrid_cells_are_corner_means():
#---------------------------------------------------------------------------------------------------
    """Compute cell centers directly from the eight cell corners."""
    kwargs = {"i": (0, 2), "j": (0, 3), "k": (0, 1)}
    egrid = EGRID_file(OPM_EGRID)

    grid = egrid.grid(**kwargs)
    cells = egrid.cells(**kwargs)

    assert cells.shape == (2, 3, 1, 3)
    np.testing.assert_allclose(cells, grid.mean(axis=-2))


#---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("path", "kwargs"),
    (
        (OPM_EGRID, {}),
        (IX_EGRID, {}),
        (EKOFISK_EGRID, {"i": (0, 2), "j": (0, 3), "k": (0, 2)}),
    ),
)
def test_egrid_grid_matches_legacy_pyvista_geometry(path, kwargs):
#---------------------------------------------------------------------------------------------------
    """Match the previous PyVista point geometry when converted to VTK point order."""
    egrid = EGRID_file(path)

    grid = egrid.grid(**kwargs)
    legacy, dim = _legacy_pyvista_grid(egrid, **kwargs)
    legacy_points = legacy.points.reshape((*dim, 8, 3))
    grid_vtk_order = grid[..., [1, 0, 2, 3, 5, 4, 6, 7], :]

    np.testing.assert_allclose(grid_vtk_order, legacy_points)
    np.testing.assert_allclose(grid.mean(axis=-2).reshape(-1, 3), legacy.cell_centers().points)


#---------------------------------------------------------------------------------------------------
def test_egrid_unstructured_grid_args_return_vtk_arrays():
#---------------------------------------------------------------------------------------------------
    """Return VTK-style connectivity, cell types, and reordered points."""
    kwargs = {"i": (1, 3), "j": (2, 5), "k": (0, 2), "scale": (2, 3, 4)}
    egrid = EGRID_file(OPM_EGRID)

    cells, cell_type, points = egrid.unstructured_grid_args(**kwargs)
    grid = egrid.grid(**kwargs)
    ncells = np.prod(grid.shape[:3])

    assert cells.shape == (ncells * 9,)
    assert cell_type.shape == (ncells,)
    assert points.shape == (ncells * 8, 3)
    assert np.all(cells.reshape(-1, 9)[:, 0] == 8)
    assert np.all(cell_type == 12)
    np.testing.assert_allclose(
        points.reshape((*grid.shape[:3], 8, 3)),
        grid[..., [1, 0, 2, 3, 5, 4, 6, 7], :],
    )


#---------------------------------------------------------------------------------------------------
def test_egrid_unstructured_grid_args_match_legacy_pyvista_geometry():
#---------------------------------------------------------------------------------------------------
    """Return VTK-compatible arrays without callers doing the point reorder manually."""
    pyvista = pytest.importorskip("pyvista")
    kwargs = {"i": (1, 3), "j": (2, 5), "k": (0, 2), "scale": (2, 3, 4)}
    egrid = EGRID_file(OPM_EGRID)

    grid = pyvista.UnstructuredGrid(*egrid.unstructured_grid_args(**kwargs))
    legacy, _ = _legacy_pyvista_grid(egrid, **kwargs)

    np.testing.assert_allclose(grid.points, legacy.points)
    np.testing.assert_allclose(grid.cell_centers().points, legacy.cell_centers().points)


#---------------------------------------------------------------------------------------------------
def test_ecllib_import_does_not_require_pyvista():
#---------------------------------------------------------------------------------------------------
    """Import the public package when PyVista imports are unavailable."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), env.get("PYTHONPATH", "")))
    code = """
import builtins

real_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "pyvista" or name.startswith("pyvista."):
        raise ModuleNotFoundError("No module named 'pyvista'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
import ECLlib
print(ECLlib.__version__)
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.strip()
