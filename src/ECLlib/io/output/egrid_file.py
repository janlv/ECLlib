"""
Eclipse/INTERSECT EGRID file

The EGRID (Extensible Grid) file is a binary unformatted output-file that defines the 
complete simulation grid geometry, including corner-point coordinates, local grid 
refinements (LGRs), inactive cells, and non-neighbor connections (NNCs). It is an 
extension of the older GRID format, designed for efficiency and flexibility, and is 
used for both structured and unstructured grid representations.
"""

from itertools import product
from math import hypot

from numpy import arange, array as nparray, concatenate, full, zeros

from ..unformatted.base import unfmt_file

__all__ = ["EGRID_file"]

VTK_HEXAHEDRON = 12

#===================================================================================================
class EGRID_file(unfmt_file):                                                          # EGRID_file
#===================================================================================================
    """Reader of Eclipse EGRID files."""

    start = 'FILEHEAD'
    end = 'ENDGRID'
    #-----------------------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                                # EGRID_file
    #-----------------------------------------------------------------------------------------------
        """Initialize the EGRID_file."""
        super().__init__(file, suffix='.EGRID', **kwargs)
        self.var_pos = {'nx': ('GRIDHEAD', 1),
                        'ny': ('GRIDHEAD', 2),
                        'nz': ('GRIDHEAD', 3),
                        'unit': ('MAPUNITS', 0)}
        self._nijk = None
        self._coord_zcorn = None

    #-----------------------------------------------------------------------------------------------
    def length(self):                                                                  # EGRID_file
    #-----------------------------------------------------------------------------------------------
        """Return the grid length along each axis."""
        convert = {'METRES':1.0, 'FEET':0.3048, 'CM':0.01}
        unit = next(self.blockdata('MAPUNITS'), None)
        if unit:
            return convert[unit]
        raise SystemError(f'ERROR Missing MAPUNITS in {self}')

    #-----------------------------------------------------------------------------------------------
    def axes(self):                                                                    # EGRID_file
    #-----------------------------------------------------------------------------------------------
        """Return arrays describing the grid axes."""
        ax = next(self.blockdata('MAPAXES'), None)
        origin = (ax[2], ax[3])
        unit_x = (ax[4]-ax[2], ax[5]-ax[3])
        unit_y = (ax[0]-ax[2], ax[1]-ax[3])
        norm_x = 1 / hypot(*unit_x)
        norm_y = 1 / hypot(*unit_y)
        return origin, (unit_x[0]*norm_x, unit_x[1]*norm_x), (unit_y[0]*norm_y, unit_y[1]*norm_y)

    #-----------------------------------------------------------------------------------------------
    def nijk(self):                                                                    # EGRID_file
    #-----------------------------------------------------------------------------------------------
        """Return the grid dimensions."""
        self._nijk = self._nijk or next(self.read('nx', 'ny', 'nz'))
        return self._nijk        

    #-----------------------------------------------------------------------------------------------
    def coord_zcorn(self):                                                             # EGRID_file
    #-----------------------------------------------------------------------------------------------
        """Return coordinate and ZCORN arrays."""
        if self._coord_zcorn is None:
            mapped = map(nparray, next(self.blockdata('COORD', 'ZCORN')))
            self._coord_zcorn = list(mapped)
        return self._coord_zcorn

    #-----------------------------------------------------------------------------------------------
    def _indices(self, ijk):                                                           # EGRID_file
    #-----------------------------------------------------------------------------------------------
        """Return index arrays for the requested block."""
        nijk = self.nijk()
        # Calculate indices for grid pillars in COORD
        pind = zeros(8, dtype=int)
        pind[0] = ijk[1]*(nijk[0]+1)*6 + ijk[0]*6
        pind[1] = pind[0] + 6
        pind[2] = pind[0] + (nijk[0]+1)*6
        pind[3] = pind[2] + 6
        pind[4:] = pind[:4]
        # Get depths from ZCORN
        zind = zeros(8, dtype=int)
        zind[0] = ijk[2]*nijk[0]*nijk[1]*8 + ijk[1]*nijk[0]*4 + ijk[0]*2
        zind[1] = zind[0] + 1
        zind[2] = zind[0] + nijk[0]*2
        zind[3] = zind[2] + 1
        zind[4:] = zind[:4] + nijk[0]*nijk[1]*4           
        #              top (xyz)                   bottom (xyz)                   depths
        return nparray((pind, pind+1, pind+2)), nparray((pind+3, pind+4, pind+5)), zind

    #-----------------------------------------------------------------------------------------------
    def cell_corners(self, ijk_iter):                                                  # EGRID_file
    #-----------------------------------------------------------------------------------------------
        """Return cell corner coordinates."""
        #nijk = self.nijk()
        coord, zcorn = self.coord_zcorn()
        #coord, zcorn = map(nparray, next(self.blockdata('COORD', 'ZCORN')))
        for ijk in ijk_iter:
            top, bot, zind = self._indices(ijk)
            z = zcorn[zind]
            xt, yt, zt = coord[top]
            xb, yb, zb = coord[bot]
            if any(a==b for a,b in zip(zt, z)):
                x = xt
                y = yt
            else:
                denom = (zt - zb) * (zt - z)
                x = xt + (xb - xt) / denom
                y = yt + (yb - yt) / denom
            # Transpose to get coordinates last, i.e (8,3) instead of (3,8)
            yield nparray((x, y, z)).T

    #-----------------------------------------------------------------------------------------------
    def grid(self, i=None, j=None, k=None, scale=(1, 1, 1)):                           # EGRID_file
    #-----------------------------------------------------------------------------------------------
        """Return cell corner coordinates with shape ``(ni, nj, nk, 8, 3)``."""
        nijk = self.nijk()
        i = i or (0, nijk[0])
        j = j or (0, nijk[1])
        k = k or (0, nijk[2])
        dim = [b-a for a, b in (i, j, k)]
        ijk = product(range(*i), range(*j), range(*k))
        corners = nparray(list(self.cell_corners(ijk))).reshape((*dim, 8, 3))
        return corners * nparray(scale)

    #-----------------------------------------------------------------------------------------------
    def cells(self, **kwargs):                                                         # EGRID_file
    #-----------------------------------------------------------------------------------------------
        """Return cell center coordinates with shape ``(ni, nj, nk, 3)``."""
        return self.grid(**kwargs).mean(axis=-2)

    #-----------------------------------------------------------------------------------------------
    def unstructured_grid_args(self, i=None, j=None, k=None, scale=(1, 1, 1)):          # EGRID_file
    #-----------------------------------------------------------------------------------------------
        """
        Return ``(cells, cell_type, points)`` arrays for a VTK unstructured grid.

        The tuple is ordered for unstructured-grid constructors that accept VTK-style
        ``(cells, celltypes, points)`` arguments. ``cells`` is a flattened connectivity
        array with a leading point count of 8 for each hexahedron, ``cell_type`` contains
        VTK hexahedron ids, and ``points`` is ordered using VTK's hexahedron corner order.
        """
        points = self.grid(i=i, j=j, k=k, scale=scale)[..., [1, 0, 2, 3, 5, 4, 6, 7], :]
        points = points.reshape(-1, 8, 3)
        ncells = points.shape[0]
        connectivity = arange(ncells * 8, dtype=int).reshape(ncells, 8)
        cells = concatenate((full((ncells, 1), 8, dtype=int), connectivity), axis=1).ravel()
        cell_type = full(ncells, VTK_HEXAHEDRON, dtype=int)
        return cells, cell_type, points.reshape(-1, 3)
