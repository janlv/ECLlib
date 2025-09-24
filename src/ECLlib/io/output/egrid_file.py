"""EGRID output file handler"""

from itertools import product, repeat
from math import hypot, prod

from numpy import array as nparray, ones, zeros
from pyvista import CellType, UnstructuredGrid

from ..unformatted.base import unfmt_file
from ...utils import batched, flatten
from .unformatted_files import INIT_file

__all__ = ["EGRID_file"]

#==================================================================================================
class EGRID_file(unfmt_file):                                                          # EGRID_file
#==================================================================================================
    """Interface for reading Eclipse EGRID files."""
    start = 'FILEHEAD'
    end = 'ENDGRID'
    #----------------------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                                # EGRID_file
    #----------------------------------------------------------------------------------------------
        """Initialize the EGRID_file."""
        super().__init__(file, suffix='.EGRID', **kwargs)
        self.var_pos = {'nx': ('GRIDHEAD', 1),
                        'ny': ('GRIDHEAD', 2),
                        'nz': ('GRIDHEAD', 3),
                        'unit': ('MAPUNITS', 0)}
        self._nijk = None
        self._coord_zcorn = None

    #----------------------------------------------------------------------------------------------
    def length(self):                                                                  # EGRID_file
    #----------------------------------------------------------------------------------------------
        """Return the grid length along each axis."""
        convert = {'METRES':1.0, 'FEET':0.3048, 'CM':0.01}
        unit = next(self.blockdata('MAPUNITS'), None)
        if unit:
            return convert[unit]
        raise SystemError(f'ERROR Missing MAPUNITS in {self}')

    #----------------------------------------------------------------------------------------------
    def axes(self):                                                                    # EGRID_file
    #----------------------------------------------------------------------------------------------
        """Return arrays describing the grid axes."""
        ax = next(self.blockdata('MAPAXES'), None)
        origin = (ax[2], ax[3])
        unit_x = (ax[4]-ax[2], ax[5]-ax[3])
        unit_y = (ax[0]-ax[2], ax[1]-ax[3])
        norm_x = 1 / hypot(*unit_x)
        norm_y = 1 / hypot(*unit_y)
        return origin, (unit_x[0]*norm_x, unit_x[1]*norm_x), (unit_y[0]*norm_y, unit_y[1]*norm_y)

    #----------------------------------------------------------------------------------------------
    def nijk(self):                                                                    # EGRID_file
    #----------------------------------------------------------------------------------------------
        """Return the grid dimensions."""
        self._nijk = self._nijk or next(self.read('nx', 'ny', 'nz'))
        return self._nijk        

    #----------------------------------------------------------------------------------------------
    def coord_zcorn(self):                                                             # EGRID_file
    #----------------------------------------------------------------------------------------------
        """Return coordinate and ZCORN arrays."""
        if self._coord_zcorn is None:
            mapped = map(nparray, next(self.blockdata('COORD', 'ZCORN')))
            self._coord_zcorn = list(mapped)
        return self._coord_zcorn

    #----------------------------------------------------------------------------------------------
    def _indices(self, ijk):                                                           # EGRID_file
    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
    def cell_corners(self, ijk_iter):                                                  # EGRID_file
    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
    def grid(self, i=None, j=None, k=None, scale=(1,1,1)):                             # EGRID_file
    #----------------------------------------------------------------------------------------------
        """Return the structured grid as arrays."""
        nijk = self.nijk()
        i = i or (0, nijk[0])
        j = j or (0, nijk[1])
        k = k or (0, nijk[2])
        dim = [b-a for a, b in (i, j, k)]
        ijk = product(range(*i), range(*j), range(*k))
        corners = list(self.cell_corners(ijk))
        # Create an unstructured VTK grid using pyvista 
        # Interchange point 1(4) and 2(5) to match HEXAHEDRON cell order
        points = nparray(corners)[:,[1,0,2,3,5,4,6,7],:] * nparray(scale)
        ncells = prod(dim)
        cells = nparray(list(flatten((a, *b) for a,b in zip(repeat(8), batched(range(ncells*8), 8)))))
        cell_type = CellType.HEXAHEDRON*ones(ncells, dtype=int)
        return UnstructuredGrid(cells, cell_type, points.reshape(-1, 3))

    #----------------------------------------------------------------------------------------------
    def cells(self, **kwargs):                                                         # EGRID_file
    #----------------------------------------------------------------------------------------------
        """Return structured grid cells."""
        points = self.grid(**kwargs).cell_centers().points
        dim = INIT_file(self.path).dim() + (3,)
        return points.reshape(dim, order='C')

