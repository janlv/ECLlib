"""
Eclipse/INTERSECT unformatted output file handlers.

Unformatted Eclipse files are Fortran-style binary files written in big-endian mode.  
Each file consists of successive records, where each record (or "block") is prefixed 
and suffixed by a 4-byte integer specifying the block size in bytes. Data blocks are 
identified by 8-character keywords and contain arrays of integers, reals, doubles, 
logicals, or character strings. The data is ordered according to Fortran array 
conventions, with the leftmost index varying fastest.

File types:
- INIT: Initial conditions file containing static grid properties (porosity, permeability)
  and tabular PVT/saturation data.
- RFT: Well vector file containing depth, pressure, saturation, and flow data at well
  connections; may include PLT and multisegment data.
- UNSMRY: Unified summary file containing time-series vector data (production, injection,
  etc.) for all report steps.
- SMSPEC: Summary specification file defining which simulation vectors are written
  to the summary output.
- RSSPEC: Restart specification file indexing the vectors, offsets, and metadata
  for UNRST files.
"""

from collections import namedtuple
from datetime import datetime, timedelta
from fnmatch import fnmatch
from itertools import groupby, islice, product, repeat
from operator import attrgetter, itemgetter

from matplotlib.pyplot import figure as pl_figure
from numpy import array as nparray, empty as npempty, stack

from ..unformatted.base import unfmt_block, unfmt_file
from ...utils import cumtrapz, flatten


__all__ = [
    "INIT_file",
    "UNRST_file",
    "RFT_file",
    "UNSMRY_file",
    "SMSPEC_file",
    "RSSPEC_file",
    "NameIndexedValues",
    "KeyIndexedValues",
]

SummaryVector = namedtuple("SummaryVector", "index key name num unit measure")
NameIndexedValues = namedtuple("NameIndexedValues", "name pos values")
KeyIndexedValues = namedtuple("KeyIndexedValues", "key groups")


#==================================================================================================
class INIT_file(unfmt_file):                                                            # INIT_file
#==================================================================================================
    """
    INIT (Initialization File)
    Binary unformatted file containing initial grid properties and tabular data. 
    Includes arrays for porosity, permeability, transmissibilities, and PVT or 
    saturation function tables. Represents the static state before the first timestep.
    """

    start = 'INTEHEAD'

    #----------------------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                                 # INIT_file
    #----------------------------------------------------------------------------------------------
        """Initialize the INIT_file."""
        super().__init__(file, suffix='.INIT', **kwargs)
        self.var_pos = {'nx'        : ('INTEHEAD',  8),
                        'ny'        : ('INTEHEAD',  9),
                        'nz'        : ('INTEHEAD', 10),
                        'day'       : ('INTEHEAD', 64),
                        'month'     : ('INTEHEAD', 65),
                        'year'      : ('INTEHEAD', 66),
                        'simulator' : ('INTEHEAD', 94),
                        'hour'      : ('INTEHEAD', 206),
                        'minute'    : ('INTEHEAD', 207),
                        'second'    : ('INTEHEAD', 410),
                        }
        self._dim = None

    #----------------------------------------------------------------------------------------------
    def dim(self):                                                                      # INIT_file
    #----------------------------------------------------------------------------------------------
        """Return the grid dimensions."""
        self._dim = self._dim or next(self.read('nx', 'ny', 'nz'))
        return self._dim
    
    #----------------------------------------------------------------------------------------------
    def cell_ijk(self, *cellnum):                                                       # INIT_file
    #----------------------------------------------------------------------------------------------
        """
        Return ijk-indices of cells given a list of cell-numbers. 
        The cell numbers (Eclipse/Intersect) are 1-based and the ijk-indices are 0-based.
        """
        if not cellnum:
            return nparray([])
        dim = self.dim()
        ni, nij = dim[0], dim[0]*dim[1]
        cellnum = nparray(cellnum) - 1
        i = cellnum % ni
        j = (cellnum % nij) // ni
        k = cellnum // nij
        return stack([i, j, k], axis=-1)
        #return nparray([i, j, k]).T

    #----------------------------------------------------------------------------------------------
    def non_neigh_conn(self):                                                           # INIT_file
    #----------------------------------------------------------------------------------------------
        """
        Identifies and returns non-neighbor connections (NNC) between cells.

        This method searches for the keys 'NNC1' and 'NNC2' in the data. If these keys
        are not found, it raises a ValueError. If the keys are found, it retrieves the
        corresponding block data and returns a zip object containing the cell indices
        for the non-neighbor connections.

        Returns:
            zip: A zip object containing tuples of cell indices for non-neighbor connections.

        Raises:
            ValueError: If 'NNC1' and 'NNC2' keys are not found in the data.
        """
        keys = ('NNC1', 'NNC2')
        self.check_missing_keys(*keys)
        nncs = next(self.blockdata(*keys, singleton=True), ((),()))
        return [self.cell_ijk(*nnc).tolist() for nnc in nncs]
        #return nparray([self.cell_ijk(*nnc) for nnc in nncs])
        #return stack([self.cell_ijk(*nnc) for nnc in nncs], axis=0)

    #----------------------------------------------------------------------------------------------
    def simulator(self):                                                                # INIT_file
    #----------------------------------------------------------------------------------------------
        """Return the simulator identifier."""
        sim_codes = {100:'ecl', 300:'ecl', 500:'ecl', 700:'ix', 800:'FrontSim'}
        if sim:=next(self.read('simulator'), None):
            if sim < 0:
                return 'other simulator'
            return sim_codes[sim]

    #----------------------------------------------------------------------------------------------
    def start_date(self):                                                               # INIT_file
    #----------------------------------------------------------------------------------------------
        """Return the simulation start date."""
        keys = ('year', 'month', 'day', 'hour', 'minute', 'second')
        if data := next(self.read(*keys), None):
            kwargs = dict(zip(keys, data))
            # Unit of second is microsecond
            kwargs['second'] = int(kwargs['second']/1e6)
            return datetime(**kwargs)



from .unrst_file import UNRST_file



#==================================================================================================
class RFT_file(unfmt_file):                                                              # RFT_file
#==================================================================================================
    """
    RFT (Well/Reservoir Flow Test File)
    Binary unformatted vector file storing well-specific data at defined timesteps. 
    Contains depth, pressure, saturation, and flow rates at each well connection. 
    May include Production Logging Tool (PLT) and Multisegment Well (MSW) data such 
    as segment pressures, velocities, and phase holdups.
    """

    start = 'TIME'
    end = 'CONNXT'
    var_pos =  {'time'     : ('TIME', 0),
                'wellname' : ('WELLETC', 1),
                'welltype' : ('WELLETC', 6),
                'waterrate': ('CONWTUB', None),
                'I'        : ('CONIPOS', None),
                'J'        : ('CONJPOS', None),
                'K'        : ('CONKPOS', None)}

    #----------------------------------------------------------------------------------------------
    def __init__(self, file):                                                            # RFT_file
    #----------------------------------------------------------------------------------------------
        """Initialize the RFT_file."""
        super().__init__(file, suffix='.RFT')
        self.current_section = 0

    #----------------------------------------------------------------------------------------------
    def end_time(self):                                                                  # RFT_file
    #----------------------------------------------------------------------------------------------
        """Return the final timestamp."""
        # Return time-value from tail of file
        return next(self.read('time', tail=True), None) or 0

    #----------------------------------------------------------------------------------------------
    def time_slice(self):                                                                # RFT_file
    #----------------------------------------------------------------------------------------------
        """
        Yield time and slice for equal time sections
        """
        endpos = self._endpos
        ends = self.section_start_end_blocks(only_new=True)
        try:
            first, last = next(ends, (None, None))
            if first is None:
                return
            time = first.data()[0]
            while True:
                a,b = next(ends, (None, None))
                if a is None:
                    yield (time, (first.header.startpos, last.header.endpos))
                    return
                if (t:=a.data()[0]) > time:
                    self._endpos = a.header.startpos
                    yield (time, (first.header.startpos, self._endpos))
                    first = a
                    time = t
                last = b
        except ValueError:
            self._endpos = endpos
            yield (None, None)
            return

    #----------------------------------------------------------------------------------------------
    def sections_matching_time(self, days, acc=1e-5):                                    # RFT_file
    #----------------------------------------------------------------------------------------------
        """
        Yield the start- and end-pos of neighbouring sections matching given time
        """
        start, end = 9e9, 0
        for sec in self.section_blocks():
            time = sec[0].data()[0]
            if days-acc < time < days+acc and sec[-1].key() == self.end:
                if (pos := sec[0].header.startpos) < start:
                    start = pos
                if (pos := sec[-1].header.endpos) > end:
                    end = pos
            if time > days + acc:
                break
        return (start, end)

    #----------------------------------------------------------------------------------------------
    def wellstart(self, *wellnames):                                                     # RFT_file
    #----------------------------------------------------------------------------------------------
        """Return the start time for each well."""
        wells = list(wellnames)
        start = {well:None for well in wells}
        for time, name in self.read('time', 'wellname'):
            if not wells:
                break
            if name in wells:
                wells.remove(name)
                start[name] = time
        if missing_wells := [well for well, pos in start.items() if pos is None]:
            raise ValueError(f'Wells {missing_wells} are missing in {self}')
        return tuple(start.values())

    #----------------------------------------------------------------------------------------------
    def wellbbox(self, *wellnames, zerobase=True):                                       # RFT_file
    #----------------------------------------------------------------------------------------------
        """Return well bounding boxes."""
        wpos = self.wellpos(*wellnames, zerobase=zerobase)
        bbox = {well:None for well in wellnames}
        for well, wp in zip(wellnames, wpos):
            bbox[well] = [(p[0], p[-1]) for p in map(sorted, zip(*wp))]
        return tuple(bbox.values())

    #----------------------------------------------------------------------------------------------
    def wellpos(self, *wellnames, zerobase=True):                                        # RFT_file
    #----------------------------------------------------------------------------------------------
        """Return well positions."""
        wells = list(wellnames)
        wpos = {well:None for well in wells}
        for name, *pos in self.read('wellname', 'I', 'J', 'K'):
            if not wells:
                break
            if name in wells:
                wells.remove(name)
                if zerobase:
                    pos = [[x-1 for x in p] for p in pos]
                wpos[name] = tuple(zip(*pos))
        if missing_wells := [well for well, pos in wpos.items() if pos is None]:
            raise ValueError(f'Wells {missing_wells} are missing in {self}')
        return tuple(wpos.values())

    #----------------------------------------------------------------------------------------------
    def grid2wellname(self, dim, *wellnames):                                            # RFT_file
    #----------------------------------------------------------------------------------------------
        """Return well names indexed by grid cells."""
        poswell = {pos:[] for pos in product(*(range(d) for d in dim))}
        for well, pos in zip(wellnames, self.wellpos(*wellnames)):
            for p in pos:
                poswell[p].append(well)
        return poswell
        
    #----------------------------------------------------------------------------------------------
    def active_wells(self):                                                              # RFT_file
    #----------------------------------------------------------------------------------------------
        """Return the active wells defined in the file."""
        wells = []
        current_time = next(self.read('time'))
        for time, well in self.read('time', 'wellname'):
            if time > current_time:
                yield current_time, wells
                wells = []
            wells.append(well)
            current_time = time


#==================================================================================================
class UNSMRY_file(unfmt_file):                                                        # UNSMRY_file
#==================================================================================================
    """
    UNSMRY (Unified Summary File)
    Binary unformatted file containing time-series vector data for the entire run. 
    Each record corresponds to a report step or ministep and stores all vectors 
    defined in the associated SMSPEC file (e.g. production rates, pressures, totals).

    A report step is initiated by a SEQHDR keyword, followed by pairs of MINISTEP
    and PARAMS keywords for each ministep. Hence, one sequence might have multiple
    MINISTEP and PARAMS keywords.
    """

    start = 'MINISTEP'
    end = 'PARAMS'
    var_pos = {'days' : ('PARAMS', 0),
               'years': ('PARAMS', 1),
               'step' : ('MINISTEP', 0)}

    #----------------------------------------------------------------------------------------------
    def __init__(self, file):                                                         # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        """Initialize the UNSMRY_file."""
        super().__init__(file, suffix='.UNSMRY')
        self.spec = SMSPEC_file(file)
        self.startdate = None
        self._plots = None

    #----------------------------------------------------------------------------------------------
    def params(self, *keys):                                                          # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        """Return parameters describing the block."""
        keypos = {key:i for i, key in enumerate(next(self.spec.blockdata('KEYWORDS')))}
        ind = [keypos[key] for key in keys]
        for param in self.blockdata('PARAMS'):
            yield [param[i] for i in ind]

    #----------------------------------------------------------------------------------------------
    def steptype(self):                                                               # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        """Return the summary step type."""
        codes = {
            1: ('Init', 'The initial time step for a simulation run'),
            2: ('TTE' , 'Time step selected on time truncation error criteria'),
            3: ('MDF' , 'The maximum decrease factor between time steps'),
            4: ('MIF' , 'The maximum increase factor between time steps'),
            5: ('Min' , 'The minimum allowed time step'),
            6: ('Max' , 'The maximum allowed time step'),
            7: ('Rep' , 'Time step required to synchronize with a report date'),
            8: ('HRep', 'Time step required to get half way to a report date'),
            9: ('Chop', 'Time step chopped due to previous convergence problems'),
            16: ('SCT', 'Time step selected on solution change criteria'),
            31: ('CFL', 'Time step selected to maintain CFL stability'),
            32: ('Mn' , 'The time step specified was selected but was limited by minimum time step'),
            37: ('Mx' , 'The time step specified was selected but was limited by maximum time step'),
            35: ('SEQ', 'Time step determined by the convergence of the sequential fully implicit solver'),
            0:  ('FM' , 'Time step selected by Field Management')
        }
        for stype in self.params('STEPTYPE'):
            yield codes[stype[0]]

    #----------------------------------------------------------------------------------------------
    @staticmethod
    def _contiguous_limits(indices):                                                   # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        """Return contiguous [start, stop) slices from index positions in traversal order."""
        if not indices:
            return ()
        limits = []
        first = indices[0]
        last = first + 1
        for index in indices[1:]:
            if index == last:
                last += 1
            else:
                limits.append((first, last))
                first = index
                last = index + 1
        limits.append((first, last))
        return tuple(limits)

    #----------------------------------------------------------------------------------------------
    def _select_num_vectors(self, keys):                                               # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        """Return vectors with valid NUM mapping in deterministic logical order."""
        key_patterns = self.spec._normalize_filter_input(keys)
        if key_patterns:
            vectors = self.spec.select_vectors(keys=key_patterns)
            if not vectors:
                raise ValueError(f'No summary vectors matched keys={key_patterns}')
            if invalid := sorted({vector.key for vector in vectors if vector.num <= 0}):
                raise ValueError(
                    f'num_indexed_vectors only supports vectors with NUM > 0. '
                    f'Invalid keys: {invalid}'
                )
            key_order = []
            for pattern in key_patterns:
                for vector in vectors:
                    if fnmatch(vector.key, pattern) and vector.key not in key_order:
                        key_order.append(vector.key)
            return tuple(vector for key in key_order for vector in vectors if vector.key == key)
        vectors = tuple(vector for vector in self.spec.select_vectors() if vector.num > 0)
        if not vectors:
            raise ValueError('No NUM-indexed summary vectors found (NUM > 0)')
        return vectors

    #----------------------------------------------------------------------------------------------
    def _prepare_num_plan(self, vectors, only_new=False, **kwargs):                   # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        """Build monotonic reader and grouped metadata for NUM-indexed vectors."""
        read_order = nparray([vector.index for vector in vectors], dtype=int).argsort(kind='stable')
        read_vectors = tuple(vectors[int(i)] for i in read_order)
        read_pos_by_logical = npempty(len(vectors), dtype=int)
        for read_pos, logical_pos in enumerate(read_order):
            read_pos_by_logical[int(logical_pos)] = read_pos

        init = INIT_file(self.path)
        if not init.is_file():
            raise FileNotFoundError(f'Unable to map NUM to ijk because {init} is missing')
        nums = tuple(sorted({vector.num for vector in vectors}))
        ijk = init.cell_ijk(*nums)
        num_to_ijk = {num: tuple(int(i) for i in xyz) for num, xyz in zip(nums, ijk)}

        grouped = {}
        for logical_pos, vector in enumerate(vectors):
            by_name = grouped.setdefault(vector.key, {})
            row = by_name.setdefault(vector.name, {'read_pos': [], 'ipos': [], 'jpos': [], 'kpos': []})
            row['read_pos'].append(int(read_pos_by_logical[logical_pos]))
            ipos, jpos, kpos = num_to_ijk[vector.num]
            row['ipos'].append(ipos)
            row['jpos'].append(jpos)
            row['kpos'].append(kpos)

        key_name_groups = []
        for key, by_name in grouped.items():
            name_groups = []
            for name, row in by_name.items():
                read_pos = nparray(row['read_pos'], dtype=int)
                pos = (
                    nparray(row['ipos'], dtype=int),
                    nparray(row['jpos'], dtype=int),
                    nparray(row['kpos'], dtype=int),
                )
                name_groups.append((name, read_pos, pos))
            key_name_groups.append((key, tuple(name_groups)))
        key_name_groups = tuple(key_name_groups)

        keylim = ['PARAMS', 0]
        for start_i, stop_i in self._contiguous_limits(tuple(vector.index for vector in read_vectors)):
            keylim.extend(('PARAMS', start_i, stop_i))
        reader = self.blockdata(*keylim, singleton=True, only_new=only_new, **kwargs)
        return reader, key_name_groups, len(read_vectors)

    #----------------------------------------------------------------------------------------------
    @staticmethod
    def _assemble_chunks(chunks, size):                                                # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        """Assemble potentially split contiguous slices into one value array."""
        if len(chunks) == 1:
            return chunks[0]
        values = npempty(size, dtype=chunks[0].dtype)
        pos = 0
        for chunk in chunks:
            end = pos + chunk.size
            values[pos:end] = chunk
            pos = end
        return values

    #----------------------------------------------------------------------------------------------
    def num_indexed_vectors(self, keys=(), only_new=False, start=0, stop=None, step=1,
                            **kwargs):                                                  # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        """
        Yield `(day, key_groups)` for NUM-indexed vectors mapped to zero-based `(i,j,k)`.

        `key_groups` is a tuple of `KeyIndexedValues`, each containing grouped
        `NameIndexedValues(name, pos, values)` in deterministic order:
        requested key-pattern order, then SMSPEC order within each key.

        Examples:
            `keys=('CWVFR',)` returns one key group for `CWVFR`.
            `keys=('CW*',)` returns multiple key groups, ordered by requested pattern.
        """
        vectors = self._select_num_vectors(keys)
        reader, key_name_groups, nvalues = self._prepare_num_plan(
            vectors, only_new=only_new, **kwargs
        )
        reader = islice(reader, start, stop, step)
        for data in reader:
            day_data, *chunks = data
            day = float(day_data[0]) if day_data.size else 0.0
            values = self._assemble_chunks(chunks, nvalues)
            key_groups = tuple(
                KeyIndexedValues(
                    key,
                    tuple(
                        NameIndexedValues(name, pos, values[read_pos])
                        for name, read_pos, pos in name_groups
                    ),
                )
                for key, name_groups in key_name_groups
            )
            yield day, key_groups

    #----------------------------------------------------------------------------------------------
    def welldata(self, keys=(), wells=(), only_new=False, as_array=False,
                 named=False, start=0, stop=None, step=1, **kwargs):                  # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        """
        named = False   : Returns days, dates and a tuple of key, well, value for each key-well combination
        named = True    : Returns days, dates, and all keys as their names
        only_new = True : Returns only previously un-read data
        as_array = True : Converts key-well data into a numpy array
        """
        if self.is_file() and self.spec.welldata(keys=keys, wells=wells, named=named):
            self.var_pos['welldata'] = ('PARAMS', *self.spec.well_pos())
            reader = self.read('days', 'welldata', only_new=only_new, singleton=True, **kwargs)
            try:
                days, data = zip(*islice(reader, start, stop, step))
                days = tuple(float(day) for day in flatten(days))
            except ValueError:
                days, data = (), ()
            if not data:
                return ()
            # Add dates
            self.startdate = self.startdate or self.spec.startdate()
            dates = (self.startdate + timedelta(days=day) for day in days)
            # Process keys and wells
            kwd = zip(self.spec.keys, self.spec.wells, zip(*data))
            if as_array:
                kwd = ((k,w,nparray(d)) for k,w,d in kwd)
            if named:
                wells = self.wells
                units = {k:{'unit':u, 'measure':m} for k,u,m in zip(*attrgetter('keys', 'units', 'measures')(self.spec))}
                grouped = groupby(kwd, key=itemgetter(0))
                Values = namedtuple('Values', wells + ('unit', 'measure'), defaults=len(wells)*((),)+2*(None,))
                values = {k:Values(**dict(g[1:] for g in gr), **units[k]) for k,gr in grouped}
                Welldata = namedtuple('Welldata', ('days', 'dates') + self.keys)
                return Welldata(days=days, dates=tuple(dates), **values)
            Values = namedtuple('Values', 'key well data')
            values = (Values(k, w, d) for k,w,d in kwd)
            return namedtuple('Welldata', 'days dates values')(days, tuple(dates), tuple(values))
        return ()

    #----------------------------------------------------------------------------------------------
    def plot(self, keys=(), wells=(), ncols=1, date=True, fignr=1,
             args=None, **kwargs):                                                    # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        """Plot well curves using matplotlib."""
        if data := self.welldata(keys=keys, wells=wells, **kwargs):
            if date:
                xlabel = 'Dates'
                time = data.dates
            else:
                xlabel = 'Days'
                time = data.days
            _keys = [k for k in keys if k in self.keys] if keys else self.keys
            if not self._plots:
                # Create figure and axes
                nrows = -(-len(_keys)//ncols) # -(-a//b) is equivalent of ceil
                fig = pl_figure(fignr, clear=True, figsize=(8*ncols,4*nrows))
                axes = {key:fig.add_subplot(nrows, ncols, i+1) for i,key in enumerate(_keys)}
                fig.subplots_adjust(hspace=0.5, wspace=0.25)
                units = self.key_units()
                for key, ax in axes.items():
                    ax.set_title(key)
                    ax.set_xlabel(xlabel)
                    ylabel = getattr(units, key)
                    ax.set_ylabel(ylabel.measure + (f' [{ylabel.unit}]' if ylabel.unit else ''))
                # Update plot args
                default = {'marker':'o', 'ms':2, 'linestyle':'None'}
                if args is None:
                    args = {}
                args.update(**{k:args.get(k) or v for k,v in default.items()})
                lines = {}
                welldata = {'time': []}
                self._plots = (fig, axes, lines, args, welldata)
            # Make plots
            fig, axes, lines, args, welldata = self._plots
            welldata['time'].extend(time)
            for val in data.values:
                key_well = (val.key, val.well)
                if data := welldata.get(key_well):
                    # Existing well, update data and line
                    data.extend(val.data)
                    lines[key_well].set_data(welldata['time'][-len(data):], data)
                else:
                    # New well, create data and line
                    data = welldata[key_well] = list(val.data)
                    lines[key_well], = axes[val.key].plot(welldata['time'][-len(data):], data, label=val.well, **args)
            for ax in axes.values():
                #ax.legend(loc='upper left', fontsize='smaller', ncols=-(-len(ax.lines)//7)) # max 7 labels each column
                ax.legend(fontsize='smaller', ncols=-(-len(ax.lines)//7)) # max 7 labels each column
                ax.relim()
                ax.autoscale_view()
            fig.canvas.draw_idle()# draw()

    #----------------------------------------------------------------------------------------------
    def key_units(self):                                                              # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        """Return unit information for each summary key."""
        Var = namedtuple('Var','unit measure')
        # If 'measures' is None, just use empty string
        kum = zip(*attrgetter('keys', 'units')(self.spec), self.spec.measures or repeat(''))
        var = {k:Var(u, m.split(':')[-1].replace('_',' ')) for k,u,m in set(kum)}
        return namedtuple('Keys', self.keys)(**var)

    #----------------------------------------------------------------------------------------------
    def __getattr__(self, item):                                                      # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        """Delegate attribute lookups."""
        try:
            # Look for attribute in File-class first
            return super().__getattr__(item)
        except AttributeError:
            return tuple(set(getattr(self.spec, item)))

    #----------------------------------------------------------------------------------------------
    def energy(self, *wells):                                                         # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        """Return well energy totals."""
        data = self.welldata(keys=('WBHP','WTHP','WWIR'), wells=wells, as_array=True, named=True)
        wells = wells or self.wells
        # Power = (WBHP - WTHP) * WWIR
        # Energy is time-integral of power (use trapezoidal rule: cumtrapz)
        Energy = namedtuple('Energy', ('unit',) + wells)
        if data.WBHP.unit == 'BARSA':
            # 1 bar = 1e5 Joule/m3
            BTI = ((getattr(kd, well) for kd in (data.WBHP, data.WTHP, data.WWIR)) for well in wells)
            energy = Energy('Joule', *(1e-5*cumtrapz((BP-TP)*IR, data.days) for BP,TP,IR in BTI))
            return namedtuple('Data', 'days dates energy')(data.days[1:], data.dates[1:], energy)
        raise SystemError(f'ERROR Energy calculation only for metric data, pressure unit is: {data.WBHP.unit}')



#==================================================================================================
class SMSPEC_file(unfmt_file):                                                        # SMSPEC_file
#==================================================================================================
    """
    Reader for Eclipse SMSPEC files.
    
    SMSPEC (Summary Specification File)
    Binary unformatted file that defines which vectors (e.g. FOPR, WCT, BHP) are 
    written to the summary output. Contains metadata such as units, vector names, 
    grid dimensions, and run start time; required to interpret UNSMRY data.
    """

    start = 'INTEHEAD'
    Data = namedtuple('Data', 'keys wells measures units nums', defaults=5*(None,))

    #----------------------------------------------------------------------------------------------
    def __init__(self, file):                                                         # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        """Initialize the SMSPEC_file."""
        super().__init__(file, suffix='.SMSPEC')
        self._inkeys = ()
        self._ind = ()
        self.measures = ()
        self.wells = ()
        self.data = self.Data()
        self._vector_meta = None

    #----------------------------------------------------------------------------------------------
    def _load_vector_metadata(self):                                                   # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        """Read and cache vector metadata from SMSPEC."""
        if self._vector_meta is not None:
            return self._vector_meta
        self._vector_meta = ()
        self.data = self.Data()
        if not self.is_file():
            return self._vector_meta
        # Do not use mmap here because SMSPEC may be truncated while writing.
        blockdata = next(
            self.blockdata('KEYWORDS', '*NAMES', 'NUMS', 'MEASRMNT', 'UNITS', use_mmap=False), None
        )
        if blockdata is None:
            return self._vector_meta
        keys_raw, names_raw, nums_raw, measures_raw, units_raw = blockdata
        keys = tuple(keys_raw.tolist())
        names = tuple(names_raw.tolist())
        nums = tuple(int(n) for n in nums_raw.tolist())
        units = tuple(units_raw.tolist())
        measures = ()
        if len(keys):
            n_measures = len(measures_raw)
            width = n_measures // len(keys) if n_measures else 0
            if width:
                measure_chars = measures_raw.tolist()
                measures = tuple(
                    ''.join(measure_chars[i*width:(i + 1)*width]) for i in range(len(keys))
                )
            else:
                measures = ('',) * len(keys)
        self.data = self.Data(keys, names, measures, units, nums)
        self._vector_meta = tuple(
            SummaryVector(i, key, name, num, unit, measure)
            for i, (key, name, num, unit, measure) in enumerate(
                zip(keys, names, nums, units, measures)
            )
        )
        return self._vector_meta

    #----------------------------------------------------------------------------------------------
    @staticmethod
    def _normalize_filter_input(values):                                               # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        """Normalize singleton and iterable filter inputs to tuples."""
        if values in (None, (), [], ''):
            return ()
        if isinstance(values, bytes):
            return (values.decode(),)
        if isinstance(values, str):
            return (values,)
        try:
            return tuple(values)
        except TypeError:
            return (values,)

    #----------------------------------------------------------------------------------------------
    def meta(self, *keys):                                                            # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        """"Return metadata for summary vectors, optionally filtered by key patterns."""
        if not self._vector_meta:
            self._load_vector_metadata()
        if not keys:
             return self._vector_meta
        return [m for m in self._vector_meta if m.key.startswith(keys)]

    #----------------------------------------------------------------------------------------------
    def list_keys(self):                                                              # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        """Return all summary keyword names."""
        keys = next(self.blockdata('KEYWORDS')) if self.is_file() else nparray()
        return sorted(set(keys.tolist()))

    #----------------------------------------------------------------------------------------------
    def select_vectors(self, keys=(), names=(), nums=()):                             # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        """Select vectors from SMSPEC metadata preserving original order."""
        vectors = self._load_vector_metadata()
        if not vectors:
            return ()
        key_patterns = self._normalize_filter_input(keys)
        name_patterns = self._normalize_filter_input(names)
        num_values = self._normalize_filter_input(nums)
        num_filter = {int(n) for n in num_values}
        if not key_patterns and not name_patterns and not num_filter:
            return vectors
        selected = []
        for vec in vectors:
            if key_patterns and not any(fnmatch(vec.key, pattern) for pattern in key_patterns):
                continue
            if name_patterns and not any(fnmatch(vec.name, pattern) for pattern in name_patterns):
                continue
            if num_filter and vec.num not in num_filter:
                continue
            selected.append(vec)
        return tuple(selected)

    #----------------------------------------------------------------------------------------------
    def welldata(self, keys=(), wells=(), named=False):                               # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        """Return data for the requested well."""
        # print('SMSPEC WELLDATA', self.is_file())
        self._inkeys = keys
        if not self.is_file():
            return False
        self._load_vector_metadata()
        if self.data.keys and self.data.wells and self.data.units:
            keys = keys or set(self.data.keys)
            all_wells = set(w for w in self.data.wells if w and not '+' in w)
            patterns = (w.split('*')[0] for w in wells if '*' in w)
            matched = [w for p in patterns for w in all_wells if w.startswith(p)]
            wells = [w for w in wells if '*' not in w]
            wells = set(wells+matched) or all_wells
            ikw = enumerate(zip(self.data.keys, self.data.wells))
            # index into UNSMRY arrays
            self._ind = tuple(i for i,(k,w) in ikw if k in keys and w in wells)
            #print(self._ind)
            if self._ind:
                getter = itemgetter(*self._ind)
                if self.data.measures:
                    self.measures = getter(tuple(self.data.measures))
                if named:
                    self.wells = getter(tuple(w.replace('-','_') for w in self.data.wells))
                else:
                    self.wells = getter(tuple(self.data.wells))
                if not isinstance(self.wells, (list, tuple)):
                    self.wells = (self.wells,)
                return True
        return False

    #----------------------------------------------------------------------------------------------
    def startdate(self):                                                              # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        """Return the summary start date."""
        if (start := next(self.blockdata('STARTDAT'), None)) is not None:
            day, month, year, hour, minute, second = start
            return datetime(year, month, day, hour, minute, second)

    #----------------------------------------------------------------------------------------------
    def __getattr__(self, item):                                                      # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        """
        Read attributes from the named-tuple Data
        """
        if (val := getattr(self.data, item, None)) is not None:
            if not self._ind:
                return ()
            items = itemgetter(*self._ind)(val)
            # Need this check because the return type of itemgetter is not consistent
            # For single indices it returns a value instead of a list 
            if len(self._ind) > 1:
                return items
            return (items,)
        return super().__getattr__(item)

    #----------------------------------------------------------------------------------------------
    def check_missing_keys(self):                                                     # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        """Ensure that the requested keywords exist."""
        return [a for a in self._inkeys if not a in self.keys]

    #----------------------------------------------------------------------------------------------
    def well_pos(self):                                                               # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        """Return coordinates for wells."""
        return self._ind


#==================================================================================================
class RSSPEC_file(unfmt_file):                                                        # RSSPEC_file
#==================================================================================================
    """
    RSSPEC (Restart Specification File)
    Binary unformatted index file describing the structure and location of data 
    arrays in UNRST files. Lists vector names, data types, sizes, and file offsets 
    for use by post-processors or custom readers.
    """

    start = 'INTEHEAD'

    #----------------------------------------------------------------------------------------------
    def __init__(self, file):                                                         # RSSPEC_file
    #----------------------------------------------------------------------------------------------
        """Initialize the RSSPEC_file."""
        super().__init__(file, suffix='.RSSPEC')
        self._units = None

    #----------------------------------------------------------------------------------------------
    def units(self, *keys):
    #----------------------------------------------------------------------------------------------
        """
        Return the unit string for a given key in the UNRST file.

        Args:
            key (str): The name of the variable whose unit is to be retrieved.

        Returns:
            str: The unit string associated with the given key.

        Raises:
            ValueError: If the 'NAME' or 'UNITS' block is missing in the RSSPEC file.
            KeyError: If the specified key is not found in the units dictionary.
        """
        if self._units is None:
            blockdata = next(self.blockdata('NAME', 'UNITS'), None)
            if blockdata is None:
                raise ValueError("Missing 'NAME' or 'UNITS' block in RSSPEC file")
            names, unitvals = blockdata
            self._units = dict(zip(map(str, names), map(str, unitvals)))
        unit_str = [self._units.get(str(key), None) for key in keys]
        if None in unit_str:
            missing_keys = [key for key, unit in zip(keys, unit_str) if unit is None]
            raise KeyError(
                f'Keys {missing_keys} not found in RSSPEC units. Available keys: {list(self._units.keys())}')
        return unit_str
