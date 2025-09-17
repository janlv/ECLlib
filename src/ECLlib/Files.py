
# -*- coding: utf-8 -*-

from fnmatch import fnmatch
from itertools import chain, product, repeat, accumulate, groupby, islice
from math import hypot, prod
from operator import attrgetter, itemgetter
from pathlib import Path
from platform import system
from mmap import mmap
from re import MULTILINE, finditer, findall, search as re_search
from collections import namedtuple
from datetime import datetime, timedelta
from struct import pack
#from locale import getpreferredencoding
from numpy import array as nparray, stack, sum as npsum, zeros, ones
from matplotlib.pyplot import figure as pl_figure
#from pandas import DataFrame
from pyvista import CellType, UnstructuredGrid

from .utils import (batched, cumtrapz, decode, flatten, flat_list,
                    flatten_all, grouper, list2text, remove_chars, float_or_str,
                    matches, split_by_words)
from .core import File, Restart, DTYPE
from .constants import ENDIAN
from .unformatted import unfmt_block, unfmt_file

#
#
#  import IORlib.ECL as ECL
#  import sys
#  fname = sys.argv[1]
#  for block in ECL.unformatted_file(fname).blocks():
#      if block.key()=='SEQNUM':
#           block.print()
#
#
#



#==================================================================================================
class DATA_file(File):
#==================================================================================================
    # Sections
    section_names = ('RUNSPEC','GRID','EDIT','PROPS' ,'REGIONS', 'SOLUTION', 'SUMMARY',
                     'SCHEDULE','OPTIMIZE')
    # Global keywords
    global_kw = ('COLUMNS','DEBUG','DEBUG3','ECHO','END', 'ENDINC','ENDSKIP','SKIP',
                 'SKIP100','SKIP300','EXTRAPMS','FORMFEED','GETDATA', 'INCLUDE','MESSAGES',
                 'NOECHO','NOWARN','WARN')
    # Common keywords
    common_kw = ('TITLE','CART','DIMENS','FMTIN','FMTOUT','GDFILE', 'FMTOUT','UNIFOUT','UNIFIN',
                 'OIL','WATER','GAS','VAPOIL','DISGAS','FIELD','METRIC','LAB','START','WELLDIMS',
                 'REGDIMS','TRACERS', 'NSTACK','TABDIMS','NOSIM','GRIDFILE','DX','DY','DZ','PORO',
                 'BOX','PERMX','PERMY','PERMZ','TOPS', 'INIT','RPTGRID','PVCDO','PVTW','PVTO','SGOF','SWOF',
                 'DENSITY','PVDG','ROCK','RPTPROPS','SPECROCK','SPECHEAT','TRACER','TRACERKP', 'TRDIFPAR',
                 'TRDIFIDE','SATNUM','FIPNUM','TRKPFPAR','TRKPFIDE','RPTSOL','RESTART','PRESSURE','SWAT',
                 'SGAS','RTEMPA','TBLKFA1','TBLKFIDE','TBLKFPAR','FOPR','FOPT','FGPR','FGPT',
                 'FWPR','FWPT','FWCT','FWIR', 'FWIT','FOIP','ROIP','WTPCHEA','WOPR','WWPR','WWIR',
                 'WBHP','WWCT','WOPT','WWIT','WTPRA1','WTPTA1','WTPCA1', 'WTIRA1','WTITA1',
                 'WTICA1','CTPRA1','CTIRA1','FOIP','ROIP','FPR','TCPU','TCPUTS','WNEWTON',
                 'ZIPEFF','STEPTYPE','NEWTON','NLINEARP','NLINEARS','MSUMLINS','MSUMNEWT',
                 'MSUMPROB','WTPRPAR','WTPRIDE','WTPCPAR','WTPCIDE','RUNSUM', 'SEPARATE',
                 'WELSPECS','COMPDAT','WRFTPLT','TSTEP','DATES','SKIPREST','WCONINJE','WCONPROD',
                 'WCONHIST','WTEMP','RPTSCHED', 'RPTRST','TUNING','READDATA', 'ROCKTABH',
                 'GRIDUNIT','NEWTRAN','MAPAXES','EQLDIMS','ROCKCOMP','TEMP', 'GRIDOPTS',
                 'VFPPDIMS','VFPIDIMS','AQUDIMS','SMRYDIMS','CPR','FAULTDIM','MEMORY','EQUALS',
                 'MINPV','COPY','ADD','MULTIPLY', 'SPECGRID', 'COORD', 'ZCORN', 'ACTNUM')

    #----------------------------------------------------------------------------------------------
    def __init__(self, file, suffix=None, check=False, sections=True, **kwargs):      # DATA_file
    #----------------------------------------------------------------------------------------------
        #print(f'Input_file({file}, check={check}, read={read}, reread={reread}, include={include})')
        suffix = Path(file).suffix or suffix or '.DATA'
        super().__init__(file, suffix=suffix, role='Eclipse input-file', **kwargs)
        self.data = None
        self._checked = False
        self._added_files = ()
        if not sections:
            self.section_names = ()
        getter = namedtuple('getter', 'section default convert pattern')
        self._getter = {
            'TSTEP'   : getter('SCHEDULE', (),      self._convert_float,
                               r'\bTSTEP\b\s+([0-9*.\s]+)/\s*'),
            'START'   : getter('RUNSPEC',  (0,),    self._convert_date,
                               r'\bSTART\b\s+(\d+\s+\'*\w+\'*\s+\d+)'),
            'DATES'   : getter('SCHEDULE', (),      self._convert_date,
                               r'\bDATES\b\s+((\d{1,2}\s+\'*\w{3}\'*\s+\d{4}\s*\s*/\s*)+)/\s*'),
            'RESTART' : getter('SOLUTION', ('', 0), self._convert_file,
                               r"\bRESTART\b\s+('*[\w./\\-]+'*\s+[0-9]+)\s*/"),
            'WELSPECS': getter('SCHEDULE', (),      self._convert_string,
                               r'\bWELSPECS\b((\s+\'*[\w/-]+?.*/\s*)+/)')}
        if check:
            self.check()


    #----------------------------------------------------------------------------------------------
    def __repr__(self):                                                   # DATA_file
    #----------------------------------------------------------------------------------------------
        return f'<{type(self)}, {self.path}>'

    #----------------------------------------------------------------------------------------------
    def __call__(self):                                                  # DATA_file
    #----------------------------------------------------------------------------------------------
        self.data = self.binarydata()
        return self

    #----------------------------------------------------------------------------------------------
    def __contains__(self, key):                                          # DATA_file
    #----------------------------------------------------------------------------------------------
        self.data = None
        return bool(self.search(key, regex=rf'^[ \t]*{key}', comments=True))
        
    #----------------------------------------------------------------------------------------------
    def mode(self):                                                       # DATA_file
    #----------------------------------------------------------------------------------------------
        return 'backward' if ('READDATA' in self) else 'forward'


    #----------------------------------------------------------------------------------------------
    def restart(self):                                                  # DATA_file
    #----------------------------------------------------------------------------------------------
        # Check if this is a restart-run
        file, step = self.get('RESTART')
        if file and step:
            # Get time and step from the restart-file
            file = UNRST_file(file)
            if not file.is_file():
                raise SystemError(f'ERROR Restart file {file.path} is missing')
            days, n = next(((t,s) for t,s in file.read('time', 'step') if s >= step), (-1,-1))
            if n != step:
                raise SystemError(f'ERROR Step {step} is missing in restart file {file}')
            start = next(file.dates())
            return Restart(start=start, days=days, step=n)
        # Get start from DATA-file
        start = self.start()
        return Restart(start=self.start(), step=step)


    #----------------------------------------------------------------------------------------------
    def binarydata(self, raise_error=False):                              # DATA_file
    #----------------------------------------------------------------------------------------------
        self.data = super().binarydata(raise_error)
        end = b'END' in self.data and re_search(rb'^[ \t]*\bEND\b', self.data, flags=MULTILINE)
        return end and self.data[:end.end()] or self.data

    #----------------------------------------------------------------------------------------------
    def check(self, include=True, uppercase=False):                          # DATA_file
    #----------------------------------------------------------------------------------------------
        self._checked = True
        # Check if file exists
        self.exists(raise_error=True)
        # If Linux, check that file name is all capital letters to avoid I/O error in Eclipse        
        if uppercase and self.suffix == '.DATA' and system() == 'Linux' and not self.name.isupper():
            err = (f'ERROR *{self.name}* must all be in uppercase letters. Under Linux, Eclipse only '
                   'accepts DATA-files with uppercase letters')
            raise SystemError(err)
        # Check if included files exists
        files = chain(self.include_files(), self.grid_files())
        if include and (missing := [f for f in files if not f.is_file()]):
            err = (f'ERROR {list2text([f.name for f in missing])} included from {self} is '
                   f'missing in folder {missing[0].parent}')
            raise SystemError(err)
        return True

    #----------------------------------------------------------------------------------------------
    def search(self, key, regex, comments=False):                         # DATA_file
    #----------------------------------------------------------------------------------------------
        data = self._matching(key)
        if not comments:
            self.data = self._remove_comments(data)
        else:
            self.data = decode(b''.join(data))
        return re_search(regex, self.data, flags=MULTILINE)

    #----------------------------------------------------------------------------------------------
    def is_empty(self):                                                  # DATA_file
    #----------------------------------------------------------------------------------------------
        """ Check if file is empty """
        return self._remove_comments() == ''

    #----------------------------------------------------------------------------------------------
    def include_files(self, data:bytes=None):                           # DATA_file
    #----------------------------------------------------------------------------------------------
        """ Return full path of INCLUDE files as a generator """
        return (f[0] for f in self._included_file_data(data))

    #----------------------------------------------------------------------------------------------
    def _included_file_data(self, data:bytes=None):                           # DATA_file
    #----------------------------------------------------------------------------------------------
        """ Return tuple of filename and binary-data for each include file """
        data = data or self.binarydata()
        # This regex is explained at: https://regex101.com/r/jTYq16/2
        regex = rb"^\s*(?:\bINCLUDE\b)(?:\s*--.*\s*|\s*)*'*([^' ]+)['\s]*/.*$"
        files = (m.group(1).decode() for m in finditer(regex, data, flags=MULTILINE))
        for file in chain(files, self._added_files):
            new_filename = self.with_name(file)
            file_data = DATA_file(new_filename).binarydata()
            yield (new_filename, file_data)
            if b'INCLUDE' in file_data:
                for inc in self._included_file_data(file_data):
                    yield inc

    #----------------------------------------------------------------------------------------------
    def grid_files(self, data:bytes=None):                           # DATA_file
    #----------------------------------------------------------------------------------------------
        data = data or self.binarydata()
        regex = rb"^\s*(?:\bGDFILE\b)(?:\s*--.*\s*|\s*)*'*([^' ]+)['\s]*/.*$"
        files = (self.with_name(m.group(1).decode()) for m in finditer(regex, data, flags=MULTILINE))
        return (file.with_suffix(file.suffix or '.EGRID') for file in files)

    #----------------------------------------------------------------------------------------------
    def including(self, *files):                                          # DATA_file
    #----------------------------------------------------------------------------------------------
        """ Add the given files and return self """
        # Added files must be an iterator to avoid an infinite recursive
        # loop when self._added_files is called in _included_file_data
        self._added_files = iter(files)
        # Disable check to avoid check to consume the above iterator
        self._checked = True
        return self

    #----------------------------------------------------------------------------------------------
    def start(self):                                                      # DATA_file
    #----------------------------------------------------------------------------------------------
        return self.get('START')[0]

    #----------------------------------------------------------------------------------------------
    def timesteps(self, start=None, negative_ok=False, missing_ok=False, pos=False, skiprest=False):     # DATA_file
    #----------------------------------------------------------------------------------------------
        """ Return tsteps, if DATES are present they are converted to tsteps """
        _start, tsteps, dates = self.get('START','TSTEP','DATES', pos=True)
        if not tsteps and not dates:
            return ()
        #print(_start, tsteps, dates)
        if skiprest:
            tsteps = []
            negative_ok = True
        times = sorted(dates+tsteps, key=itemgetter(1))
        start = start or _start[0][0]
        if not start:
            raise SystemError('ERROR Missing start-date in DATA_file.tsteps()')
        tsteps = tuple(self._days(times, start=start))
        ## Checks
        if not negative_ok and any(t<=0 for t,_ in tsteps):
            raise SystemError(
                f'ERROR Zero or negative timestep in {self} (check if TSTEP or RESTART oversteps a DATES keyword)')
        if not missing_ok and not tsteps:
            raise SystemError(f'ERROR No TSTEP or DATES in {self} (or the included files)')
        return tsteps if pos else tuple(next(zip(*tsteps)))
        #return pos and tsteps or tuple(next(zip(*tsteps))) # Do not return positions

    #----------------------------------------------------------------------------------------------
    def report_dates(self):                                               # DATA_file
    #----------------------------------------------------------------------------------------------
        return [self.start() + timedelta(days=days) for days in accumulate(self.timesteps())]
    
    #----------------------------------------------------------------------------------------------
    def wellnames(self):                                                  # DATA_file
    #----------------------------------------------------------------------------------------------
        """
        Return tuple of wellnames from WELSPECS and UNRST file for RESTART runs
        """
        wells = self.welspecs()
        restart, step = self.get('RESTART')
        if restart and step:
            unrst = UNRST_file(restart, role='RESTART file')
            unrst.exists(raise_error=True)
            wells += unrst.wells(stop=int(step))
        return list(set(wells))

    #----------------------------------------------------------------------------------------------
    def welspecs(self):                                                   # DATA_file
    #----------------------------------------------------------------------------------------------
        """
        Get wellnames from WELSPECS definitions in the DATA-file or in a
        schedule-file
        """
        welspecs = self.get('WELSPECS')
        if not welspecs or not welspecs[0]:
            # If no WELSPECS in DATA-file, look for WELSPECS in a separate SCH-file 
            # This is the case for backward runs
            sch_file = self.with_suffix('.SCH', ignore_case=True, exists=True)
            if sch_file:
                welspecs = DATA_file(sch_file, sections=False).get('WELSPECS')
        # The wellname is the first value, but it might contain spaces. If so, it is quoted
        # and we need to check if the first char is a quote or not. If the line starts with
        # a quote, we split on quote+space, otherwise we just split on space
        splits = (w.split("' ") if w.startswith("'") else w.split() for w in welspecs if w)
        return tuple(set(s[0].replace("'","") for s in splits))

    #----------------------------------------------------------------------------------------------
    def get(self, *keywords, raise_error=False, pos=False):                # DATA_file
    #----------------------------------------------------------------------------------------------
        #print('get', keywords)
        #FAIL = len(keywords)*((),)
        keywords = [key.upper() for key in keywords]
        getters = [self._getter.get(key) for key in keywords]
        FAIL = [g.default for g in getters]
        FAIL = FAIL[0] if len(FAIL) == 1 else FAIL
        #print(FAIL)
        if not self.exists(raise_error=raise_error):
            return FAIL
        if missing:=[k for g,k in zip(getters, keywords) if not g]:
            if raise_error:
                raise SystemError(f'ERROR Missing get-pattern for {list2text(missing)} in DATA_file')
            return FAIL
        names = set(g.section for g in getters)
        self.data = self._remove_comments(self.section(*names)._matching(*keywords))
        error_msg = f'ERROR Keyword {list2text(keywords)} not found in {self.path}'
        if not self.data:
            if raise_error:
                raise SystemError(error_msg)
            return FAIL
        result = ()
        for keyword, getter in zip(keywords, getters):
            # match_list = re_compile(getter.pattern).finditer(self.data)
            # val_span = tuple((m.group(1), m.span()) for m in match_list) 
            val_span = tuple((m.group(1), m.span()) for m in finditer(getter.pattern, self.data)) 
            if not val_span:
                result += (getter.default,)
                continue
            values, span = zip(*val_span)
            values = getter.convert(values, keyword)
            if pos:
                values = (tuple(zip(v,repeat(p))) for v,p in zip(values, span))
            result += (flat_list(values),)
        if len(result) == 1:
            return result[0]
        return result

    #----------------------------------------------------------------------------------------------
    def lines(self):                                                       # DATA_file
    #----------------------------------------------------------------------------------------------
        return (line for line in self._remove_comments(self._matching()).split('\n') if line)

    #----------------------------------------------------------------------------------------------
    def text(self):                                                       # DATA_file
    #----------------------------------------------------------------------------------------------
        return self._remove_comments(self._matching())

    #----------------------------------------------------------------------------------------------
    def summary_keys(self, matching=()):                                # DATA_file
    #----------------------------------------------------------------------------------------------
        return [k for k in self.section('SUMMARY').text().split() if k in matching]

    #----------------------------------------------------------------------------------------------
    def section_positions(self, *sections):                               # DATA_file
    #----------------------------------------------------------------------------------------------
        data = self.data or self.binarydata()
        sec_pos = {sec.upper().decode():(a,b) for sec,a,b in split_by_words(data, self.section_names)}
        if not sections:
            return sec_pos
        return  {sec:pos for sec in sections if (pos := sec_pos.get(sec))}

    #----------------------------------------------------------------------------------------------
    def section(self, *sections, raise_error=True):                       # DATA_file
    #----------------------------------------------------------------------------------------------
        #print('section', sections)
        if not self._checked:
            self.check()
        self.data = self.binarydata()
        ### Get section-names and file positions
        if not self.section_names:
            return self
        sec_pos = self.section_positions(*sections)
        if not sec_pos:
            if raise_error:
                raise SystemError(f'ERROR Section {list2text(sections)} not found in {self}')
            return self
        self.data = b''.join(self.data[a:b] for a,b in sorted(sec_pos.values()))
        return self

    #----------------------------------------------------------------------------------------------
    def replace_keyword(self, keyword, new_string):                      # DATA_file
    #----------------------------------------------------------------------------------------------
        ### Get keyword value and position in file
        match = self.get(keyword, pos=True)
        if match:
            _, pos = match[0] # Get first match
        else:
            raise SystemError(f'ERROR Missing {keyword} in {self}')
        out = self.data[:pos[0]] + new_string + self.data[pos[1]:]
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(out)

    #----------------------------------------------------------------------------------------------
    def _remove_comments(self, data=None):                   # DATA_file
    #----------------------------------------------------------------------------------------------
        data = data or (self.binarydata(),)
        lines = (l for d in data for l in d.split(b'\n'))
        text = (l.split(b'--')[0].strip() for l in lines)
        text = b'\n'.join(t for t in text if t)
        text = decode(text)
        return text+'\n' if text else ''

    #----------------------------------------------------------------------------------------------
    def _matching(self, *keys):                                           # DATA_file
    #----------------------------------------------------------------------------------------------
        #print('_matching', keys)
        self.data = self.data or self.binarydata()
        keys = [key.encode() for key in keys]
        if keys == [] or any(key in self.data for key in keys):
            yield self.data
        for file, data in self._included_file_data(self.data):
            if keys == [] or any(key in data for key in keys):
                yield data

    #----------------------------------------------------------------------------------------------
    def _days(self, time_pos, start=None):                           # DATA_file
    #----------------------------------------------------------------------------------------------
        """ Return relative timestep in days given a timestep or a datetime """
        last_date = start
        for t,p in time_pos:
            if isinstance(t, datetime):
                dt = t
            else:
                dt = last_date + timedelta(hours=t*24)
            yield (dt-last_date).total_seconds()/86400, p
            last_date = dt
            
    #----------------------------------------------------------------------------------------------
    def _convert_string(self, values, key):                               # DATA_file
    #----------------------------------------------------------------------------------------------
        ret = [v for val in values for v in val.split('\n') if v and v != '/']
        return (ret,)

    #----------------------------------------------------------------------------------------------
    def _convert_float(self, values, key):                                # DATA_file
    #----------------------------------------------------------------------------------------------
        #mult = lambda x, y : list(repeat(float(y),int(x))) # Process x*y statements
        def mult(x,y):
            # Process x*y statements
            return list(repeat(float(y),int(x)))
        values = ([mult(*n.split('*')) if '*' in n else [float(n)] for n in v.split()] for v in values)
        values = tuple(flat_list(v) for v in values)
        return values or self._getter[key].default

    #----------------------------------------------------------------------------------------------
    def _convert_date(self, dates, key):                                  # DATA_file
    #----------------------------------------------------------------------------------------------
        ### Remove possible quotes
        ### Extract groups of 3 from the dates strings 
        dates = (grouper(remove_chars("'/\n", v).split(), 3) for v in dates)
        dates = tuple([datetime.strptime(' '.join(d), '%d %b %Y') for d in date] for date in dates)
        return dates or self._getter[key].default

    #----------------------------------------------------------------------------------------------
    def _convert_file(self, values, key):                                 # DATA_file
    #----------------------------------------------------------------------------------------------
        """ Return full path of file """
        ### Remove quotes and backslash
        values = (val.replace("'",'').replace('\\','/').split() for val in values)
        ### Unzip values in a files (always) and numbers lists (only for RESTART)
        unzip = zip(*values)
        files = ([(self.path.parent/file).resolve()] for file in next(unzip))
        numbers = [[float(num)] for num in next(unzip, ())]
        files = tuple([f[0],n[0]] for f,n in zip(files, numbers)) if numbers else tuple(files)
        #print(key, files)
        ### Add suffix for RESTART keyword
        if key == 'RESTART' and files:
            files[0][0] = files[0][0].with_suffix('.UNRST')
        return files or self._getter[key].default


#==================================================================================================
class EGRID_file(unfmt_file):                                            # EGRID_file
#==================================================================================================
    start = 'FILEHEAD'
    end = 'ENDGRID'
    #----------------------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                  # EGRID_file
    #----------------------------------------------------------------------------------------------
        super().__init__(file, suffix='.EGRID', **kwargs)
        self.var_pos = {'nx': ('GRIDHEAD', 1),
                        'ny': ('GRIDHEAD', 2),
                        'nz': ('GRIDHEAD', 3),
                        'unit': ('MAPUNITS', 0)}
        self._nijk = None
        self._coord_zcorn = None

    #----------------------------------------------------------------------------------------------
    def length(self):                                                    # EGRID_file
    #----------------------------------------------------------------------------------------------
        convert = {'METRES':1.0, 'FEET':0.3048, 'CM':0.01}
        unit = next(self.blockdata('MAPUNITS'), None)
        if unit:
            return convert[unit]
        raise SystemError(f'ERROR Missing MAPUNITS in {self}')

    #----------------------------------------------------------------------------------------------
    def axes(self):                                                      # EGRID_file
    #----------------------------------------------------------------------------------------------
        ax = next(self.blockdata('MAPAXES'), None)
        origin = (ax[2], ax[3])
        unit_x = (ax[4]-ax[2], ax[5]-ax[3])
        unit_y = (ax[0]-ax[2], ax[1]-ax[3])
        norm_x = 1 / hypot(*unit_x)
        norm_y = 1 / hypot(*unit_y)
        return origin, (unit_x[0]*norm_x, unit_x[1]*norm_x), (unit_y[0]*norm_y, unit_y[1]*norm_y)

    #----------------------------------------------------------------------------------------------
    def nijk(self):                                           # EGRID_file
    #----------------------------------------------------------------------------------------------
        self._nijk = self._nijk or next(self.read('nx', 'ny', 'nz'))
        return self._nijk        

    #----------------------------------------------------------------------------------------------
    def coord_zcorn(self):                                           # EGRID_file
    #----------------------------------------------------------------------------------------------
        self._coord_zcorn = self._coord_zcorn or list(map(nparray, next(self.blockdata('COORD', 'ZCORN'))))
        return self._coord_zcorn

    #----------------------------------------------------------------------------------------------
    def _indices(self, ijk):                                         # EGRID_file
    #----------------------------------------------------------------------------------------------
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
    def cell_corners(self, ijk_iter):                                     # EGRID_file
    #----------------------------------------------------------------------------------------------
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
    def grid(self, i=None, j=None, k=None, scale=(1,1,1)):                     # EGRID_file
    #----------------------------------------------------------------------------------------------
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
    def cells(self, **kwargs):                                            # EGRID_file
    #----------------------------------------------------------------------------------------------
        points = self.grid(**kwargs).cell_centers().points
        dim = INIT_file(self.path).dim() + (3,)
        return points.reshape(dim, order='C')


#==================================================================================================
class INIT_file(unfmt_file):                                              # INIT_file
#==================================================================================================
    start = 'INTEHEAD'

    #----------------------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                   # INIT_file
    #----------------------------------------------------------------------------------------------
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
    def dim(self):                                                       # INIT_file
    #----------------------------------------------------------------------------------------------
        self._dim = self._dim or next(self.read('nx', 'ny', 'nz'))
        return self._dim
    
    # #--------------------------------------------------------------------------------
    # def reshape_dim(self, *data, dtype=None):                             # INIT_file
    # #--------------------------------------------------------------------------------
    #     return [asarray(d, dtype=dtype).reshape(self.dim(), order='F') for d in data]

    #----------------------------------------------------------------------------------------------
    def cell_ijk(self, *cellnum):                                         # INIT_file
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
    def non_neigh_conn(self):                                             # INIT_file
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
    def simulator(self):                                                  # INIT_file
    #----------------------------------------------------------------------------------------------
        sim_codes = {100:'ecl', 300:'ecl', 500:'ecl', 700:'ix', 800:'FrontSim'}
        if sim:=next(self.read('simulator'), None):
            if sim < 0:
                return 'other simulator'
            return sim_codes[sim]

    #----------------------------------------------------------------------------------------------
    def start_date(self):                                                  # INIT_file
    #----------------------------------------------------------------------------------------------
        keys = ('year', 'month', 'day', 'hour', 'minute', 'second')
        if data := next(self.read(*keys), None):
            kwargs = dict(zip(keys, data))
            # Unit of second is microsecond
            kwargs['second'] = int(kwargs['second']/1e6)
            return datetime(**kwargs)

    # #--------------------------------------------------------------------------------
    # def is_flushed(self, end='TRANNNC'):                                  # INIT_file
    # #--------------------------------------------------------------------------------
    #     return super().is_flushed(end)

#==================================================================================================
class UNRST_file(unfmt_file):                                            # UNRST_file
#==================================================================================================
    start = 'SEQNUM'
    end = 'ENDSOL'
    #           variable   keyword   position (None = whole array)
    var_pos =  {'step'  : ('SEQNUM'  ,  0),
                'nx'    : ('INTEHEAD',  8),
                'ny'    : ('INTEHEAD',  9),
                'nz'    : ('INTEHEAD', 10),
                'nwell' : ('INTEHEAD', 16),
                'day'   : ('INTEHEAD', 64),
                'month' : ('INTEHEAD', 65),
                'year'  : ('INTEHEAD', 66),
                'hour'  : ('INTEHEAD', 206),
                'min'   : ('INTEHEAD', 207),
                'sec'   : ('INTEHEAD', 410),
                'time'  : ('DOUBHEAD', 0),
                'wells' : ('ZWEL'    , None)}  # No ZWEL in first section 
                                                
    #----------------------------------------------------------------------------------------------
    # def __init__(self, file, suffix='.UNRST', wait_func=None, end=None, role=None, 
    #              **kwargs):                                              # UNRST_file
    def __init__(self, file, suffix='.UNRST', end=None, role=None):      # UNRST_file
    #----------------------------------------------------------------------------------------------
        super().__init__(file, suffix=suffix, role=role)
        self.end = end or self.end
        #self.check = check_blocks(self, start=self.start, end=self.end, wait_func=wait_func, **kwargs)
        self._dim = None
        self._units = None
        #self._dates = None

    #----------------------------------------------------------------------------------------------
    def __len__(self):                                                   # UNRST_file
    #----------------------------------------------------------------------------------------------
        return len(list(self.steps()))

    #----------------------------------------------------------------------------------------------
    def dim(self):                                                       # UNRST_file
    #----------------------------------------------------------------------------------------------
        self._dim = self._dim or next(self.read('nx', 'ny', 'nz'))
        return self._dim
    
    # #--------------------------------------------------------------------------------
    # def reshape_dim(self, *data, dtype=None):                         # UNRST_file
    # #--------------------------------------------------------------------------------
    #     return [asarray(d, dtype=dtype).reshape(self.dim(), order='F') for d in data]
    #     # if flip:
    #     #     return [npflip(a, axis=-1) for a in arr]
    #     # return arr

    #----------------------------------------------------------------------------------------------
    def _check_for_missing_keys(self, *in_keys, keys=None):              # UNRST_file
    #----------------------------------------------------------------------------------------------
        keys = keys or self.find_keys(*in_keys)
        if missing := [ik for ik in in_keys if not any(fnmatch(k, ik) for k in keys)]:
            raise ValueError(f'The following keywords are missing in {self}: {missing}')
        return keys

    # #--------------------------------------------------------------------------------
    # def cell_ijk(self, *cellnum):                                    # UNRST_file
    # #--------------------------------------------------------------------------------
    #     """
    #     Return ijk-indices of cells given a list of cell-numbers
    #     """
    #     dim = self.dim()
    #     ni, nij = dim[0], dim[0]*dim[1]
    #     for cell in cellnum:
    #         cell -= 1
    #         yield (cell % ni, (cell % nij) // ni, cell // nij)

    # #--------------------------------------------------------------------------------
    #def cell_ijk(self, *cellnum):                                    # UNRST_file
    # #--------------------------------------------------------------------------------
    #     """
    #     Return ijk-indices of cells given a list of cell-numbers
    #     """
        # dim = self.dim()
        # ni, nij = dim[0], dim[0]*dim[1]
        # cellnum = nparray(cellnum) - 1
        # i = cellnum % ni
        # j = (cellnum % nij) // ni
        # k = cellnum // nij
        # return nparray([i, j, k]).T


    #----------------------------------------------------------------------------------------------
    def _cellnr(self, coord, base=0):                                    # UNRST_file
    #----------------------------------------------------------------------------------------------
        """
        Return position in 1D array given 3D coordinate of base=0 (default) or base=1
        """
        dim = self.dim()
        # Apply negative index from the end
        coord = [c if c>=0 else dim[i]+c+base for i,c in enumerate(coord)]
        return base + coord[0]-base + dim[0]*(coord[1]-base) + dim[0]*dim[1]*(coord[2]-base)

    #----------------------------------------------------------------------------------------------
    def celldata(self, coord, *keywords, base=0):                         # UNRST_file
    #----------------------------------------------------------------------------------------------
        """
        Return the given keywords as a celldata namedtuple for the given cell-coordinate.
        
        Keyword arguments 
            base     : Zero- or one-based indexing (0 is default)
            time_res : Time resolution, valid values are 'day', 'hour', 'min', 'sec' ('day' is default)
        """
        self._check_for_missing_keys(*keywords)
        cellnr = self._cellnr(coord, base=base)
        args = flatten((key, cellnr) for key in keywords)
        data = (zip(*self.blockdata(*args, singleton=False)))
        celldata = namedtuple('celldata', ('days',)+keywords)
        return celldata(tuple(self.days()), *data)
        #return celldata(tuple(self.days(resolution=time_res)), *data)
        # for dd in zip(self.days(resolution=time_res), *data):
        #     yield celldata(*dd)

    # #--------------------------------------------------------------------------------
    # def celldata_as_dataframe(self, *args, **kwargs):        # UNRST_file
    # #--------------------------------------------------------------------------------
    #     """
    #     Write given keywords for the given cell-coordinate to a tab-separated file
    #     """
    #     # data = self.celldata(coord, *keywords, base=base, time_res=time_res)
    #     data = self.celldata(*args, **kwargs)
    #     return DataFrame(data._asdict())

    #----------------------------------------------------------------------------------------------
    def cellarray(self, *in_keys, start=None, stop=None, step=1, warn_missing=True):   # UNRST_file                  
    #----------------------------------------------------------------------------------------------
        step = step or self.count_sections()
        keys = self.find_keys(*in_keys)
        if warn_missing:
            self._check_for_missing_keys(*in_keys, keys=keys)
        names = [remove_chars('+-', k) for k in keys]
        celltuple = namedtuple('cellarray', ['days', 'date'] + names)
        dim = self.dim()
        dds = zip(self.days(), self.dates(), self.section_blocks())
        for day, date, section in islice(dds, start, stop, step):
            blockdata = {k:None for k in keys}
            for block in section:
                if (key:=block.key()) in keys:
                    blockdata[key] = block.data()
            yield celltuple(day, date, *[nparray(d).reshape(dim, order='F') for d in blockdata.values()])
    
    # #--------------------------------------------------------------------------------
    # def griddata(self, grid, *keys, start=None, stop=None, step=1):   # UNRST_file                  
    # #--------------------------------------------------------------------------------
    #     pass

    #----------------------------------------------------------------------------------------------
    def wells(self, stop=None):                                          # UNRST_file
    #----------------------------------------------------------------------------------------------
        wells = flatten_all(islice(self.read('wells'), 0, stop))
        unique_wells = set(w for well in wells if (w:=well.strip()))
        return tuple(unique_wells)

    #----------------------------------------------------------------------------------------------
    def open_wells(self):                                                # UNRST_file
    #----------------------------------------------------------------------------------------------
        for ihead, icon in self.blockdata('INTEHEAD', 'ICON'):
            niconz, ncwmax, nwells  = ihead[32], ihead[17], ihead[16]
            icon = nparray(icon).reshape((niconz, ncwmax, nwells), order='F')
            yield sum(npsum(icon[5,:,:], axis=0) > 0)


    #----------------------------------------------------------------------------------------------
    def steps(self):                                                     # UNRST_file
    #----------------------------------------------------------------------------------------------
        return flatten_all(self.read('step'))

    #----------------------------------------------------------------------------------------------
    def end_step(self):                                                  # UNRST_file
    #----------------------------------------------------------------------------------------------
        return self.last_value('step')

    #----------------------------------------------------------------------------------------------
    def end_time(self):                                                  # UNRST_file
    #----------------------------------------------------------------------------------------------
        return self.last_value('time')

    #----------------------------------------------------------------------------------------------
    def end_date(self):                                                  # UNRST_file
    #----------------------------------------------------------------------------------------------
        return next(self.dates(tail=True), None)

    #----------------------------------------------------------------------------------------------
    def dates(self, resolution='day', **kwargs):                         # UNRST_file
    #----------------------------------------------------------------------------------------------
        varnames = ('year', 'month', 'day')
        if resolution == 'day':
            pass
        elif resolution == 'hour':
            varnames += ('hour',)
        elif resolution == 'min':
            varnames += ('hour', 'min')
        elif resolution == 'sec':
            varnames += ('hour', 'min', 'sec')
            # Seconds are reported as microseconds, integer-divide by 1e6
            return (datetime(*vars[:-1], int(vars[-1]//1e6)) for vars in self.read(*varnames, **kwargs))
        else:
            raise SyntaxError("resolution must be 'hour', 'min', or 'sec'")
        return (datetime(*vars) for vars in self.read(*varnames, **kwargs))

    #----------------------------------------------------------------------------------------------
    def units(self):                                                     # UNRST_file
    #----------------------------------------------------------------------------------------------
        if self._units is None:
            ihead2 = next(self.blockdata('INTEHEAD', 2), None)
            if ihead2:
                self._units = {1:'metric', 2:'field', 3:'lab', 4:'pvt-m'}[ihead2]
        return self._units

    #----------------------------------------------------------------------------------------------
    def days(self, **kwargs):                                            # UNRST_file
    #----------------------------------------------------------------------------------------------
        # Read units only once
        convert = 1
        if self.units() == 'lab':
            # DOUBHEAD[0] is given in hours in lab units
            convert = 1/24
        return (next(flatten(dh))*convert for dh in self.blockdata('DOUBHEAD', singleton=True, **kwargs))

    #----------------------------------------------------------------------------------------------
    def section(self, days=None, date=None):                      # UNRST_file
    #----------------------------------------------------------------------------------------------
        stop = None
        if days:
            data_func = self.days
            stop = days
            return next(i for i,val in enumerate(self.days()) if val >= days)
        if date:
            data_func = self.dates
            stop = datetime(*date)
        if not stop:
            raise ValueError('Either days or date must be given')
        return next(i for i,val in enumerate(data_func()) if val >= stop)


    #----------------------------------------------------------------------------------------------
    def end_key(self):                                                   # UNRST_file
    #----------------------------------------------------------------------------------------------
        block = next(self.tail_blocks(), None)
        if block:
            return block.key()

    #----------------------------------------------------------------------------------------------
    def from_Xfile(self, xfile, log=False, delete=False):                 # UNRST_file
    #----------------------------------------------------------------------------------------------
        """
        Append a SEQNUM block at the beginning of the non-unified restart X-file. 
        """
        xfile = File(xfile)
        if not xfile.exists():
            raise FileNotFoundError(f'{xfile} is missing')
        # Add missing SEQNUM at beginning
        step = int(xfile.suffix[-4:])
        seqnum = unfmt_block.from_data('SEQNUM', [step], 'int')
        self.merge([(step, seqnum.as_bytes())], [(step, xfile.binarydata())])
        if delete:
            xfile.delete(raise_error=True)
            if log:
                log(f'Deleted {xfile}')
        if callable(log):
            log(f'Created {self} from {xfile}')
        #return self

    #----------------------------------------------------------------------------------------------
    def as_Xfiles(self, log=False, stop=None):                           # UNRST_file
    #----------------------------------------------------------------------------------------------
        for i, sec in enumerate(self.section_blocks()):
            xfile = self.with_suffix(f'.X{i:04d}')
            with open(xfile, 'wb') as file:
                for block in sec:
                    key = block.key()
                    if key != 'SEQNUM':
                        file.write(block.binarydata())
                    if key == 'ENDSOL':
                        break
            if callable(log):
                log(f'Wrote {xfile}')
            if stop and i == stop:
                return
            

# #====================================================================================
# class X_files:                                                              # X_files
# #====================================================================================

#     #--------------------------------------------------------------------------------
#     def __init__(self, root):                                               # X_files
#     #--------------------------------------------------------------------------------
#         self.root = root

#     #--------------------------------------------------------------------------------
#     def files(self):
#     #--------------------------------------------------------------------------------
#         return File(self.root).glob('*.X????')

#     #--------------------------------------------------------------------------------
#     def file(self, num):
#     #--------------------------------------------------------------------------------
#         unfmt_file(self.root.with_suffix(f'.X{num:04d}'))


#==================================================================================================
class RFT_file(unfmt_file):                                                # RFT_file
#==================================================================================================
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
    #def __init__(self, file, wait_func=None, **kwargs):                    # RFT_file
    def __init__(self, file):                                              # RFT_file
    #----------------------------------------------------------------------------------------------
        super().__init__(file, suffix='.RFT')
        #self.check = check_blocks(self, start=self.start, end=self.end, wait_func=wait_func, **kwargs)
        self.current_section = 0

    # #--------------------------------------------------------------------------------
    # def not_in_sync(self, time, prec=0.1):                                 # RFT_file
    # #--------------------------------------------------------------------------------
    #     data = self.check.data()
    #     if data and any(abs(nparray(data)-time) > prec):
    #         return True
    #     return False
        
    #----------------------------------------------------------------------------------------------
    def end_time(self):                                                    # RFT_file
    #----------------------------------------------------------------------------------------------
        # Return data from last check if it exists
        # if data := self.check.data():
        #     return data[-1]
        # Return time-value from tail of file
        return next(self.read('time', tail=True), None) or 0
        #return time
        #return (data := self.check.data()) and data[-1] or 0

    #----------------------------------------------------------------------------------------------
    def time_slice(self):                                                  # RFT_file
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
    def sections_matching_time(self, days, acc=1e-5):                      # RFT_file
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
    def wellstart(self, *wellnames):                                       # RFT_file
    #----------------------------------------------------------------------------------------------
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
    def wellbbox(self, *wellnames, zerobase=True):                          # RFT_file
    #----------------------------------------------------------------------------------------------
        wpos = self.wellpos(*wellnames, zerobase=zerobase)
        bbox = {well:None for well in wellnames}
        for well, wp in zip(wellnames, wpos):
            bbox[well] = [(p[0], p[-1]) for p in map(sorted, zip(*wp))]
        return tuple(bbox.values())

    #----------------------------------------------------------------------------------------------
    def wellpos(self, *wellnames, zerobase=True):                          # RFT_file
    #----------------------------------------------------------------------------------------------
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
    def grid2wellname(self, dim, *wellnames):                              # RFT_file
    #----------------------------------------------------------------------------------------------
        poswell = {pos:[] for pos in product(*(range(d) for d in dim))}
        for well, pos in zip(wellnames, self.wellpos(*wellnames)):
            for p in pos:
                poswell[p].append(well)
        return poswell
        
    #----------------------------------------------------------------------------------------------
    def active_wells(self):                                                # RFT_file
    #----------------------------------------------------------------------------------------------
        wells = []
        current_time = next(self.read('time'))
        for time, well in self.read('time', 'wellname'):
            if time > current_time:
                yield current_time, wells
                wells = []
            wells.append(well)
            current_time = time
            

#==================================================================================================
class UNSMRY_file(unfmt_file):
#==================================================================================================
    """
    Unformatted unified summary file
    --------------------------------
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
    def __init__(self, file):                                           # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        super().__init__(file, suffix='.UNSMRY')
        self.spec = SMSPEC_file(file)
        self.startdate = None
        self._plots = None

    #----------------------------------------------------------------------------------------------
    def params(self, *keys):                                            # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        keypos = {key:i for i, key in enumerate(next(self.spec.blockdata('KEYWORDS')))}
        ind = [keypos[key] for key in keys]
        for param in self.blockdata('PARAMS'):
            yield [param[i] for i in ind]

    #----------------------------------------------------------------------------------------------
    def steptype(self):                                          # UNSMRY_file
    #----------------------------------------------------------------------------------------------
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
    def welldata(self, keys=(), wells=(), only_new=False, as_array=False, 
                 named=False, start=0, stop=None, step=1, **kwargs):    # UNSMRY_file
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
             args=None, **kwargs):                                      # UNSMRY_file
    #----------------------------------------------------------------------------------------------
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
    def key_units(self):                                                # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        Var = namedtuple('Var','unit measure')
        # If 'measures' is None, just use empty string
        kum = zip(*attrgetter('keys', 'units')(self.spec), self.spec.measures or repeat(''))
        var = {k:Var(u, m.split(':')[-1].replace('_',' ')) for k,u,m in set(kum)}
        return namedtuple('Keys', self.keys)(**var)

    #----------------------------------------------------------------------------------------------
    def __getattr__(self, item):                                        # UNSMRY_file
    #----------------------------------------------------------------------------------------------
        try:
            # Look for attribute in File-class first
            return super().__getattr__(item)
        except AttributeError:
            return tuple(set(getattr(self.spec, item)))

    #----------------------------------------------------------------------------------------------
    def energy(self, *wells):                                           # UNSMRY_file
    #----------------------------------------------------------------------------------------------
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
class SMSPEC_file(unfmt_file):                                          # SMSPEC_file
#==================================================================================================
    start = 'INTEHEAD'

    #----------------------------------------------------------------------------------------------
    def __init__(self, file):                                           # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        super().__init__(file, suffix='.SMSPEC')
        self._inkeys = ()
        self._ind = ()
        self.measures = ()
        self.wells = ()
        self.data = ()
        #self.wellkey = None

    #----------------------------------------------------------------------------------------------
    def list_keys(self):                                                # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        keys = next(self.blockdata('KEYWORDS')) if self.is_file() else nparray()
        return sorted(set(keys.tolist()))

    #----------------------------------------------------------------------------------------------
    def welldata(self, keys=(), wells=(), named=False):                 # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        # print('SMSPEC WELLDATA', self.is_file())
        self._inkeys = keys
        if not self.is_file():
            return False
        Data = namedtuple('Data','keys wells measures units', defaults=4*(None,))
        # Do not use mmap here because the SMSPEC-file might 
        # get truncated while mmap'ed which will cause a bus-error
        blockdata = next(self.blockdata('KEYWORDS', '*NAMES', 'MEASRMNT', 'UNITS', use_mmap=False), None)
        self.data = Data(*(bd.tolist() for bd in blockdata)) if blockdata else Data()
        #self.data = Data(*next(self.blockdata('KEYWORDS', '*NAMES', 'MEASRMNT', 'UNITS', use_mmap=False), ()))
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
                    width = len(self.data.measures)//max(len(self.data.keys), 1)
                    measure_strings = map(''.join, grouper(self.data.measures, width))
                    self.measures = getter(tuple(measure_strings))
                if named:
                    self.wells = getter(tuple(w.replace('-','_') for w in self.data.wells))
                else:
                    self.wells = getter(tuple(self.data.wells))
                if not isinstance(self.wells, (list, tuple)):
                    self.wells = (self.wells,)
                return True
        return False

    #----------------------------------------------------------------------------------------------
    def startdate(self):                                                # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        if (start := next(self.blockdata('STARTDAT'), None)) is not None:
            day, month, year, hour, minute, second = start
            return datetime(year, month, day, hour, minute, second)

    #----------------------------------------------------------------------------------------------
    def __getattr__(self, item):                                        # SMSPEC_file
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
    def check_missing_keys(self):                                             # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        return [a for a in self._inkeys if not a in self.keys]

    #----------------------------------------------------------------------------------------------
    def well_pos(self):                                                 # SMSPEC_file
    #----------------------------------------------------------------------------------------------
        return self._ind


#==================================================================================================
class text_file(File):                                                    # text_file
#==================================================================================================
    #----------------------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                   # text_file
    #----------------------------------------------------------------------------------------------
        super().__init__(file, **kwargs)
        self._pattern = {} 
        self._convert = {}
        # self._flavor = None          # 'ecl' or 'ix'

    #----------------------------------------------------------------------------------------------
    def __contains__(self, key):                                          # text_file
    #----------------------------------------------------------------------------------------------
        return key.encode() in self.binarydata()
        
    #----------------------------------------------------------------------------------------------
    def contains_any(self, *keys, head=None, tail=None):                  # text_file
    #----------------------------------------------------------------------------------------------
        if head:
            data = self.head(size=head)
        elif tail:
            data = self.tail(size=tail)
        else:
            data = self.binarydata()
            keys = (key.encode() for key in keys)
        return any(key in data for key in keys)

    #----------------------------------------------------------------------------------------------
    def read(self, *var_list):                                            # text_file
    #----------------------------------------------------------------------------------------------
        #if not self.is_ready():
        #    return ()
        values = []
        #pattern = self._pattern[self._flavor] if self._flavor else self._pattern
        for var in var_list:
            match = matches(file=self.path, pattern=self._pattern[var])
            values.append([self._convert[var](m.group(1)) for m in match])
        return list(zip(*values))



#==================================================================================================
class MSG_file(text_file):
#==================================================================================================
    #----------------------------------------------------------------------------------------------
    def __init__(self, file):
    #----------------------------------------------------------------------------------------------
        super().__init__(file, suffix='.MSG')
        #'time' : r'<\s*\bmessage\b\s+\bdate\b="[0-9/]+"\s+time="([0-9.]+)"\s*>',
        self._pattern = {'date' : r'<message date="([0-9/]+)"',
                         'time' : r'<message date="[0-9/]+" time="([0-9.]+)"',
                         'step' : r'\bRESTART\b\s+\bFILE\b\s+\bWRITTEN\b\s+\bREPORT\b\s+([0-9]+)'}
        self._convert = {'date' : lambda x: datetime.strptime(x.decode(),'%d/%m/%Y'),
                         'time' : float,
                         'step' : int}



#==================================================================================================
class PRT_file(text_file):                                                # PRT_file
#==================================================================================================
    #----------------------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                    # PRT_file
    #----------------------------------------------------------------------------------------------
        super().__init__(file, suffix='.PRT', **kwargs)
        self._pattern['time'] = r'TIME(?:[ a-zA-Z\s/%-]+;|=) +([\d.]+)'
        #self._pattern['time'] = r' (?:Rep    ;|Init   ;|TIME=)\s*([0-9.]+)\s+'
        self._pattern['days'] = self._pattern['time']
        self._convert = {key:float for key in self._pattern}

    #----------------------------------------------------------------------------------------------
    def end_time(self):                                                    # PRT_file
    #----------------------------------------------------------------------------------------------
        chunks = (txt for txt in self.reversed(size=10*1024) if 'TIME' in txt)
        if data:=next(chunks, None):
            days = findall(self._pattern['time'], data)
            return float(days[-1]) if days else 0
        return 0


#==================================================================================================
class PRTX_file(text_file):                                                # PRTX_file
#==================================================================================================
    #----------------------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                    # PRTX_file
    #----------------------------------------------------------------------------------------------
        super().__init__(file, suffix='.PRTX', **kwargs)
        self._var_index = {}

    #----------------------------------------------------------------------------------------------
    def var_index(self):                                                   # PRTX_file
    #----------------------------------------------------------------------------------------------
        if not self._var_index:
            names = next(self.lines(), '').split(',')
            self._var_index = {name:i for i,name in enumerate(names)}
        return self._var_index

    #----------------------------------------------------------------------------------------------
    def end_time(self):                                                   # PRTX_file
    #----------------------------------------------------------------------------------------------
        """
        Note that time in PRTX seems to be delayed compared to PRT and RFT
        """
        time = 0
        if (line:=self.last_line()) and (index:=self.var_index()):
            time = line.split(',')[index['Simulation Time']]
            time = float(time) if time[0] != 'S' else 0
        return time



#==================================================================================================
class fmt_block:                                                         # fmt_block
    #
    # Block of formatted Eclipse data
    #
#==================================================================================================
    #----------------------------------------------------------------------------------------------
    def __init__(self, key=None, length=None, datatype=None, data=(), filemap:mmap=None, start=0, size=0): # fmt_block
    #----------------------------------------------------------------------------------------------
        self._key = key
        self._length = length
        self._dtype = DTYPE[datatype]
        self.data = data
        self.filemap = filemap
        self.startpos = start
        self.size = size
        self.endpos = start + size

    #----------------------------------------------------------------------------------------------
    def __str__(self):                                                    # fmt_block
    #----------------------------------------------------------------------------------------------
        return (f'key={self.key():8s}, type={self._dtype.name:4s},' 
                f'length={self._length:8d}, start={self.startpos:8d}, end={self.endpos:8d}')

    #--------------------------------------------------------------------------------                                                            
    def __repr__(self):                                                   # fmt_block                                                           
    #--------------------------------------------------------------------------------                                                            
        return f'<{type(self)}, key={self.key():8s}, type={self._dtype.name}, length={self._length:8d}>'

    #----------------------------------------------------------------------------------------------
    def __contains__(self, key:str):                                      # fmt_block
    #----------------------------------------------------------------------------------------------
        return self.key() == key

    #----------------------------------------------------------------------------------------------
    def is_last(self):                                                    # fmt_block
    #----------------------------------------------------------------------------------------------
        return self.endpos == self.filemap.size()

    #----------------------------------------------------------------------------------------------
    def formatted(self):                                                  # fmt_block
    #----------------------------------------------------------------------------------------------
        return self.filemap[self.startpos:self.endpos]

    #----------------------------------------------------------------------------------------------
    def key(self):                                                        # fmt_block
    #----------------------------------------------------------------------------------------------
        return self._key.decode().strip()
        #return self.keyword.strip()
    
    # #--------------------------------------------------------------------------------
    # def position(self):                                                   # fmt_block
    # #--------------------------------------------------------------------------------
    #     return (self.startpos, self.endpos)

    #----------------------------------------------------------------------------------------------
    def as_binary(self):                                                  # fmt_block
    #----------------------------------------------------------------------------------------------
        dtype = self._dtype
        count = self._length//dtype.max
        rest = self._length%dtype.max
        pack_fmt = ENDIAN + 'i8si4si' + ''.join(repeat(f'i{dtype.max}{dtype.unpack}i', count))
        if rest:
            pack_fmt += f'i{rest}{dtype.unpack}i'
        size = dtype.size
        head_values = (16, self._key, self._length, dtype.name.encode(), 16)
        data_values = ((size*len(d), *d, size*len(d)) for d in batched(self.data, dtype.max))
        values = chain((head_values,), data_values)
        return pack(pack_fmt, *flatten(values))
                
    #----------------------------------------------------------------------------------------------
    def print(self):                                                      # fmt_block
    #----------------------------------------------------------------------------------------------
        print(self._key, self._length, self._dtype.name)

        
#==================================================================================================
class fmt_file(File):                                                      # fmt_file
    #
    # Class to handle formatted Eclipse files.
    #
#==================================================================================================
    #----------------------------------------------------------------------------------------------
    def __init__(self, filename, **kwargs):                                # fmt_file
    #----------------------------------------------------------------------------------------------
        super().__init__(filename, **kwargs)
        self.start = None 

    #----------------------------------------------------------------------------------------------
    def blocks(self):                                                      # fmt_file
    #----------------------------------------------------------------------------------------------
        def double(string):
            return float(string.replace(b'D',b'E'))
        def logi(string):
            return True if string=='T' else False
        wordsize = {b'INTE':12,  b'LOGI':3,    b'DOUB':23,     b'REAL':17}
        rows     = {b'INTE':6,   b'LOGI':20,   b'DOUB':3,      b'REAL':4}
        cast     = {b'INTE':int, b'LOGI':logi, b'DOUB':double, b'REAL':float}
        head_size = 32
        with self.mmap() as filemap:
            pos = 0
            while pos < filemap.size():
                head = filemap[pos:pos+head_size]
                key, length, typ = head[2:10], int(head[12:23]), head[25:29]
                pos += head_size
                # Here -(-a//b) is used as the ceil function
                size = length*wordsize[typ] + 2*(-(-length//rows[typ])) 
                data = (cast[typ](d) for d in filemap[pos:pos+size].split())
                yield fmt_block(key, length, typ, data, filemap, pos-head_size, size+head_size)
                pos += size

    #----------------------------------------------------------------------------------------------
    def first_section(self):                                               # fmt_file
    #----------------------------------------------------------------------------------------------
        # Get number of blocks and size of first section
        secs = ((i,b) for i, b in enumerate(self.blocks()) if 'SEQNUM' in b)
        count, first_block_next_section = tuple(islice(secs, 2))[-1]
        return namedtuple('section','count size')(count, first_block_next_section.startpos)

    #----------------------------------------------------------------------------------------------
    def section_blocks(self, count=None, with_attr:str=None):              # fmt_file
    #----------------------------------------------------------------------------------------------
        count = count or self.section_count()
        if with_attr:
            return batched((getattr(b, with_attr)() for b in self.blocks()), count)    
        return batched(self.blocks(), count)

    #----------------------------------------------------------------------------------------------
    def as_binary(self, outfile, stop:int=None, buffer=100, rename=(),
                  progress=lambda x:None, cancel=lambda:None):             # fmt_file
    #----------------------------------------------------------------------------------------------
        buffer *= 1024**3
        section = self.first_section()
        N = self.size()/section.size
        if N-int(N) != 0:
            raise SystemError(f'ERROR Uneven section size for {self}')
        progress(-int(N))
        n = 0
        m = 0 # counter for resize
        resized = False
        progress(n)
        with open(outfile, 'wb') as out:
            sectiondata = self.section_blocks(count=section.count, with_attr='as_binary')
            while data:=next(sectiondata, None):
                data = b''.join(data)
                if rename and (names:=[rn for rn in rename if rn[0] in data]):
                    for old, new in names:
                        data = data.replace(old.ljust(8), new.ljust(8))
                out.write(data)
                n += 1
                m += 1
                if stop and n >= stop:
                    return Path(outfile)
                progress(n)
                cancel()
                # Resize file by removing data already processed
                # NB! This is a slow operation for large files
                if (end:=m*section.size) > buffer:
                    resized = True
                    m = 0
                    self.resize(start=0, end=end)
                    sectiondata = self.section_blocks(count=section.count, with_attr='as_binary')
        # Delete the rest of the file if it has been resized
        if resized:
            self.delete()
        return Path(outfile)


#==================================================================================================
class FUNRST_file(fmt_file):
#==================================================================================================
    #----------------------------------------------------------------------------
    def __init__(self, filename):                           # FUNRST_file
    #----------------------------------------------------------------------------
        super().__init__(filename, suffix='.FUNRST')
        self.start = 'SEQNUM'

    #----------------------------------------------------------------------------------------------
    def data(self, *keys):                                       # FUNRST_file
    #----------------------------------------------------------------------------------------------
        data = {}
        for block in self.blocks():
            if block.key() == 'SEQNUM':
                if data:
                    yield data
                data = {}
                data['SEQNUM'] = block.data[0]
            if block.key() == 'INTEHEAD':
                data['DATE'] = tuple(block.data[64:67]) #data[206:208], data[410] 
            for key in keys:
                if block.key() == key:
                    data[key] = (block.data.min(), block.data.max())

    #----------------------------------------------------------------------------
    def as_unrst(self, outfile=None, **kwargs):  # FUNRST_file 
    #----------------------------------------------------------------------------
        outfile = Path(outfile) if outfile else self.path
        outfile = outfile.with_suffix('.UNRST')
        return UNRST_file( super().as_binary(outfile, **kwargs) )



#==================================================================================================
class RSM_block:                                                          # RSM_block
#==================================================================================================
    #----------------------------------------------------------------------------------------------
    def __init__(self, var, unit, well, data):                            # RSM_block
    #----------------------------------------------------------------------------------------------
        self.var = var
        self.unit = unit
        self.well = well
        self.data = data
        self.nrow = len(self.data)
        
    #----------------------------------------------------------------------------------------------
    def get_data(self):                                                   # RSM_block
    #----------------------------------------------------------------------------------------------
        for col,(v,u,w) in enumerate(zip(self.var, self.unit, self.well)):
            yield (v, u, w, [self.data[row][col] for row in range(self.nrow)])
        
        
#==================================================================================================
class RSM_file(File):                                                      # RSM_file
#==================================================================================================
    #----------------------------------------------------------------------------------------------
    def __init__(self, filename, **kwargs):
    #----------------------------------------------------------------------------------------------
        #self.file = Path(filename)
        super().__init__(filename, **kwargs)
        self.fh = None
        self.tag = '1'
        self.nrow = self.block_length()-10
        self.colpos = None
        
    #----------------------------------------------------------------------------------------------
    def get_data(self):                                                    # RSM_file
    #----------------------------------------------------------------------------------------------
        if not self.path.is_file():
            return ()
        with open(self.path, 'r', encoding='utf-8') as self.fh:
            for line in self.fh:
                # line is now at the tag-line
                for block in self.read_block():
                    for data in block.get_data():
                        yield data
                            
    #----------------------------------------------------------------------------------------------
    def read_block(self):                                                  # RSM_file
    #----------------------------------------------------------------------------------------------
        self.skip_lines(3)
        var, unit, well = self.read_var_unit_well()
        self.skip_lines(2)
        data = self.read_data(ncol=len(var))
        yield RSM_block(var, unit, well, data)
                
    #----------------------------------------------------------------------------------------------
    def skip_lines(self, n):                                               # RSM_file
    #----------------------------------------------------------------------------------------------
        next(islice(self.fh, n, n), None)
        # for i in range(n):
        #     next(self.fh)
        
    #----------------------------------------------------------------------------------------------
    def read_data(self, ncol=None):                                        # RSM_file
    #----------------------------------------------------------------------------------------------
        data = [None]*self.nrow        
        for l in range(self.nrow):
            line = next(self.fh)
            cols = line.rstrip().split()
            if len(cols)<ncol:
                # missing column entries
                cols = self.get_columns_by_position(line=line)
            data[l] = [float_or_str(c) for c in cols]
        return data

    
    #----------------------------------------------------------------------------------------------
    def get_columns_by_position(self, line=None):                          # RSM_file
    #----------------------------------------------------------------------------------------------
        """
        Return None if column is empty
        """
        n = len(self.colpos)
        words = [None]*n
        for i in range(n-1):
            a, b = self.colpos[i], self.colpos[i+1]
            words[i] = line[a:b].strip() or None
        return words

    
    #----------------------------------------------------------------------------------------------
    def read_var_unit_well(self):                                          # RSM_file
    #----------------------------------------------------------------------------------------------
        line = next(self.fh)
        var = line.split()
        start = 0
        self.colpos = []
        for v in var:
            i = line.index(v,start)
            self.colpos.append(i)
            start = i+len(var)
        self.colpos.append(len(line))
        unit = self.get_columns_by_position(line=next(self.fh))
        well = self.get_columns_by_position(line=next(self.fh))
        return var, unit, well

    #----------------------------------------------------------------------------------------------
    def block_length(self):                                                # RSM_file
    #----------------------------------------------------------------------------------------------
        with open(self.path, 'r', encoding='utf-8') as fh:
            nb, n = 0, 0
            for line in fh:
                n += 1
                if line[0]==self.tag:
                    nb += 1
                    if nb==2:
                        return int(n)



