"""Input file handlers."""

from collections import defaultdict, namedtuple
from datetime import datetime, timedelta
from itertools import accumulate, chain, repeat
from operator import itemgetter
from pathlib import Path
from platform import system
from re import MULTILINE, finditer, search as re_search

from ...core import File, Restart
from ..output.unrst_file import UNRST_file
from ...utils import decode, flat_list, grouper, list2text, remove_chars, split_by_words

__all__ = ["DATA_file", "ECL_input"]

#==================================================================================================
class DATA_file(File):
#==================================================================================================
    """Loader for Eclipse DATA files."""
    well_control_kw = ('WCONPROD', 'WCONHIST', 'WCONINJE', 'WCONINJH')
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
    def __init__(self, file, suffix=None, check=False, sections=True, **kwargs):        # DATA_file
    #----------------------------------------------------------------------------------------------
        """Initialize the DATA_file.

        Args:
            file: Path to the input deck.
            suffix: Optional suffix enforced on the resolved path.
            check: Whether to validate the file immediately.
            sections: Whether to populate :attr:`section_names`.
            **kwargs: Extra arguments forwarded to :class:`~ECLlib.core.file.File`.
        """
        #print(f'Input_file({file}, check={check}, read={read}, reread={reread}, include={include})')
        suffix = Path(file).suffix or suffix or '.DATA'
        super().__init__(file, suffix=suffix, role='Eclipse input-file', **kwargs)
        self.data = None
        self._checked = False
        self._added_files = ()
        self._wells_by_type = None
        self._section_cache = {}
        self._section_pos = None
        if not sections:
            self.section_names = ()
        getter = namedtuple('getter', 'section default convert pattern')
        self._getter = {
            'DIMENS'  : getter('RUNSPEC',  (),      self._convert_int,
                               r'\bDIMENS\b\s+((?:\d+\s+){2}\d+)\s*/\s*'),
            'SPECGRID': getter('GRID',     (),      self._convert_int,
                               r'\bSPECGRID\b\s+((?:\d+\s+){2}\d+)\b[\s\S]*?/\s*'),
            'TSTEP'   : getter('SCHEDULE', (),      self._convert_float,
                               r'\bTSTEP\b\s+([0-9*.\s]+)/\s*'),
            'START'   : getter('RUNSPEC',  (0,),    self._convert_date,
                               r'\bSTART\b\s+(\d+\s+\'*\w+\'*\s+\d+)'),
            'DATES'   : getter('SCHEDULE', (),      self._convert_date,
                               r'\bDATES\b\s+((\d{1,2}\s+\'*\w{3}\'*\s+\d{4}\s*\s*/\s*)+)/\s*'),
            'RESTART' : getter('SOLUTION', ('', 0), self._convert_file,
                               r"\bRESTART\b\s+('*[\w./\\-]+'*\s+[0-9]+)\s*/"),
            'WELSPECS': getter('SCHEDULE', (),      self._convert_string,
                               r'\bWELSPECS\b((\s+\'*[\w/-]+?.*/\s*)+/)'),
            'WCONPROD': getter('SCHEDULE', (),      self._convert_string,
                               r'\bWCONPROD\b\s+((?:^(?!/).*\n)+)^/\s*$'),
            'WCONHIST': getter('SCHEDULE', (),      self._convert_string,
                               r'\bWCONHIST\b\s+((?:^(?!/).*\n)+)^/\s*$'),
            'WCONINJE': getter('SCHEDULE', (),      self._convert_string,
                               r'\bWCONINJE\b\s+((?:^(?!/).*\n)+)^/\s*$'),
            'WCONINJH': getter('SCHEDULE', (),      self._convert_string,
                               r'\bWCONINJH\b\s+((?:^(?!/).*\n)+)^/\s*$')}
        if check:
            self.check()


    #----------------------------------------------------------------------------------------------
    def __repr__(self):                                                                 # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return a developer-friendly representation."""
        return f'<{type(self)}, {self.path}>'

    #----------------------------------------------------------------------------------------------
    def __call__(self):                                                                 # DATA_file
    #----------------------------------------------------------------------------------------------
        """Invoke the underlying callable."""
        self.data = self.binarydata()
        return self

    #----------------------------------------------------------------------------------------------
    def __contains__(self, key):                                                        # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return whether the value exists.

        Args:
            key: Keyword searched for in the DATA file.
        """
        self.data = None
        return bool(self.search(key, regex=rf'^[ \t]*{key}', comments=True))
        
    #----------------------------------------------------------------------------------------------
    def mode(self):                                                                     # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return the mode string."""
        return 'backward' if ('READDATA' in self) else 'forward'


    #----------------------------------------------------------------------------------------------
    def restart(self):                                                                  # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return restart metadata for the block."""
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
    def binarydata(self, raise_error=False):                                            # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return the file contents as bytes.

        Args:
            raise_error: Whether to raise if the file is missing.
        """
        self.data = super().binarydata(raise_error)
        end = b'END' in self.data and re_search(rb'^[ \t]*\bEND\b', self.data, flags=MULTILINE)
        return end and self.data[:end.end()] or self.data

    #----------------------------------------------------------------------------------------------
    def check(self, include=True, uppercase=False):                          # DATA_file
    #----------------------------------------------------------------------------------------------
        """Check the file for inconsistencies.

        Args:
            include: Whether to verify that included files exist.
            uppercase: Whether to enforce uppercase filenames on Linux.
        """
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
        """Search for matching blocks.

        Args:
            key: Keyword to pre-filter sections.
            regex: Regular expression used to extract values.
            comments: Whether to keep comments in the search data.
        """
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
        """Return full path of INCLUDE files as a generator.

        Args:
            data: Optional binary blob to inspect instead of reading from disk.
        """
        return (f[0] for f in self._included_file_data(data))

    #----------------------------------------------------------------------------------------------
    def _included_file_data(self, data:bytes=None):                           # DATA_file
    #----------------------------------------------------------------------------------------------
        """Yield tuples of filename and binary data for each include file.

        Args:
            data: Optional binary blob to inspect instead of reading from disk.
        """
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
        """Return the grid related filenames.

        Args:
            data: Optional binary blob to inspect instead of reading from disk.
        """
        data = data or self.binarydata()
        regex = rb"^\s*(?:\bGDFILE\b)(?:\s*--.*\s*|\s*)*'*([^' ]+)['\s]*/.*$"
        files = (self.with_name(m.group(1).decode()) for m in finditer(regex, data, flags=MULTILINE))
        return (file.with_suffix(file.suffix or '.EGRID') for file in files)

    #----------------------------------------------------------------------------------------------
    def including(self, *files):                                          # DATA_file
    #----------------------------------------------------------------------------------------------
        """Add the given files and return self.

        Args:
            *files: Iterable of filenames to treat as additional includes.
        """
        # Added files must be an iterator to avoid an infinite recursive
        # loop when self._added_files is called in _included_file_data
        self._added_files = iter(files)
        self._clear_cache()
        # Disable check to avoid check to consume the above iterator
        self._checked = True
        return self

    #----------------------------------------------------------------------------------------------
    def start(self):                                                      # DATA_file
    #----------------------------------------------------------------------------------------------
        """Start progress reporting."""
        return self.get('START')[0]

    #----------------------------------------------------------------------------------------------
    def dim(self):                                                        # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return the grid dimensions."""
        dim = self.get('DIMENS') or self.get('SPECGRID')
        return tuple(dim[:3]) if dim else None

    #----------------------------------------------------------------------------------------------
    def wells_by_type(self):                                              # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return wells grouped as producers or injectors."""
        if self._wells_by_type is not None:
            return {well_type:list(names) for well_type, names in self._wells_by_type.items()}
        grouped = defaultdict(list)
        well_types = {}
        records = self.get(*self.well_control_kw)
        if not any(records) and (sch_file := self.with_suffix('.SCH', ignore_case=True, exists=True)):
            records = DATA_file(sch_file, sections=False).get(*self.well_control_kw)
        for keyword, control_records in zip(self.well_control_kw, records):
            well_type = 'PRODUCER' if keyword in ('WCONPROD', 'WCONHIST') else 'INJECTOR'
            for record in control_records:
                wellname = self._first_record_field(record)
                if wellname.startswith('*') or wellname.endswith('*'):
                    raise SystemError(f'ERROR Well templates/lists are not supported in {self}')
                well_types[wellname] = well_type
        for wellname, well_type in well_types.items():
            grouped[well_type].append(wellname)
        self._wells_by_type = dict(grouped)
        return {well_type:list(names) for well_type, names in self._wells_by_type.items()}

    #----------------------------------------------------------------------------------------------
    def timesteps(self, start=None, negative_ok=False, missing_ok=False, pos=False, skiprest=False):     # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return timestep information for the schedule.

        Args:
            start: Optional start date overriding the deck value.
            negative_ok: Whether negative or zero durations are allowed.
            missing_ok: Whether missing timestep data should pass silently.
            pos: Whether to return positions alongside values.
            skiprest: Whether to ignore restart-induced negative timesteps.
        """
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
        """Return formatted report dates."""
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
    def welspecs(self):                                                                 # DATA_file
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
        return tuple(set(self._first_record_field(record) for record in welspecs if record))

    #----------------------------------------------------------------------------------------------
    def get(self, *keywords, raise_error=False, pos=False):                             # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return the requested value.

        Args:
            *keywords: Keyword names to extract from the DATA file.
            raise_error: Whether to raise if a keyword is missing.
            pos: Whether to return positions in addition to values.
        """
        keywords = [key.upper() for key in keywords]
        getters = [self._getter.get(key) for key in keywords]
        failed = [g.default for g in getters]
        failed = failed[0] if len(failed) == 1 else failed
        if not self.exists(raise_error=raise_error):
            return failed
        if missing:=[k for g,k in zip(getters, keywords) if not g]:
            if raise_error:
                raise SystemError(f'ERROR Missing get-pattern for {list2text(missing)} in DATA_file')
            return failed
        names = tuple(g.section for g in getters)
        self.data = self._section_text(*names, raise_error=raise_error)
        if not self.data:
            error_msg = f'ERROR Keyword {list2text(keywords)} not found in {self.path}'
            if raise_error:
                raise SystemError(error_msg)
            return failed
        result = ()
        for keyword, getter in zip(keywords, getters):
            val_span = tuple((m.group(1), m.span()) for m in finditer(getter.pattern, self.data,
                                                                       flags=MULTILINE))
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
    def get_old(self, *keywords, raise_error=False, pos=False):                         # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return keyword data using the original uncached implementation."""
        keywords = [key.upper() for key in keywords]
        getters = [self._getter.get(key) for key in keywords]
        failed = [g.default for g in getters]
        failed = failed[0] if len(failed) == 1 else failed
        if not self.exists(raise_error=raise_error):
            return failed
        if missing:=[k for g,k in zip(getters, keywords) if not g]:
            if raise_error:
                raise SystemError(f'ERROR Missing get-pattern for {list2text(missing)} in DATA_file')
            return failed
        names = set(g.section for g in getters)
        self.data = self._remove_comments(self.section(*names, raise_error=raise_error)._matching(*keywords))
        error_msg = f'ERROR Keyword {list2text(keywords)} not found in {self.path}'
        if not self.data:
            if raise_error:
                raise SystemError(error_msg)
            return failed
        result = ()
        for keyword, getter in zip(keywords, getters):
            val_span = tuple((m.group(1), m.span()) for m in finditer(getter.pattern, self.data,
                                                                       flags=MULTILINE))
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
    def lines(self):                                                                    # DATA_file
    #----------------------------------------------------------------------------------------------
        """Iterate over the file lines."""
        return (line for line in self._remove_comments(self._matching()).split('\n') if line)

    #----------------------------------------------------------------------------------------------
    def text(self):                                                                     # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return textual data for the block."""
        return self._remove_comments(self._matching())

    #----------------------------------------------------------------------------------------------
    def summary_keys(self, matching=()):                                                # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return metadata for summary keys."""
        return [k for k in self.section('SUMMARY').text().split() if k in matching]

    #----------------------------------------------------------------------------------------------
    def section_positions(self, *sections):                                             # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return start offsets for each section."""
        if self._section_pos is None:
            data = self.binarydata()
            self._section_pos = {sec.upper().decode():(a,b) for sec,a,b in split_by_words(data, self.section_names)}
        if not sections:
            return dict(self._section_pos)
        return {sec:pos for sec in sections if (pos := self._section_pos.get(sec))}

    #----------------------------------------------------------------------------------------------
    def section(self, *sections, raise_error=True):                                     # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return a named section."""
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
    def replace_keyword(self, keyword, new_string):                                     # DATA_file
    #----------------------------------------------------------------------------------------------
        """Replace a keyword within the data.

        Args:
            keyword: Keyword to search for in the deck.
            new_string: Replacement text including trailing slash.
        """
        ### Get keyword value and position in file
        match = self.get(keyword, pos=True)
        if match:
            _, pos = match[0] # Get first match
        else:
            raise SystemError(f'ERROR Missing {keyword} in {self}')
        out = self.data[:pos[0]] + new_string + self.data[pos[1]:]
        with open(self.path, 'w', encoding='utf-8') as f:
            f.write(out)
        self._clear_cache()

    #----------------------------------------------------------------------------------------------
    def _remove_comments(self, data=None):                                              # DATA_file
    #----------------------------------------------------------------------------------------------
        """Strip comment lines from the file.

        Args:
            data: Optional iterable of binary blobs to process.
        """
        data = data or (self.binarydata(),)
        lines = (l for d in data for l in d.split(b'\n'))
        text = (l.split(b'--')[0].strip() for l in lines)
        text = b'\n'.join(t for t in text if t)
        text = decode(text)
        return text+'\n' if text else ''

    #----------------------------------------------------------------------------------------------
    def _remove_comments_bytes(self, data=None):                                        # DATA_file
    #----------------------------------------------------------------------------------------------
        """Strip comment lines from the file and keep the result as bytes.

        Args:
            data: Optional iterable of binary blobs to process.
        """
        if data is None:
            data = (self.binarydata(),)
        elif isinstance(data, (bytes, bytearray)):
            data = (data,)
        lines = (line for chunk in data for line in bytes(chunk).split(b'\n'))
        text = (line.split(b'--')[0].strip() for line in lines)
        return b'\n'.join(line for line in text if line)

    #----------------------------------------------------------------------------------------------
    def _matching(self, *keys):                                                         # DATA_file
    #----------------------------------------------------------------------------------------------
        """Yield matches for the provided patterns.

        Args:
            *keys: Keyword names used to filter the DATA file.
        """
        #print('_matching', keys)
        self.data = self.data if isinstance(self.data, (bytes, bytearray)) else self.binarydata()
        keys = [key.encode() for key in keys]
        if keys == [] or any(key in self.data for key in keys):
            yield self.data
        for file, data in self._included_file_data(self.data):
            if keys == [] or any(key in data for key in keys):
                yield data

    #----------------------------------------------------------------------------------------------
    def _clear_cache(self):                                                             # DATA_file
    #----------------------------------------------------------------------------------------------
        """Reset cached text, section metadata, and derived well classifications."""
        self._section_cache = {}
        self._section_pos = None
        self._wells_by_type = None

    #----------------------------------------------------------------------------------------------
    def _section_text(self, *sections, raise_error=True):                               # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return cleaned text for the requested sections and their includes."""
        if not self._checked:
            self.check()
        key = tuple(sorted(set(sections)))
        if key in self._section_cache:
            return self._section_cache[key]
        if not self.section_names:
            data = self.binarydata()
        else:
            sec_pos = self.section_positions(*key)
            if not sec_pos:
                if raise_error:
                    raise SystemError(f'ERROR Section {list2text(key)} not found in {self}')
                return ''
            data = b''.join(self.binarydata()[a:b] for a,b in sorted(sec_pos.values()))
        text = self._remove_comments(chain((data,), (chunk for _, chunk in self._included_file_data(data))))
        self._section_cache[key] = text
        return text

    #----------------------------------------------------------------------------------------------
    def _first_record_field(self, record):                                              # DATA_file
    #----------------------------------------------------------------------------------------------
        """Return the first field from an Eclipse record, preserving quoted names."""
        record = bytes(record).strip() if not isinstance(record, str) else record.encode().strip()
        if not record:
            return ''
        if record[0] in (ord("'"), ord('"')):
            end = record.find(record[:1], 1)
            if end < 0:
                raise SystemError(f'ERROR Unterminated quoted string in {self}')
            return decode(record[1:end])
        return decode(record.split(None, 1)[0].rstrip(b'/'))

    #----------------------------------------------------------------------------------------------
    def _days(self, time_pos, start=None):                                              # DATA_file
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
    def _convert_string(self, values, key):                                             # DATA_file
    #----------------------------------------------------------------------------------------------
        """Convert textual values to strings."""
        ret = [v for val in values for v in val.split('\n') if v and v != '/']
        return (ret,)

    #----------------------------------------------------------------------------------------------
    def _convert_int(self, values, key):                                                # DATA_file
    #----------------------------------------------------------------------------------------------
        """Convert textual values to integers."""
        values = tuple([int(n) for n in v.split()] for v in values)
        return values or self._getter[key].default

    #----------------------------------------------------------------------------------------------
    def _convert_float(self, values, key):                                              # DATA_file
    #----------------------------------------------------------------------------------------------
        """Convert textual values to floats."""
        #mult = lambda x, y : list(repeat(float(y),int(x))) # Process x*y statements
        def mult(x,y):
            """Return the data converted to floats."""
            # Process x*y statements
            return list(repeat(float(y),int(x)))
        values = ([mult(*n.split('*')) if '*' in n else [float(n)] for n in v.split()] for v in values)
        values = tuple(flat_list(v) for v in values)
        return values or self._getter[key].default

    #----------------------------------------------------------------------------------------------
    def _convert_date(self, dates, key):                                                # DATA_file
    #----------------------------------------------------------------------------------------------
        """Convert raw date values to datetime objects."""
        ### Remove possible quotes
        ### Extract groups of 3 from the dates strings 
        dates = (grouper(remove_chars("'/\n", v).split(), 3) for v in dates)
        dates = tuple([datetime.strptime(' '.join(d), '%d %b %Y') for d in date] for date in dates)
        return dates or self._getter[key].default

    #----------------------------------------------------------------------------------------------
    def _convert_file(self, values, key):                                               # DATA_file
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
class ECL_input:                                                                        # ECL_input
#==================================================================================================
    """Thin Eclipse-side metadata wrapper around :class:`DATA_file`."""

    #----------------------------------------------------------------------------------------------
    def __init__(self, case, check=False, **kwargs):                                    # ECL_input
    #----------------------------------------------------------------------------------------------
        """Initialize the ECL_input."""
        self.data = DATA_file(case, check=check, **kwargs)
        self.path = self.data.path

    #----------------------------------------------------------------------------------------------
    def __str__(self):                                                                  # ECL_input
    #----------------------------------------------------------------------------------------------
        """Return a human-readable representation."""
        return str(self.data)

    #----------------------------------------------------------------------------------------------
    def __contains__(self, key):                                                       # ECL_input
    #----------------------------------------------------------------------------------------------
        """Return whether the value exists."""
        return key in self.data

    #----------------------------------------------------------------------------------------------
    def check(self, *args, **kwargs):                                                  # ECL_input
    #----------------------------------------------------------------------------------------------
        """Delegate deck validation to :class:`DATA_file`."""
        return self.data.check(*args, **kwargs)

    #----------------------------------------------------------------------------------------------
    def include_files(self):                                                           # ECL_input
    #----------------------------------------------------------------------------------------------
        """Return INCLUDE statements discovered in the deck."""
        return self.data.include_files()

    #----------------------------------------------------------------------------------------------
    def including(self, *files):                                                       # ECL_input
    #----------------------------------------------------------------------------------------------
        """Add extra include files and return ``self``."""
        self.data.including(*files)
        return self

    #----------------------------------------------------------------------------------------------
    def dim(self):                                                                     # ECL_input
    #----------------------------------------------------------------------------------------------
        """Return the grid dimensions."""
        return self.data.dim()

    #----------------------------------------------------------------------------------------------
    def wells_by_type(self):                                                           # ECL_input
    #----------------------------------------------------------------------------------------------
        """Return wells grouped by normalized Eclipse control type."""
        return self.data.wells_by_type()
