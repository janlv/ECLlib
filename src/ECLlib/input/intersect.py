# -*- coding: utf-8 -*-
"""Intersect specific file helpers."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import accumulate, chain, zip_longest
from operator import attrgetter
from pathlib import Path
from re import MULTILINE, findall, finditer, compile as re_compile, search as re_search
from subprocess import Popen, STDOUT
from time import sleep

from numpy import fromstring

from proclib import Process

from ..core import File, Restart
from ..constants import ECL2IX_LOG
#from .eclipse import DATA_file
from ..utils import (
    any_cell_in_box,
    bounding_box,
    date_range,
    flatten,
    list2text,
    pairwise,
    split_in_lines,
)

__all__ = [
    "AFI_file",
    "IXF_node",
    "IXF_file",
    "IX_input",
]

#--------------------------------------------------------------------------------------------------
def Eclipse_input(path):
#--------------------------------------------------------------------------------------------------
    """ Return the Eclipse input file (DATA-file) based on case name """
    # Define Eclipse input file (DATA-file) based on case name to avoid import from .eclipse
    # which would create a circular import 
    return File(path, suffix='.DATA', ignore_suffix_case=True)

#==================================================================================================
class AFI_file(File):                                                      # AFI_file
#==================================================================================================
    #include_regex = rb'^[ \t]*\bINCLUDE\b\s*"*([\w.-]+)"*'
    # Return 'filename' and 'key1=val1 key2=val2' as groups from the following format:
    # INCLUDE "filename" {key1=val1, key2=val2}
    #include_regex = rb'\bINCLUDE\b\s*"*([^\"]+)"*\s*\{([^}]+)\}'
    #include_regex = rb'^\s*\bINCLUDE\b\s*\"*([^\"]+)'
    # Process 'INCLUDE "filename" {metadata}' where metadata is optional
    include_regex = rb'^\s*\bINCLUDE\b\s*"([^\"]+)"(?:\s*\{([^}]*)\})?'

    #----------------------------------------------------------------------------------------------
    def __init__(self, file, check=False, **kwargs):                       # AFI_file
    #----------------------------------------------------------------------------------------------
        super().__init__(file, suffix='.afi', role='Top level Intersect input-file',
                         ignore_suffix_case=True, **kwargs)
        self._data = None
        self.pattern = None
        if check:
            self.exists(raise_error=True)

    #----------------------------------------------------------------------------------------------
    def data(self):                                                        # AFI_file
    #----------------------------------------------------------------------------------------------
        self._data = self._data or self.binarydata()
        return self._data

    #----------------------------------------------------------------------------------------------
    def ixf_files(self):                                                   # AFI_file
    #----------------------------------------------------------------------------------------------
        #self.data = self.data or self.binarydata()
        # self.pattern = self.pattern or re_compile(self.include_regex, flags=MULTILINE)
        #matches_ = findall(self.include_regex, self.data, flags=MULTILINE)
        #files = (Path(m[0].decode()) for m in matches_)
        #return (self.with_name(file) for file in files if file.suffix.lower() == '.ixf')
        #return (self.with_name(file) for file in self.files(self.data) if file.suffix.lower() == '.ixf')
        return (file for file in self.include_files(self.data()) if file.suffix.lower() == '.ixf')

    #----------------------------------------------------------------------------------------------
    def matches(self, data:bytes=None):                                   # AFI_file
    #----------------------------------------------------------------------------------------------
        self.pattern = self.pattern or re_compile(self.include_regex, flags=MULTILINE)
        return self.pattern.finditer(data or self.data())

    #----------------------------------------------------------------------------------------------
    def include_files(self, data:bytes=None):                              # AFI_file
    #----------------------------------------------------------------------------------------------
        return (self.path.with_name(m[1].decode()) for m in self.matches(data or self.data()))
        #data =  #binarydata()
        #self.pattern = self.pattern or re_compile(self.include_regex, flags=MULTILINE)
        #return (self.path.with_name(m.group(1).decode()) for m in self.finditer(data or self.data()))
        # return (f[0] for f in self._included_file_data(self.data))

    #----------------------------------------------------------------------------------------------
    def included_file_data(self, data:bytes=None):                        # AFI_file
    #----------------------------------------------------------------------------------------------
        """ Return tuple of filename and binary-data for each include file """
        # data = data or self.binarydata()
        # self.pattern = self.pattern or re_compile(self.include_regex, flags=MULTILINE)
        # files = (m.group(1).decode() for m in self.pattern.finditer(data))
        # #regex = self.include_regex
        # #files = (m.group(1).decode() for m in re_compile(regex, flags=MULTILINE).finditer(data))
        # Loop over files included in self
        # for file in files:
        for inc_file in self.include_files(data):
            #inc_file = self.with_name(file)
            inc_data = File(inc_file).binarydata()
            yield (inc_file, inc_data)
            # Recursive call for files included deeper down
            if b'INCLUDE' in inc_data:
                yield from self.included_file_data(inc_data)
                # for inc_inc in self._included_file_data(inc_data):
                #     yield inc_inc


#==================================================================================================
@dataclass
class IXF_node:                                                            # IXF_node
#==================================================================================================
    """
    Intersect Input Format (IXF) nodes currently implemented:

    Context node:
        type "name" {
            command()
            variable=value
        }

    Table node:
        type "name" [
            col1_name        col2_name        ... coln_name
            col1_row1_value  col2_row1_value  ... coln_row1_value
            ...              ...              ... ...
            col1_rowm_value  col2_rowm_value  ... coln_rowm_value
        ]

    Arguments:
        type: str, default: ''
            Type of node

        name: any, default: None
            Name of node (not including quotes). Can also hold numbers, datetime, etc.

        content: str, default: ''
            Content of node including the braces which is used to define the
            node as a table- or context-node. The braces are removed during
            init, and self.content returns content without braces.

        pos: tuple of two ints, default: None
            Begin and end position of node in file

        file: str, default = ''
            Name of file holding node
    """
    type : str = ''
    name : any = None
    content : str = ''
    pos : any = None
    file : str = ''

    #----------------------------------------------------------------------------------------------
    def __post_init__(self):                                               # IXF_node
    #----------------------------------------------------------------------------------------------
        self.is_table = False
        self.is_context = False
        self.brace = None
        if self.content:
            self.brace = (self.content[0], self.content[-1])
            self.content = self.content[1:-1]
            if self.brace[0] == '{':
                self.is_context = True
            else:
                self.is_table = True

    #----------------------------------------------------------------------------------------------
    def __str__(self):                                                     # IXF_node
    #----------------------------------------------------------------------------------------------
        return f'{self.type} "{self.name}" {self.brace[0]}{self.content}{self.brace[1]}'

    #----------------------------------------------------------------------------------------------
    def __repr__(self):                                                     # IXF_node
    #----------------------------------------------------------------------------------------------
        return (f'<IXF_node type={self.type}, name={self.name}, '
                f'is_table={self.is_table}, is_context={self.is_context} >')

    #----------------------------------------------------------------------------------------------
    def copy(self):                                                        # IXF_node
    #----------------------------------------------------------------------------------------------
        return IXF_node(self.type, self.name, self.brace[0]+self.content+self.brace[1])

    #----------------------------------------------------------------------------------------------
    def __contains__(self, key):                                           # IXF_node
    #----------------------------------------------------------------------------------------------
        return key in self.content

    #----------------------------------------------------------------------------------------------
    def set_content(self, rows):                                           # IXF_node
    #----------------------------------------------------------------------------------------------
        if self.is_context:
            lines = (f'    {k}{"=" if v else ""}{v}' for k,v in rows)
        else:
            width = [max(map(len, col)) for col in zip(*rows)]
            lines = (''.join(f'    {v:>{w}s}' for v,w in zip(row, width)) for row in rows)
        self.content = '\n' + '\n'.join(lines) + '\n'

    #----------------------------------------------------------------------------------------------
    def lines(self):                                                       # IXF_node
    #----------------------------------------------------------------------------------------------
        return split_in_lines(self.content)

    #----------------------------------------------------------------------------------------------
    def columns(self):                                                     # IXF_node
    #----------------------------------------------------------------------------------------------
        delimiter = '=' if self.is_context else None # None equals any whitespace
        data = (row for line in self.lines() if (row:=line.split(delimiter)))
        # Use repeat('', 2) to get two columns also if data = (('key',),)
        # instead of data = (('key','value'),)
        return tuple(v[:-1] for v in zip_longest(*data, ('', ''), fillvalue=''))

    #----------------------------------------------------------------------------------------------
    def rows(self):                                                        # IXF_node
    #----------------------------------------------------------------------------------------------
        return tuple(zip(*self.columns()))

    #----------------------------------------------------------------------------------------------
    def as_dict(self):                                                     # IXF_node
    #----------------------------------------------------------------------------------------------
        return {k:v for k,*v in self.rows()}

    #----------------------------------------------------------------------------------------------
    def get(self, *items):                                                 # IXF_node
    #----------------------------------------------------------------------------------------------
        #return self.as_dict().get(item)
        values = flatten(self.as_dict().get(item) or [None] for item in items)
        return [val.split('#')[0].strip().replace('"', '') for val in values if val]
        # return list(flatten(self.as_dict().get(item) or [None] for item in items))

    #----------------------------------------------------------------------------------------------
    def update(self, node=None): #, on_top=False):                         # IXF_node
    #----------------------------------------------------------------------------------------------
        adict = self.as_dict()
        ndict = node.as_dict()
        adict.update(ndict)
        # if on_top:
        #     # Top line of node is also the top line of adict
        #     key, val = next(iter(ndict.items()))
        #     adict.pop(key, None)
        #     adict = {key:val, **adict}
        rows = [(k,*v) for k,v in adict.items()]
        self.set_content(rows)


#==================================================================================================
class IXF_file(File):                                                      # IXF_file
#==================================================================================================

    """
    Intersect Input Format (IXF):

    type "name" { (or [)
        content
    } (or ])

    """

    #----------------------------------------------------------------------------------------------
    def __init__(self, file, check=False, **kwargs):                       # IXF_file
    #----------------------------------------------------------------------------------------------
        super().__init__(file, role='Intersect input-file', **kwargs)
        self.data = None
        if check:
            self.exists(raise_error=True)


    #----------------------------------------------------------------------------------------------
    def __contains__(self, key):                                           # IXF_file
    #----------------------------------------------------------------------------------------------
        self.data = self.data or self.binarydata()
        return bool(re_search(rf'^[ \t]*\b{key}\b'.encode(), self.data, flags=MULTILINE))


    #----------------------------------------------------------------------------------------------
    def binarydata(self, raise_error=False):                               # IXF_file
    #----------------------------------------------------------------------------------------------
        self.data = super().binarydata(raise_error)
        end_key = b'END_INPUT'
        if end_key in self.data and not b'#'+end_key in self.data:
            self.data, _ = self.data.split(end_key, maxsplit=1)
        return self.data
        # end = b'END' in self.data and re_search(rb'^[ \t]*\bEND\b', self.data, flags=MULTILINE)
        # return end and self.data[:end.end()] or self.data

    #----------------------------------------------------------------------------------------------
    #def node(self, *nodes, convert=(), brace=(rb'{',rb'}')):               # IXF_file
    def node(self, *nodes, convert=(), table=False):               # IXF_file
    #----------------------------------------------------------------------------------------------
        if table:
            begin, end = b'\\[', b'\\]'
        else:
            # Context node
            begin, end = rb'{', rb'}'
        self.data = self.data or self.binarydata()
        if nodes[0] == 'all':
            keys = rb'\w+'
        else:
            keys = '|'.join(nodes).encode()
        # The pattern is explained here: https://regex101.com/r/4FVBxU/5
        not_brackets = rb'[^' + begin + end + rb']*'
        nested_brackets = begin + not_brackets + end
        pattern = ( rb'^\s*\b(' + keys + rb')\b *[\"\']?([\w .:\\/*-]+)?[\"\']? *(' + begin +
                    rb'(?:' + not_brackets + rb'|' + nested_brackets + rb')+' + end + rb')?' )
        for match in finditer(pattern, self.data, flags=MULTILINE):
            val = [g.decode().strip() if g else g for g in match.groups()]
            if convert:
                ind = nodes.index(val[0])
                val[1] = convert[ind](val[1])
            yield IXF_node(type=val[0], name=val[1], content=val[2], pos=match.span(), file=self.path)


#==================================================================================================
class IX_input:                                                            # IX_input
#==================================================================================================
    STAT_FILE = '.ecl2ix' # Check if ECL-input has changed and new conversion is needed

    @classmethod
    #----------------------------------------------------------------------------------------------
    def datesteps(cls, start, stop, step=1):
    #----------------------------------------------------------------------------------------------
        """
        datesteps((1971, 7, 1), 10, 5) -> DATE "01-Jul-1971"
                                          DATE "06-Jul-1971"
        """
        fmt = '%d-%b-%Y %H:%M:%S'
        dates = (f'DATE "{date}"' for date in date_range(start, stop, step, fmt=fmt))
        print('\n'.join(dates))

    #----------------------------------------------------------------------------------------------
    def __init__(self, case, check=False, **kwargs):                       # IX_input
    #----------------------------------------------------------------------------------------------
        self.afi = AFI_file(case)
        self.path = self.afi.path
        self.ixf_files = [IXF_file(file) for file in self.afi.ixf_files()]
        self._checked = False
        if check:
            self.check()

    #----------------------------------------------------------------------------------------------
    def __str__(self):                                                     # IX_input
    #----------------------------------------------------------------------------------------------
        return f'{self.afi}'

    #----------------------------------------------------------------------------------------------
    def __getattr__(self, item):                                           # IX_input
    #----------------------------------------------------------------------------------------------
        #print('IX_INPUT GETATTR')
        return getattr(self.path, item)

    #----------------------------------------------------------------------------------------------
    def __contains__(self, key):                                           # IX_input
    #----------------------------------------------------------------------------------------------
        return any(key in ixf for ixf in self.ixf_files)

    #----------------------------------------------------------------------------------------------
    def dim(self):                                                         # IX_input
    #----------------------------------------------------------------------------------------------
        """
        Returns the grid dimensions as a list of integers [I, J, K], representing the
        number of cells in each direction. Searches for a 'StructuredInfo' node, extracts
        its dictionary, and retrieves 'NumberCellsInI', 'NumberCellsInJ', and
        'NumberCellsInK' values. Converts these to integers.

        Returns:
            list[int]: Number of cells in I, J, and K directions.

            StopIteration: If 'StructuredInfo' node is not found.
            KeyError: If expected keys are missing.
            ValueError: If values cannot be converted to integers.
        """
        if node := next(self.nodes('StructuredInfo'), None):
            dim_node = node.as_dict()
            return tuple(int(dim_node[f'NumberCellsIn{ijk}'][0]) for ijk in 'IJK')
            #return [int(dim_node[f'NumberCellsIn{ijk}'][0]) for ijk in 'IJK']

    #----------------------------------------------------------------------------------------------
    def ifind(self, astr:str):                                             # IX_input
    #----------------------------------------------------------------------------------------------
        enc_str = astr.encode()
        for file, data in self.afi.included_file_data():
            if enc_str in data:
                yield file

    #----------------------------------------------------------------------------------------------
    def find(self, *args):                                                 # IX_input
    #----------------------------------------------------------------------------------------------
        return next(self.ifind(*args), None)

    #----------------------------------------------------------------------------------------------
    def findall(self, *args):                                              # IX_input
    #----------------------------------------------------------------------------------------------
        return tuple(self.ifind(*args))

    @classmethod
    #----------------------------------------------------------------------------------------------
    def need_convert(self, path):                                          # IX_input
    #----------------------------------------------------------------------------------------------
        path = Path(path)
        afi_file = AFI_file(path)
        eclipse_inp = Eclipse_input(path)
        if afi_file.is_file() and not eclipse_inp.is_file():
            # No need for convert
            return
        if not afi_file.is_file():
            if not eclipse_inp.is_file():
                raise SystemError('ERROR Eclipse input is missing, unable to create Intersect input.')
            return 'Intersect input is missing for this case, but can be created from the Eclipse input.'
        # Check if input is complete
        if any(file for file in afi_file.include_files() if not file.is_file()):
            return 'Intersect input is incomplete for this case (missing include files).'
        # Check if DATA-file has changed since last convert
        stat_file = path.with_suffix(self.STAT_FILE)
        mtime, size = attrgetter('st_mtime_ns', 'st_size')(eclipse_inp.stat())
        if stat_file.is_file():
            old_mtime, old_size = map(int, stat_file.read_text(encoding='utf-8').split())
            if mtime > old_mtime and size > old_size:
                return 'Intersect input exists for this case, but the Eclipse input has changed since the previous convert.'
        else:
            stat_file.write_text(f'{mtime} {size}', encoding='utf-8')
            #stat_file.write_text(f'{data_file.name} {mtime} {size}')


    @classmethod
    #----------------------------------------------------------------------------------------------
    def from_eclipse(self, path, progress=None, abort=None, freq=20):      # IX_input
    #----------------------------------------------------------------------------------------------
        # Create IX input from Eclipse input
        #if not DATA_file(path).is_file():
        eclipse_inp = Eclipse_input(path)
        if not eclipse_inp.is_file():
            raise SystemError('ERROR Eclipse input is missing, convert aborted...')
        path = Path(path)
        cmd = ['eclrun', 'ecl2ix', path]
        #msg = 'Creating Intersect input from Eclipse input'
        # How often to check if convert is completed
        sec = 1/freq
        logfile = path.with_name(ECL2IX_LOG)
        with open(logfile, 'w', encoding='utf-8') as log:
            with Popen(cmd, stdout=log, stderr=STDOUT) as popen:
            #popen = Popen(cmd, stdout=log, stderr=STDOUT)
                proc = Process(pid=popen.pid)
                i = 0
                while (proc.is_running()):
                    if abort and abort():
                        proc.kill(children=True)
                        return False
                    if progress:
                        i += 1
                        progress(i)
                        #dots = ((1+i%5)*'.').ljust(5)
                        #print(f'\r   {msg} {dots}', end='')
                    sleep(sec)
                #print('\r',' '*80, end='')
            if not AFI_file(path).is_file():
                return False, logfile
            # If successful, save modification time and current size of DATA_file
            mtime, size = attrgetter('st_mtime_ns', 'st_size')(eclipse_inp.stat())
            path.with_name(self.STAT_FILE).write_text(f'{mtime} {size}', encoding='utf-8')
            return True, logfile


    #----------------------------------------------------------------------------------------------
    def check(self, include=True):                                         # IX_input
    #----------------------------------------------------------------------------------------------
        self._checked = True
        # Check if top level afi-file exist
        self.afi.exists(raise_error=True)
        # Check if included files exists
        if include and (missing := [f for f in self.include_files() if not f.exists()]):
            raise SystemError(f'ERROR {list2text([f.name for f in missing])} '
                              f'included from {self} is missing in folder {missing[0].parent}')
        return True

    #----------------------------------------------------------------------------------------------
    def files_matching(self, *keys):                                        # IX_input
    #----------------------------------------------------------------------------------------------
        """ Return only ixf-files that match the given keys """
        return (ixf for ixf in self.ixf_files if any(key in ixf for key in keys))

    #----------------------------------------------------------------------------------------------
    def include_files(self):                                               # IX_input
    #----------------------------------------------------------------------------------------------
        return self.afi.include_files()

    #----------------------------------------------------------------------------------------------
    def including(self, *files):                                           # IX_input
    #----------------------------------------------------------------------------------------------
        """ For compatibility with DATA_file """
        return self

    #----------------------------------------------------------------------------------------------
    def nodes(self, *types, files=None, table=False, context=False, **kwargs): # IX_input
    #----------------------------------------------------------------------------------------------
        """
        Return generator of nodes with node syntax:

            node_type "node_name" {
                node_content
            }

        Will also return nodes without content
        """
        if files is None:
            if types[0] == 'all':
                files = self.ixf_files
            else:
                # Only return nodes from relevant files (might be faster for large files)
                files = list(self.files_matching(*types))
        #files = files or ixf_files
        # Prepare generators of both context and table nodes
        contexts = flatten(file.node(*types, table=False, **kwargs) for file in files)
        # tables = flatten(file.node(*types, brace=(b'\\[',b'\\]'), **kwargs) for file in files)
        tables = flatten(file.node(*types, table=True, **kwargs) for file in files)
        if table and context:
            return (node for node in chain(contexts, tables) if node.content)
        if table:
            return (table for table in tables if table.content)
        if context:
            return (context for context in contexts if context.content)
        return contexts


    #----------------------------------------------------------------------------------------------
    def get_node(self, node):                                              # IX_input
    #----------------------------------------------------------------------------------------------
        return next(self.nodes(node.type, table=node.is_table, context=node.is_context), None)

    #----------------------------------------------------------------------------------------------
    def start(self):                                                       # IX_input
    #----------------------------------------------------------------------------------------------
        pattern = r'Start(\w+) *= *(\w+)'
        key_val = findall(pattern, next(self.nodes('Simulation')).content)
        # Use lowerkey names
        values = {k.lower():v for k,v in key_val}
        # Convert year, day, hour, minute, second
        int_values = {k:int(v) for k,v in values.items() if v.isnumeric()}
        values.update(int_values)
        # Convert month name to month number
        values['month'] = datetime.strptime(values['month'],'%B').month
        return datetime(**values)

    #----------------------------------------------------------------------------------------------
    def _timestep_files(self):                                             # IX_input
    #----------------------------------------------------------------------------------------------
        date_files = [ixf for ixf in self.ixf_files if 'DATE' in ixf]
        field_files = [ixf for ixf in date_files if next(ixf.node('FieldManagement'), None)]
        if field_files:
            return field_files[0]
        if date_files:
            return date_files[-1]
            #return date_files[0]

    #----------------------------------------------------------------------------------------------
    def timesteps(self, start=None, **kwargs):                             # IX_input
    #----------------------------------------------------------------------------------------------
        """
        Return list of timesteps for each report step
        The TIME node gives the cumulative timesteps, but a list of
        separate timesteps is returned, similar to how TSTEP is used
        in ECLIPSE (DATA-file)
        """
        start = start or self.start()
        def date(string):
            pattern = '%d-%b-%Y'
            if ':' in string:
                pattern += ' %H:%M:%S'
            if '.' in string:
                pattern += '.%f'
            return (datetime.strptime(string, pattern) - start).total_seconds()/86400
        file = self._timestep_files()
        nodes = self.nodes('DATE','TIME', files=(file,), convert=(date, float))
        cum_steps = (node.name for node in nodes)
        steps = [b-a for a,b in pairwise(chain([0], sorted(set(cum_steps))))]
        # Check for negative steps (could happen if the same DATE/TIME is given in more than one file)
        # if neg := next((i for i,val in enumerate(steps) if val <= 0), None):
        #     # Ignore steps after the negative step
        #     steps = steps[:neg]
        return steps

    #----------------------------------------------------------------------------------------------
    def report_dates(self):                                                # IX_input
    #----------------------------------------------------------------------------------------------
        start = self.start()
        return [start + timedelta(days=days) for days in accumulate(self.timesteps())]

    #----------------------------------------------------------------------------------------------
    def wellnames(self, contains:str=''):                                  # IX_input
    #----------------------------------------------------------------------------------------------
        wells = self.wells()
        if contains:
            return sorted(well[0] for well in wells if contains in well[1])
        return sorted(well[0] for well in wells)
        #return tuple(set(node.name for node in self.nodes('Well')))
        #return tuple(set(node.name for node in self.nodes('WellDef')))

    #----------------------------------------------------------------------------------------------
    def wells(self):                                                       # IX_input
    #----------------------------------------------------------------------------------------------
        return (set((well.name, _type[0]) for well in self.nodes('Well') if (_type:=well.get('Type'))))

    #----------------------------------------------------------------------------------------------
    def wells_by_type(self):                                               # IX_input
    #----------------------------------------------------------------------------------------------
        """
        Categorizes wells by their type.

        This method organizes wells into a dictionary where the keys are well types
        and the values are lists of well names corresponding to each type.

        Returns:
            dict: A dictionary where the keys are well types (e.g., 'producer', 'injector')
                and the values are lists of well names belonging to each type.
        """
        wells = defaultdict(list)
        for wname, wtype in self.wells():
            wells[wtype].append(wname)
        return dict(wells)

    #----------------------------------------------------------------------------------------------
    def wellpos(self, *wellnames):                                         # IX_input
    #----------------------------------------------------------------------------------------------
        """
        Retrieves the well positions for the specified well names.

        For each well name provided, this method searches through the 'WellDef' nodes,
        extracts the cell connection indices from the node content using a regular expression,
        and returns the positions as tuples of zero-based (i, j, k) indices.

        Parameters:
            *wellnames (str): Variable length argument list of well names to retrieve positions for.

        Returns:
            tuple: A tuple containing lists of (i, j, k) index tuples for each well name, in the order provided.

        Notes:
            - Only wells that have both 'WellToCellConnections' and 'Completion' in their node attributes are considered.
            - If a well name is not found, its corresponding entry will be an empty list.
        """
        well_list = list(wellnames)
        wpos = {well:[] for well in wellnames}
        regex = re_compile(r'^\s*\(([\d ]+)\)', MULTILINE)
        for node in self.nodes('WellDef'):
            if 'WellToCellConnections' in node and 'Completion' in node and node.name in well_list:
                strings = (m.group(1) for m in regex.finditer(node.content))
                # Subtract 1 to make indices zero-based
                index = fromstring(' '.join(strings), dtype=int, sep=' ') - 1
                wpos[node.name] = tuple(map(tuple, index.reshape(-1, 3).tolist()))
                well_list.remove(node.name)
                if not well_list:
                    break
        return tuple(wpos.values())

    #----------------------------------------------------------------------------------------------
    def wellpos_by_name(self, ijk=False, wellnames=None):                  # IX_input
    #----------------------------------------------------------------------------------------------
        """
        Returns a dictionary mapping well names to their positions.

        Parameters:
            ijk (bool, optional): If True, returns positions as tuples of I, J, K indices.
                                If False (default), returns positions as returned by self.wellpos.
            wellnames (list of str, optional): List of well names to include. If None, uses all well names.

        Returns:
            dict: A dictionary where keys are well names and values are positions.
                If ijk is True, positions are tuples of (I, J, K) indices.
                If ijk is False, positions are as returned by self.wellpos.

        """
        wellnames = wellnames or self.wellnames()
        name_pos = dict(zip(wellnames, self.wellpos(*wellnames)))
        if ijk:
            return {name:tuple(zip(*pos)) for name, pos in name_pos.items()}
        return name_pos

    #----------------------------------------------------------------------------------------------
    def injectors(self, *wellnames):                                         # IX_input
    #----------------------------------------------------------------------------------------------
        inj_names = self.wells_by_type()['WATER_INJECTOR']
        pos, = self.wellpos(*inj_names)
        return pos

    #----------------------------------------------------------------------------------------------
    def wells_near_cells(self, *cells):                                         # IX_input
    #----------------------------------------------------------------------------------------------
        wells, kind = zip(*self.wells())
        wellbbox = [bounding_box(wp) for wp in self.wellpos(*wells)]
        return [(w, k, bb) for w, k, bb in zip(wells, kind, wellbbox) if any_cell_in_box(cells, bb)]


    #----------------------------------------------------------------------------------------------
    def summary_keys(self, matching=()):                                   # IX_input
    #----------------------------------------------------------------------------------------------
        """
        Summary keys can be in table format (using []) or in node format (using {}).
        Hence, we need to extract keys from both formats.
        """
        keys = ('WellProperties', 'FieldProperties')
        # Extract table node values
        # Get last (second) column, but skip first row
        table_keys = flatten(table.columns()[-1][1:] for table in self.nodes(*keys, table=True))
        # Extract context node values
        node_data = ''.join(node.content for node in self.nodes(*keys, context=True))
        # Ignore commented lines [^#]+? (?=lazy expansion)
        pattern = r'^[^#]+?report_label *= *"*(\w+)'
        node_keys = (m.group(1) for m in finditer(pattern, node_data, flags=MULTILINE))
        # Set of unique keys
        keys = (key.replace('"','') for key in set(chain(table_keys, node_keys)))
        if matching:
            assert isinstance(matching, (list, tuple)), "'matching' must be list or tuple"
            return [key for key in keys if key in matching]
        return list(keys)


    #----------------------------------------------------------------------------------------------
    def mode(self):                                                        # IX_input
    #----------------------------------------------------------------------------------------------
        return 'forward'

    #----------------------------------------------------------------------------------------------
    def restart(self):                                                     # IX_input
    #----------------------------------------------------------------------------------------------
        # Check if this is a restart-run
        match = next((m for m in self.afi.matches() if m[2] and b'restart' in m[2]), None)
        if match:
            folder = self.path.with_name(match[1].decode())
            keymatch = finditer(rb'(\w+)=["\']([^"\']+)["\']', match[2])
            days = float(next(keymatch)[2])
            if not folder.exists():
                raise SystemError(f'ERROR Restart folder {folder} is missing')
            ixf = IXF_file(folder/'fm/fmworld.ixf')
            step = 0
            if ixf.is_file():
                fm = next(ixf.node('FieldManagement'), None)
                if fm:
                    step = int(fm.get('ConvergedTimeStepsCount')[0])
                    timeunits = ('Year', 'Month', 'Day', 'Hour', 'Minute', 'Second')
                    date = '-'.join(fm.get(*['Start'+name for name in timeunits]))
                    start = datetime.strptime(date, '%Y-%B-%d-%H-%M-%S')
                else:
                    raise SystemError(f'ERROR Missing FieldManagement node in {ixf}')
            else:
                raise SystemError(f'ERROR {ixf.path} is missing')
            return Restart(start=start, days=days, step=int(step))
        return Restart()

    #----------------------------------------------------------------------------------------------
    def UNRST_settings(self):                                              # IX_input
    #----------------------------------------------------------------------------------------------
        nodename = 'Recurrent3DReport'
        nodes = list(self.nodes(nodename, context=True))
        if nodes:
            return nodes[-1]
        raise SystemError((f"ERROR Node {nodename} not found in {self}"))

    #----------------------------------------------------------------------------------------------
    def write_unified_UNRST(self):                                         # IX_input
    #----------------------------------------------------------------------------------------------
        unified = self.UNRST_settings().get('Unified')
        # Default is True if not set
        if not unified or unified[0] in ('TRUE', 'True', 'true'):
            return True
        return False
