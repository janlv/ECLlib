
# -*- coding: utf-8 -*-

DEBUG = False 
ENDIAN = '>'  # Big-endian

from dataclasses import dataclass
from pathlib import Path
from numpy import zeros, int32, float32, float64, ceil, bool_ as np_bool, array as nparray, append as npappend 
from mmap import ACCESS_WRITE, mmap, ACCESS_READ
from re import finditer, compile
from copy import deepcopy
from collections import namedtuple
from datetime import datetime
from struct import unpack, pack, error as struct_error
#from numba import njit, jit
from .utils import file_contains, list2str, float_or_str, remove_comments

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

@dataclass
class Dtyp:
    name   : str   # ECL type name
    unpack : str   # Char used by struct.unpack/pack to read/write binary data
    size   : int   # Bytesize 
    max    : int   # Maximum number of data records in one block
    nptype : type  # Type used in numpy arrays 

DTYPE = {b'INTE' : Dtyp('INTE', 'i', 4, 1000, int32),
         b'REAL' : Dtyp('REAL', 'f', 4, 1000, float32),
         b'LOGI' : Dtyp('LOGI', 'i', 4, 1000, np_bool),
         b'DOUB' : Dtyp('DOUB', 'd', 8, 1000, float64),
         b'CHAR' : Dtyp('CHAR', 's', 8, 105 , str),
         b'MESS' : Dtyp('MESS', ' ', 1, 1   , str)}

DTYPE_LIST = [v.name for v in DTYPE.values()]


# #-----------------------------------------------------------------------
# def get_tsteps_from_schedule_files(root, raise_error=False):
# #-----------------------------------------------------------------------
#     print('get_tsteps_from_schedule_files')
#     # Search for DATES in schedule-files and convert to TSTEP
#     DATA_file = Input_file(f'{root}.DATA')
#     start = DATA_file.get('START')
#     # Find schedule-files in root folder, ignore suffix case 
#     sch_files = [f for f in Path(root).parent.glob('**/*') if f.suffix.lower() == '.sch']
#     dates = [start]
#     for fil in sch_files:
#         # Check if file is used in the .DATA-file before searching
#         # if file_contains(DATA_file, fil.name, end='END') and file_contains(fil, 'DATES'):
#         if file_contains(input.file, fil.name, end='END') and file_contains(fil, 'DATES'):
#             dates = Input_file(fil).get('DATES')
#             break
#     days = [(d-start).days for d in dates]
#     tsteps = [days[i+1]-days[i] for i in range(len(days)-1)]
#     tsteps.insert(0, days[0])
#     return tsteps

        

#====================================================================================
class unfmt_block:
    #
    # Block of formatted Eclipse data
    #
#====================================================================================
    #  | h e a d e r  |   d a t a     |
    #  |4i|8s|4i|4s|4i|4i|1000 data|4i| 
    #  |    24 bytes  |
    #  |              |4i|8d| 

    #--------------------------------------------------------------------------------
    def __init__(self, key=b'', length=0, type=b'', start=0, end=0, data=None, data_start=0, file=None):
    #--------------------------------------------------------------------------------
        self._key = key
        self._length = length
        self._type = type
        self._dtype = DTYPE[type]
        self._mmap = data
        self._file = file
        self._startpos = start
        self._end = end
        self._data_start = data_start
        DEBUG and print(f'Creating {self}')

    #--------------------------------------------------------------------------------
    def __str__(self):                                                  # unfmt_block
    #--------------------------------------------------------------------------------
        return f'<unfmt_block(key={self.key():8s}, type={self.type():4s}, bytes={self.bytes():8d}, length={self.length():8d}, start={self._startpos:8d}, end={self._end:8d}>'


    #--------------------------------------------------------------------------------
    def __del__(self):                                                  # unfmt_block
    #--------------------------------------------------------------------------------
        DEBUG and print(f'Deleting {self}')


    #--------------------------------------------------------------------------------
    def length(self):                                                   # unfmt_block
    #--------------------------------------------------------------------------------
        return self._length*(self._type==b"CHAR" and 8 or 1)


    #--------------------------------------------------------------------------------
    def bytes(self):                                                    # unfmt_block
    #--------------------------------------------------------------------------------
        return self._type and self._length*self._dtype.size or 0


    #--------------------------------------------------------------------------------
    def key(self):                                                      # unfmt_block
    #--------------------------------------------------------------------------------
        return self._key.decode().strip()

    #--------------------------------------------------------------------------------
    def type(self):                                                     # unfmt_block
    #--------------------------------------------------------------------------------
        return self._type.decode()
        
    #--------------------------------------------------------------------------------
    def start(self):                                                    # unfmt_block
    #--------------------------------------------------------------------------------
        return self._startpos

    #--------------------------------------------------------------------------------
    def end(self):                                                      # unfmt_block
    #--------------------------------------------------------------------------------
        return self._end

    #--------------------------------------------------------------------------------
    def info(self, details=False):                                      # unfmt_block
    #--------------------------------------------------------------------------------
        s = f'{self._key.decode()}'
        if details:
            # s += f' block of {self._length*datasize[self._type]} bytes' 
            s += f' block of {self._length*self._dtype.size} bytes' 
            # s += f' holding {self._length*(self._type==b"CHAR" and 8 or 1)} {self._type.decode()}'
            s += f' holding {self._length*(self._dtype.name=="CHAR" and 8 or 1)} {self._dtype.name}'
            s += f' at [{self._startpos}, {self._end}]'
        return s

    #--------------------------------------------------------------------------------
    def print(self, data=False, details=False):                         # unfmt_block
    #--------------------------------------------------------------------------------
        print(self.info(details=details), end='')
        if data:
            print(':',self.data())
        else:
            print()

    #--------------------------------------------------------------------------------
    def read_size_at(self, pos):                                        # unfmt_block
    #--------------------------------------------------------------------------------
        size = unpack(ENDIAN+'i',self._mmap[pos:pos+4])[0]
        if self._type == b'CHAR':
            n = size
        else:
            n = int(size/self._dtype.size)
        return size, n 


    #--------------------------------------------------------------------------------
    def replace(self, func, key=None):                                  # unfmt_block
    #--------------------------------------------------------------------------------
        with open(self._file, mode='r+') as f:
            with mmap(f.fileno(), length=0, access=ACCESS_WRITE) as mmap_write:
                pos = self._startpos + 4
                if key is None:
                    key = self.key()
                mmap_write[pos:pos+8] = f'{key:8s}'.encode()
                newdata = func(self.data())
                pos = self._startpos + 24  # header: 4+8+4+4+4 = 24, data: 4 + 1000 data + 4
                N = 0
                while N < len(newdata):
                    size, n = self.read_size_at(pos)
                    pos += 4
                    mmap_write[pos:pos+size] = pack(ENDIAN+f'{n}{self._dtype.unpack}', *newdata[N:N+n])
                    pos += size + 4
                    N += n
                mmap_write.flush()


    #--------------------------------------------------------------------------------
    def data(self, raise_error=False):                                  # unfmt_block
    #--------------------------------------------------------------------------------
        value = []
        a = self._data_start
        while a < self._end:
            size, n = self.read_size_at(a)
            a += 4
            try:
                value.extend(unpack(ENDIAN+f'{n}{self._dtype.unpack}', self._mmap[a:a+size]))
            except struct_error as e:
                if raise_error:
                    raise SystemError(f'ERROR Unable to read {self._file.name}, corrupted file?')
                return None
            a += size + 4
        if self._type == b'CHAR':
            value = [b''.join(value).decode()]
        return value

#====================================================================================
@dataclass
class keypos:
#====================================================================================
    key: str = ''
    pos: int = 0
    name: str = ''


#====================================================================================
class unfmt_file:
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, filename):                                        # unfmt_file
    #--------------------------------------------------------------------------------
        self.fileobj = None
        self.file = Path(filename)
        self.endpos = 0
        DEBUG and print(f'Creating {self}')


    #--------------------------------------------------------------------------------
    def __str__(self):                                                  # unfmt_file
    #--------------------------------------------------------------------------------
        return f'<unfmt_file(filename={self.file}>'


    #--------------------------------------------------------------------------------
    def __del__(self):                                                  # unfmt_file
    #--------------------------------------------------------------------------------
        DEBUG and print(f'Deleting {self}')




    #--------------------------------------------------------------------------------
    def is_file(self):                                                   # unfmt_file
    #--------------------------------------------------------------------------------
        return self.file.is_file()

        
    #--------------------------------------------------------------------------------
    def size(self):                                                      # unfmt_file
    #--------------------------------------------------------------------------------
        return self.file.stat().st_size


    #--------------------------------------------------------------------------------
    def name(self):                                                      # unfmt_file
    #--------------------------------------------------------------------------------
        return self.file.name


    #--------------------------------------------------------------------------------
    def blocks(self, only_new=False, start=None):                        # unfmt_file
    #--------------------------------------------------------------------------------
        if not self.is_file() or self.size()<24: # Header is 24 bytes
            return
        startpos = 0
        if only_new:
            startpos = self.endpos
        if start:
            startpos = start
        with open(self.file, mode='rb') as file:
            with mmap(file.fileno(), length=0, access=ACCESS_READ) as data:
                data.seek(startpos, 1)
                while data.tell() < data.size():
                    start = data.tell()
                    # Header
                    try:
                        data.seek(4, 1)
                        key, length, type = unpack(ENDIAN+'8si4s', data.read(16))
                        data.seek(4, 1)
                        # Value array
                        data_start = data.tell()
                        bytes = length*DTYPE[type].size + 8*int(ceil(length/DTYPE[type].max))
                        data.seek(bytes, 1)
                    except (ValueError, struct_error): # as e:
                        # Catch 'seek out of range' error
                        #print(f'break in blocks(): {e}')
                        break
                    yield unfmt_block(key=key, length=length, type=type, start=start, end=data.tell(), 
                                      data=data, data_start=data_start, file=self.file)
                self.endpos = data.tell()


    #--------------------------------------------------------------------------------
    def tail_blocks(self):                                               # unfmt_file
    #--------------------------------------------------------------------------------
        if not self.is_file() or self.size()<24: # Header is 24 bytes
            return
        with open(self.file, mode='rb') as file:
            with mmap(file.fileno(), length=0, access=ACCESS_READ) as data:
                # Goto end of file
                data.seek(0, 2)
                while data.tell() > 0:
                    end = data.tell()
                    # Header
                    # Rewind until we find a header
                    while data.tell() > 0:
                        try:
                            data.seek(-4, 1)
                            size = unpack(ENDIAN+'i',data.read(4))[0]
                            data.seek(-4-size, 1)
                            # if self.is_header(data, size, data.tell()):
                            pos = data.tell()
                            # Check if this is a header
                            if size == 16 and data[pos+12:pos+16] in DTYPE.keys():
                                start = data.tell()-4
                                key, length, type = unpack(ENDIAN+'8si4s', data.read(16))
                                data.seek(4, 1)
                                break
                            else:
                                data.seek(-4, 1)
                        except (ValueError, struct_error):
                            return None
                    # Value array
                    data_start = data.tell()
                    data.seek(start, 0)
                    yield unfmt_block(key=key, length=length, type=type, start=start, end=end, 
                                    data=data, data_start=data_start)


    #--------------------------------------------------------------------------------
    def get(self, var_list, N=0, stop=(), raise_error=True):       # unfmt_file
    #--------------------------------------------------------------------------------
        blocks = self.blocks
        if N < 0:
            # Read data from end of file
            blocks = self.tail_blocks
            N = -N
        varmap = {k:v for k,v in self.varmap.items() if k in var_list}
        # Create dict of keywords with varname and position:
        #  {'INTEHEAD':[('day',64), ('month',65), ('year',66)]}
        var_pos = {v.key:[] for v in varmap.values()}
        [var_pos[v.key].append( (k, v.pos) ) for k,v in varmap.items()]        
        values = {v:[] for v in var_list}
        size = lambda : (len(v) for v in values.values())
        for b in blocks():
            if b.key() in var_pos.keys():
                for var, pos in var_pos[b.key()]:
                    values[var].append( b.data()[pos] )
            if N and set(size()) == set([N]): 
                break
            if stop and stop[1] == values[stop[0]][-1] and len(set(size())) == 1:
                break
        if raise_error and not all(values.values()):
            raise SystemError(f'ERROR Unable to read {var_list} from {self.file.name}')
        return list(values.values())        


    #--------------------------------------------------------------------------------
    def exists(self):                                                    # unfmt_file
    #--------------------------------------------------------------------------------
        #print('Checking for ' + self._filename)
        if self.file.is_file():
            return True
        return False


    #--------------------------------------------------------------------------------
    def create(self, *args, progress=lambda x:None, cancel=lambda:None): # unfmt_file
    #--------------------------------------------------------------------------------
        sections = args
        # Make sure the sizes of each section are equal
        # Remove last unit if they are unequal
        min_size = min([sec.size() for sec in sections])
        for sec in sections:
            while sec.size() > min_size:
                sec.pop_unit(-1)
                #print('pop from '+str(sec.filename()))
        # Open files
        progress(-(min_size+1))
        out_file = open(self.file, 'wb')
        for sec in sections:
            sec.open_file()
        # Write sections to out_file
        OK = True
        n = 0
        while OK: 
            for sec in sections:
                OK = sec.write_next_unit(out_file)
            n += 1
            progress(n)
            cancel()
        # Close files
        for sec in sections:
            sec.close_file()
        out_file.close()
        return self.file



#====================================================================================
class Input_file:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file, check=True, read=True):
    #--------------------------------------------------------------------------------
        self.file = Path(file)
        if check and read and not self.file.is_file():
            raise SystemError(f'ERROR Eclipse input-file {self.file.name} is missing in folder {self.file.parent}')        
        self._data = None
        self._read = read
        if read:
            self._remove_comments()
        self._restart_file = None
        self._restart_time = None
        getter = namedtuple('getter', 'default convert pattern')
        self._get = {'TSTEP'   : getter([],      self._float, r'\bTSTEP\b\s+([0-9*.\s]+)/'),
                     'START'   : getter([0],     self._date,  r'\bSTART\b\s+(\d+\s+\'*\w+\'*\s+\d+)'),
                     'DATES'   : getter([0],     self._date,  r'\bDATES\b\s+(\d+\s+\'*\w+\'*\s+\d+)'), 
                     'INCLUDE' : getter([''],    self._file,  r"\bINCLUDE\b\s+'*([a-zA-Z0-9_./\\-]+)'*\s*/"), 
                     'RESTART' : getter(['', 0], self._file,  r"\bRESTART\b\s+('*[a-zA-Z0-9_./\\-]+'*\s+[0-9]+)\s*/")}


    #--------------------------------------------------------------------------------
    def __str__(self):
    #--------------------------------------------------------------------------------
        return f'{self.file}'


    #--------------------------------------------------------------------------------
    def include_file(self, suffix):
    #--------------------------------------------------------------------------------
        ''' Return first included file with given suffix (case-insensitive) or None '''
        return next((f for f in self.get('INCLUDE') if suffix in str(f).lower()), None)


    #--------------------------------------------------------------------------------
    def date2tstep(self, dates):
    #--------------------------------------------------------------------------------
        start = self.get('START')[0]
        days = [(d-start).days for d in dates]
        tsteps = [days[i+1]-days[i] for i in range(len(days)-1)]
        tsteps.insert(0, days[0])
        return tsteps


    #--------------------------------------------------------------------------------
    def is_file(self):
    #--------------------------------------------------------------------------------
        return self.file.is_file()


    #--------------------------------------------------------------------------------
    def exists(self, raise_error=True):
    #--------------------------------------------------------------------------------
        if self.file.is_file():
            return True
        if raise_error:
            raise SystemError(f'ERROR DATA-file is missing in folder {self.file.parent}')


    #--------------------------------------------------------------------------------
    def _remove_comments(self):
    #--------------------------------------------------------------------------------
        self._data = remove_comments(self.file, end='END')

    #-----------------------------------------------------------------------
    def _float(self, values, key, raise_error=False):
    #-----------------------------------------------------------------------
        # Process x*y statements
        mult = lambda x, y : int(x)*(' '+y) 
        values = [a if not '*' in a else mult(*a.split('*')) for a in values]
        values = [float(b) for a in values for b in a.split()] or [0]
        return values or self._get[key].default

    #-----------------------------------------------------------------------
    def _date(self, values, key, raise_error=False):
    #-----------------------------------------------------------------------
        dates = [' '.join((values[i], values[i+1], values[i+2])).replace("'",'') for i in range(0, len(values), 3)]
        dates = [datetime.strptime(d, '%d %b %Y').date() for d in dates]
        return dates or self._get[key].default


    #--------------------------------------------------------------------------------
    def _file(self, values, key, raise_error=True):
    #--------------------------------------------------------------------------------
        # Remove quotes and backslash
        values = [val.replace("'",'').replace('\\','/') for val in values]
        # Convert numbers if they exist
        for i, val in enumerate(values):
            try:
                values[i] = int(val)
            except ValueError:
                pass
        values = [(self.file.parent/val).resolve() if isinstance(val, str) else val for val in values]
        # Add suffix for RESTART keyword
        if key == 'RESTART' and values:
            values[0] = values[0].with_suffix('.UNRST')
        # Check if files are missing
        missing = [str(val) for val in values if not isinstance(val, int) and not val.is_file()]
        if missing and raise_error:
            raise SystemError(f'ERROR {key}-files requested in {self.file.name} are not found: {", ".join(missing)}')
        return values or self._get[key].default


    #-----------------------------------------------------------------------
    def get(self, keyword, raise_error=False):
    #-----------------------------------------------------------------------
        if not self._read:
            self._remove_comments()
        keyword = keyword.upper()
        key = self._get[keyword]
        match = compile(key.pattern).findall(self._data)
        values = [n for m in match for n in m.split()]
        if raise_error and not values:
            raise SystemError(f'ERROR Keyword {keyword} not found in {self.file}')
        return key.convert(values, keyword, raise_error=raise_error)






#====================================================================================
class UNRST_file(unfmt_file):
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, file, wait_func=None):
    #--------------------------------------------------------------------------------
        suffix = '.UNRST'
        super().__init__(Path(file).with_suffix(suffix))
        self.varmap = {'step'  : keypos(key='SEQNUM'),
                       'nwell' : keypos('INTEHEAD', 16 , 'NWELLS'), 
                       'day'   : keypos('INTEHEAD', 64 , 'IDAY'),
                       'month' : keypos('INTEHEAD', 65 , 'IMON'),
                       'year'  : keypos('INTEHEAD', 66 , 'IYEAR'),
                       'hour'  : keypos('INTEHEAD', 206, 'IHOURZ'),
                       'min'   : keypos('INTEHEAD', 207, 'IMINTS'),
                       'sec'   : keypos('INTEHEAD', 410, 'ISECND'),
                       'time'  : keypos(key='DOUBHEAD')}
        self.check = check_blocks(self, start='SEQNUM', end='ENDSOL', wait_func=wait_func)


    #--------------------------------------------------------------------------------
    def date(self, N=0):                                          # UNRST_file
    #--------------------------------------------------------------------------------
        y, m, d = self.get(['year','month','day'], N=N)
        return datetime.strptime(f'{d[-1]} {m[-1]} {y[-1]}', '%d %m %Y').date()




#====================================================================================
class RFT_file(unfmt_file):
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file, wait_func=None, nwell=0):
    #--------------------------------------------------------------------------------
        suffix = '.RFT'
        super().__init__(Path(file).with_suffix(suffix))
        self.check = check_blocks(self, start='TIME', end='CONNXT', wait_func=wait_func)


#====================================================================================
class UNSMRY_file(unfmt_file):
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file):
    #--------------------------------------------------------------------------------
        suffix = '.UNSMRY'
        super().__init__(Path(file).with_suffix(suffix))
        self.varmap = {'time' : keypos(key='PARAMS'), 
                       'step' : keypos(key='MINISTEP')}


#====================================================================================
class MSG_file:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file):
    #--------------------------------------------------------------------------------
        self.file = Path(file).with_suffix('.MSG')
        self._pattern = {'time' : r'<\s*\bmessage\b\s+\bdate\b="[0-9/]+"\s+time="([0-9.]+)"\s*>',
                        'step' : r'\bRESTART\b\s+\bFILE\b\s+\bWRITTEN\b\s+\bREPORT\b\s+([0-9]+)'}
        self._convert = {'time' : float,
                         'step' : int}

    #-----------------------------------------------------------------------
    def size(self):
    #-----------------------------------------------------------------------
        return self.file.stat().st_size

    #-----------------------------------------------------------------------
    def get(self, var_list, N=0, raise_error=True):
    #-----------------------------------------------------------------------
        with open(self.file) as f:
            lines = f.readlines()
        lines = ''.join(lines)
        values = {}
        for var in var_list:
            match = compile(self._pattern[var]).findall(lines)
            values[var] = [self._convert[var](m) for m in match]
        if raise_error and not all(values.values()):
            raise SystemError(f'ERROR Unable to read {var_list} from {self.file.name}')
        if N == 0:
            return list(values.values())
        else:
            if N > 0:
                N -= 1
            return [[v[N]] if v else [] for v in values.values()]


#====================================================================================
class check_blocks:                                                    # check_blocks
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, file, start=None, end=None, wait_func=None):      # check_blocks
    #--------------------------------------------------------------------------------
        if isinstance(file, unfmt_file):
            self._unfmt = file
        else:
            self._unfmt = unfmt_file(file)
        self._key = {'start':start.ljust(8).encode(), 'end':end.ljust(8).encode()}
        self._start = []
        self._end = 0
        self._startpos = 0
        self._wait_func = wait_func
        DEBUG and print(f'Creating {self}')

    #--------------------------------------------------------------------------------
    def __str__(self):                                                 # check_blocks
    #--------------------------------------------------------------------------------
        return f'<check_blocks(file={self._unfmt}>'


    #--------------------------------------------------------------------------------
    def __del__(self):                                                 # check_blocks
    #--------------------------------------------------------------------------------
        DEBUG and print(f'Deleting {self}')


    #--------------------------------------------------------------------------------
    def info(self):                                                    # check_blocks
    #--------------------------------------------------------------------------------
        return f"  {self._key['start'].decode()} : {list2str(self._start)}"

        
    #--------------------------------------------------------------------------------
    def _blocks_complete(self, nblocks=1):                           # check_blocks
    #--------------------------------------------------------------------------------
        self._start, self._end = [], 0
        for b in self._unfmt.blocks(start=self._startpos):
            if b._key == self._key['start']:
                self._start.append(b.data()[0])
                # self.out['startpos'].append(b.start())
            if b._key == self._key['end']:
                self._end += 1 
                if self._end == nblocks and len(self._start) == nblocks:
                    #self.file.set_startpos(b.end())
                    self._startpos = b.end()
                    return True
        return False


    #--------------------------------------------------------------------------------
    def data_saved_maxmin(self, max=1, min=1, iter=100, **kwargs):      # check_blocks
    #--------------------------------------------------------------------------------
        f'''
            Loop for {iter} iterations until {max} start/end-blocks are found.
            If {max} start/end-blocks are not found within the iteration limit, 
            look for {max}-1 start/end-blocks and so on until {min} blocks. 
        '''
        msg = ''
        for n in range(max, min-1, -1):
            passed = self._wait_func( self._blocks_complete, nblocks=n, log=self.info, limit=iter, **kwargs )
            if passed:
                break
        if n < max:
            msg = f'WARNING! Only {n} start/stop-blocks found in {self._unfmt.file.name}, expected {max}'
        return msg


    #--------------------------------------------------------------------------------
    def data_saved(self, nblocks=1, **kwargs):               # check_blocks
    #--------------------------------------------------------------------------------
        return self._wait_func( self._blocks_complete, nblocks=nblocks, log=self.info, **kwargs)



#====================================================================================
class Section:                                                              # Section
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, filename, init_key='SEQNUM', start_before=None, start_after=None,
                 end_before=None, end_after=None, skip_sections=None, remove_blocks=None):
    #--------------------------------------------------------------------------------
        self._filename = Path(filename)
        self._fh = None # filehandle
        self.init_key = init_key
        self.start_before = start_before
        self.start_after = start_after
        self.end_before = end_before
        self.end_after = end_after
        if remove_blocks and not isinstance(remove_blocks, tuple):
            remove_blocks = (remove_blocks,)
        self.remove_blocks = remove_blocks
        self._units = self.startpos_and_size()
        if skip_sections != None:
            if not isinstance(skip_sections, tuple):
                skip_sections = (skip_sections,)
            for sec in skip_sections:
                #print('sec:'+str(sec))
                self._units.pop(sec)
            
    #--------------------------------------------------------------------------------
    def open_file(self):                                                    # Section
    #--------------------------------------------------------------------------------
        self._fh = open(self._filename, 'rb')
        
    #--------------------------------------------------------------------------------
    def close_file(self):                                                   # Section
    #--------------------------------------------------------------------------------
        self._fh.close()
        
    #--------------------------------------------------------------------------------
    def filename(self):                                                     # Section
    #--------------------------------------------------------------------------------
        return self._filename
        
    #--------------------------------------------------------------------------------
    def size(self):                                                         # Section
    #--------------------------------------------------------------------------------
        return len(self._units)
        
    #--------------------------------------------------------------------------------
    def pop_unit(self, nr):                                                 # Section
    #--------------------------------------------------------------------------------
        return self._units.pop(nr)
        
    #--------------------------------------------------------------------------------
    def write_next_unit(self, out_file):                                    # Section
    #--------------------------------------------------------------------------------
        if len(self._units)==0:
            return False
        unit = self._units.pop(0)
        for pos, size in zip(unit[::2],unit[1::2]):
            self._fh.seek(pos)
            out_file.write( self._fh.read(size) )
        return True
    
    #--------------------------------------------------------------------------------
    def print(self):                                                        # Section
    #--------------------------------------------------------------------------------
        for unit in self._units:
            print(unit)

                             
    #--------------------------------------------------------------------------------
    def startpos_and_size(self):                                            # Section
    #--------------------------------------------------------------------------------
        # A unit is a list of consecutive absolute filepositions and relative
        # byte chuncks corresponding to the length of the blocks to be kept.
        # A unit list always starts with a pos and ends with a size
        if not self._filename.is_file():
            raise FileNotFoundError(str(self._filename) + ' not found in Section')
        units = []
        inside = False
        #self.keys = []
        for block in unfmt_file(self._filename).blocks():
            key = block.key()
            if inside and key==self.end_before:# and len(start)>1:
                inside = False
                # size
                unit.append(block.start()-unit[-1])
                #self.keys.append('A:'+key+',size:'+str(size[-1]))
            if inside and key==self.end_after:
                inside = False
                # size
                unit.append(block.end()-unit[-1])
                #self.keys.append('B:'+key+',size:'+str(size[-1]))
            if not inside and key==self.start_before:
                inside = True
                # pos
                unit = [block.start(),]
                units.append(unit)
                #self.keys.append('C:'+key+',start:'+str(start[-1]))
            if not inside and key==self.start_after:
                inside = True
                # pos
                unit = [block.end(),]
                units.append(unit)
                #self.keys.append('D:'+key+',start:'+str(start[-1]))
            if inside and self.remove_blocks and key in self.remove_blocks:
                # size
                unit.append(block.start()-unit[-1])
                # pos
                unit.append(block.end())
                #self.keys.append('E:'+key+',size:'+str(size[-1])+',start:'+str(start[-1]))
        if self.end_before==self.init_key:
            unit.append(self._filename.stat().st_size-unit[-1])
            #self.keys.append('F:'+self.end_before+',size:'+str(size[-1]))
        return units
        


#====================================================================================
class fmt_block:                                                         # fmt_block
    #
    # Block of formatted Eclipse data
    #
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, keyword=None, length=None, datatype=None, data=zeros(0)): # fmt_block
    #--------------------------------------------------------------------------------
        self.keyword = keyword
        self.length = length
        #self.datatype = datatype
        self._dtype = DTYPE[datatype.encode()]
        self.data = data
        #self.max_length = max_length

    #--------------------------------------------------------------------------------
    def key(self):                                                        # fmt_block
    #--------------------------------------------------------------------------------
        return self.keyword.strip()
    
    #--------------------------------------------------------------------------------
    def set_key(self, keyword):                                           # fmt_block
    #--------------------------------------------------------------------------------
        self.keyword = keyword #.ljust(8)
    
    #--------------------------------------------------------------------------------
    def unformatted(self):                                                # fmt_block
    #--------------------------------------------------------------------------------
        #dtype = self.datatype.encode()
        length = self.length
        bytes_ = bytearray()
        # header
        bytes_ += pack(ENDIAN + 'i8si4si', 16, self.keyword.encode(), length, self._dtype.name.encode(), 16)
        # data is split in multiple records if length > 1000 
        data = self.data
        while data.size > 0:
            length = min(len(data), self._dtype.max)
            size = self._dtype.size*length
            bytes_ += pack(f'{ENDIAN}i{length}{self._dtype.unpack}i', size, *data[:length], size)
            data = data[length:]
        return bytes_
                
    #--------------------------------------------------------------------------------
    def print(self):                                                      # fmt_block
    #--------------------------------------------------------------------------------
        data = ''
        if len(self.data) > 0:
            data = 'data[0,-1] = ['+str(self.data[0])+', '+str(self.data[-1])+']'
        print(self.keyword, self.length, self._dtype.name, data)

        
#====================================================================================
class fmt_file:                                                            # fmt_file
    #
    # Class to handle formatted Eclipse files.
    #
    # Functions:
    #             blocks(warn_missing=False)
    #             convert()
    #             is_file()
    #
    #
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, filename):                                          # fmt_file
    #--------------------------------------------------------------------------------
        self.name = Path(filename)
        self.fh = None

    #--------------------------------------------------------------------------------
    def is_file(self):                                                     # fmt_file
    #--------------------------------------------------------------------------------
        if self.name.is_file():
            return True
        else:
            #print(f'File {self.name} does not exist!')
            return False
        
    #--------------------------------------------------------------------------------
    def blocks(self, warn_missing=False):                                  # fmt_file
    #--------------------------------------------------------------------------------
        if not self.is_file():
             return
        with open(self.name) as self.fh:
            for line in self.fh:
                try:
                    keyword, length, dtype = self.read_header(line)
                    data = self.read_data(length, dtype)
                except StopIteration:
                    if warn_missing:
                        print(f"\n  WARNING: Missing data in '{keyword}' block, file {self.name.name} not complete!")
                    return
                except TypeError:
                    return
                else:
                    yield fmt_block(keyword, length, dtype, data)
                           
                
    #--------------------------------------------------------------------------------
    def read_header(self, line):                                           # fmt_file
    #--------------------------------------------------------------------------------
        words = line.rstrip().split("\'")
        try:
            return words[1], int(words[2]), words[3]
        except IndexError:
            return
        
    #--------------------------------------------------------------------------------
    def read_data(self, length, dtype, skip=False):                        # fmt_file
    #--------------------------------------------------------------------------------
        if skip:
            data = None
        else:
            # data = zeros(length, dtype=datatype[dtype])
            data = zeros(length, dtype=DTYPE[dtype.encode()].nptype)
        n = 0
        while n < length:
            line = next(self.fh)
            cols = line.rstrip().split()
            m = len(cols)
            if not skip:
                if dtype=='DOUB':
                    cols = [c.replace('D','E') for c in cols]
                try:
                    data[n:n+m] = cols
                except ValueError:
                    return data
            n += m
        return data
            

    #----------------------------------------------------------------------------
    def convert(self, ext='UNRST', init_key='SEQNUM', rename_duplicate=True,
                rename_key=None, echo=False, progress=lambda x:None, cancel=lambda:None):  # fmt_file
    #--------------------------------------------------------------------------------
        if rename_key and len(rename_key)<2:
            raise SystemError(f"ERROR in convert: Format of rename_keyword options is ('old name', 'new name'), but {rename_key} were given")
        stem = self.name.stem.upper()
        fname = str(self.name.parent/stem)+'.'+ext
        out_file = open(fname, 'wb')
        bytes_ = bytearray()
        n = 0
        count = {}
        for block in self.blocks():
            key = block.key()
            if key==init_key and len(bytes_)>0:
                # write previous block to file, and reset bytes_
                n += 1
                out_file.write(bytes_)
                # reset bytes for next section
                bytes_ = bytearray()
                progress(n)
                cancel()
                count = {}
            if rename_duplicate:
                if count.get(key):
                    # duplicate keyname, rename key
                    block.set_key(key[:-1]+str(count[key]+1))
                else:
                    # create new entry
                    count[key] = 0
                count[key] += 1
            if rename_key and key==rename_key[0]:
                block.set_key(rename_key[1])
            bytes_ += block.unformatted()
        out_file.close()
        if echo:
            print(f'{self.name.name} converted to {Path(fname)}')
        return Path(fname)

    #----------------------------------------------------------------------------
    def get_blocks(self, filemap, init_key, rename_duplicate, rename_key): # fmt_file
    #----------------------------------------------------------------------------
        n = 0
        # pos = {k:0 for k in datasize.keys()}
        pos = {k:0 for k in DTYPE.keys()}
        size = {'blocks':0, 'bytes':0}
        num = {'chunks':0, 'blocks':0}
        Blocks = namedtuple('Blocks',['format', 'type', 'head', 'tail', 'slice', 'stride','size','num'])
        blocks = Blocks(format=[], type=[], head=[], tail=[], slice=[], stride=pos, size=size, num=num)
        head_format='i8si4si' # 4+8+4+4+4 = 24
        count = {}
        for match in finditer(b" \'(.{8})\'([0-9 ]{13})\'(.{4})\'", filemap):
            # header
            key, length, dtype = match.groups()
            key = key.decode().strip()
            length = int(length.decode())
            if key==init_key:
                n += 1
                if n>1:
                    size['bytes'] = sum(blocks.tail) + len(blocks.format)*24
                    size['blocks'] = match.start()
                    num['blocks'] = len(blocks.format)
                    num['chunks'] = len(blocks.type)
                    return blocks
            if rename_duplicate:
                #print(key)
                if count.get(key):
                    key = key[:-1]+str(count[key]+1) #).encode()
                    count[key] = 0
                else:
                    # create new entry
                    count[key] = 0
                count[key] += 1
            if rename_key and key==rename_key[0]:
                key = rename_key[1] #.encode()
                #print(key)
            head_data = [16, key.ljust(8).encode(), length, dtype, 16]
            # split block data in chunks if max_length
            # max_l = max_length[dtype]
            max_l = DTYPE[dtype].max
            #L = [min(max_length, length-n*max_length) for n in range(int(length/max_length)+1)]
            L = [min(max_l, length-n*max_l) for n in range(int(length/max_l)+1)]
            L = [l for l in L if l>0]  # Remove possible 0's at the end
            # blocks.format.append( head_format+''.join(['i'+str(l)+unpack_char[dtype]+'i' for l in L]) )
            blocks.format.append( head_format+''.join(['i'+str(l)+DTYPE[dtype].unpack+'i' for l in L]) )
            for i,l in enumerate(L):
                blocks.type.append( dtype )
                # head = [l*datasize[dtype],]
                head = [l*DTYPE[dtype].size,]
                if i==0:
                    head = head_data + head  
                blocks.head.append( head )
                # blocks.tail.append( l*datasize[dtype] )
                blocks.tail.append( l*DTYPE[dtype].size )
                blocks.slice.append( [pos[dtype]+sum(L[:i]), pos[dtype]+sum(L[:i])+l] )
            pos[dtype] += length


    #----------------------------------------------------------------------------
    def get_data_pos(self, filemap, size):                             # fmt_file
    #----------------------------------------------------------------------------
        data = filemap[:size].split()
        # dtypes = [("'"+k+"'").encode() for k in datatype.keys()]
        dtypes = [("'"+k+"'").encode() for k in DTYPE_LIST]
        # data_pos = {k:[] for k in datasize.keys()}
        data_pos = {k:[] for k in DTYPE.keys()}
        for i in range(len(data)):
            if data[i] in dtypes:
                dty = data[i][1:-1] # remove quotes
                data_pos[dty].append([i+1, i+1+int(data[i-1])])
        return data_pos, len(data)

    #----------------------------------------------------------------------------
    def prepare_helper_arrays(self, blocks, nblocks):                  # fmt_file
    #----------------------------------------------------------------------------
        block_slices = nparray(blocks.slice)
        slices = nparray(block_slices)
        stride = nparray( [[blocks.stride[blocks.type[i]],] for i in range(blocks.num['chunks'])] )
        heads = deepcopy(blocks.head)
        tails = deepcopy(blocks.tail)
        types = deepcopy(blocks.type)
        for n in range(1,nblocks):
            slices = npappend(slices, n*stride+block_slices, axis=0)
            heads += blocks.head
            tails += blocks.tail
            types += blocks.type
        return heads, slices, tails, types

    #----------------------------------------------------------------------------
    def fast_convert(self, nblocks=1, ext='.UNRST', init_key='SEQNUM', rename_duplicate=True,
                rename_key=None, echo=False, progress=lambda x:None, cancel=lambda:None):  # fmt_file 
    #--------------------------------------------------------------------------------
        outfile = self.name.with_suffix(ext)
        with open(self.name) as f:
            with mmap(f.fileno(), length=0, offset=0, access=ACCESS_READ) as filemap:
                # prepare 
                blocks = self.get_blocks(filemap, init_key, rename_duplicate, rename_key)
                unit_format = ''.join(blocks.format) 
                data_pos, pos_stride = self.get_data_pos(filemap, blocks.size['blocks'])
                N = int(len(filemap)/blocks.size['blocks'])
                progress(-N)
                heads, slices, tails, types = self.prepare_helper_arrays(blocks, nblocks)
                # process file
                with open(outfile, 'wb') as out:
                    a = 0
                    end = len(filemap)
                    finished = False
                    n = 0
                    while not finished:
                        # convert from string to array datatype
                        b = a + nblocks*blocks.size['blocks']
                        if b>end:
                            b = end
                            nblocks = int((b-a)/blocks.size['blocks'])
                            finished = True
                        data = filemap[a:b].split()
                        a = b
                        buffer = self.string_to_num(nblocks, blocks, data_pos, data, pos_stride)
                        data_chunks = ((*heads[i], *buffer[types[i]][slices[i][0]:slices[i][1]], tails[i]) for i in range(nblocks*blocks.num['chunks']))
                        out.write(pack(ENDIAN+nblocks*unit_format, *[x for y in data_chunks for x in y]))
                        n += nblocks
                        progress(n)
                        cancel()
        return outfile

    #----------------------------------------------------------------------------
    def string_to_num(self, nblocks, blocks, data_pos, data, pos_stride): # fmt_file
    #----------------------------------------------------------------------------
        buffer = {}
        # Loop over all datatypes (INTE, REAL, DOUB, etc.)
        for dtyp in blocks.stride.keys():
            # dtype=datatype[dtyp.decode()]
            dtype = DTYPE[dtyp].nptype
            buf = []
            for i,j in data_pos[dtyp]:
                for nb in range(nblocks):
                    buf.append(data[i+nb*pos_stride:j+nb*pos_stride])
                    #print(f'{dtyp}: data[{i}]={data[i]}, data[{j}]={data[j-1]}')
            try: 
                buffer[dtyp] = nparray([x for y in buf for x in y], dtype=dtype)
            except ValueError:
                buffer[dtyp] = nparray([x.decode().replace('D','E') for y in buf for x in y], dtype=dtype)
        return buffer


            
# #@njit
# #----------------------------------------------------------------------------
# def numba_convert_buffer(buffer, dtyp):
# #----------------------------------------------------------------------------
#     for i in range(len(buffer)):
#         buffer[i] = numba_str_to_num(buffer[i], dtyp)
#     return buffer
#     # if dtyp=='INTE':
#     #     return nparray(buffer, dtype=int)
#     # if dtyp=='REAL':
#     #     return nparray(buffer, dtype=float)
#     # if dtyp=='DOUB':
#     #     return nparray(buffer, dtype=float)
#     # if dtyp=='LOGI':
#     #     return nparray(buffer, dtype=bool)

# #@njit
# #----------------------------------------------------------------------------
# def numba_str_to_num(string, dtyp):
# #----------------------------------------------------------------------------
#     #print(string)
#     if dtyp=='INTE':
#         return int(string)
#     if dtyp=='REAL':
#         return float(string)
#     if dtyp=='DOUB':
#         return float(string[:15]+'E'+string[17:])
#     if dtyp=='LOGI':
#         return string=='T'

    
#====================================================================================
class RSM_block:                                                          # RSM_block
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, var, unit, well, data):                            # RSM_block
    #--------------------------------------------------------------------------------
        self.var = var
        self.unit = unit
        self.well = well
        self.data = data
        self.nrow = len(self.data)
        
    #--------------------------------------------------------------------------------
    def get_data(self):                                                   # RSM_block
    #--------------------------------------------------------------------------------
        for col,(v,u,w) in enumerate(zip(self.var, self.unit, self.well)):
            yield (v, u, w, [self.data[row][col] for row in range(self.nrow)])
        
        
#====================================================================================
class RSM_file:                                                            # RSM_file
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, filename):
    #--------------------------------------------------------------------------------
        self.name = Path(filename)
        self.fh = None
        self.tag = '1'
        self.nrow = self.block_length()-10
        self.colpos = None
        
    #--------------------------------------------------------------------------------
    def get_data(self):                                                    # RSM_file
    #--------------------------------------------------------------------------------
        if not self.name.is_file():
            return ()
        with open(self.name) as self.fh:
            for line in self.fh:
                # line is now at the tag-line
                for block in self.read_block():
                    for data in block.get_data():
                        yield data
                            
    #--------------------------------------------------------------------------------
    def read_block(self):                                                  # RSM_file
    #--------------------------------------------------------------------------------
        self.skip_lines(3)
        var, unit, well = self.read_var_unit_well()
        self.skip_lines(2)
        data = self.read_data(ncol=len(var))
        yield RSM_block(var, unit, well, data)
                
    #--------------------------------------------------------------------------------
    def skip_lines(self, n):                                               # RSM_file
    #--------------------------------------------------------------------------------
        for i in range(n): 
            next(self.fh) 
        
    #--------------------------------------------------------------------------------
    def read_data(self, ncol=None):                                        # RSM_file
    #--------------------------------------------------------------------------------
        data = [None]*self.nrow        
        for l in range(self.nrow):
            line = next(self.fh)
            cols = line.rstrip().split()
            if len(cols)<ncol:
                # missing column entries
                cols = self.get_columns_by_position(line=line)
            data[l] = [float_or_str(c) for c in cols]
        return data

    
    #--------------------------------------------------------------------------------
    def get_columns_by_position(self, line=None):                          # RSM_file
    #--------------------------------------------------------------------------------
        '''
        Return None if column is empty
        '''
        n = len(self.colpos)
        words = [None]*n
        for i in range(n-1):
            a, b = self.colpos[i], self.colpos[i+1]
            words[i] = line[a:b].strip() or None
        return words

    
    #--------------------------------------------------------------------------------
    def read_var_unit_well(self):                                          # RSM_file
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def block_length(self):                                                # RSM_file
    #--------------------------------------------------------------------------------
        with open(self.name) as fh: 
            nb, n = 0, 0 
            for line in fh: 
                n += 1 
                if line[0]==self.tag: 
                    nb += 1 
                    if nb==2: 
                        return int(n) 


