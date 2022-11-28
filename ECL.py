
# -*- coding: utf-8 -*-

DEBUG = False 
ENDIAN = '>'  # Big-endian

from dataclasses import dataclass
from itertools import repeat
from operator import itemgetter
from pathlib import Path
from numpy import zeros, int32, float32, float64, bool_ as np_bool, array as nparray, append as npappend 
from mmap import ACCESS_WRITE, mmap, ACCESS_READ
from re import IGNORECASE, MULTILINE, finditer, compile
from copy import deepcopy
from collections import namedtuple
from datetime import datetime, timedelta
from struct import unpack, pack, error as struct_error
#from numba import njit, jit
from .utils import flatten, flatten_all, grouper, list2text, pairwise, remove_chars, remove_comments, safezip, list2str, float_or_str, matches, split_by_words, string_chunks, triplewise

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
        

#====================================================================================
class unfmt_block:
    #
    # Block of unformatted Eclipse data
    #
#====================================================================================
    #  | h e a d e r  |   d a t a     |
    #  |4i|8s|4i|4s|4i|4i|1000 data|4i| 
    #  |    24 bytes  |
    #  |              |4i|8d| 

    #--------------------------------------------------------------------------------
    def __init__(self, key=b'', length=0, type=b'', start=0, end=0, data=None, file=None):
    #--------------------------------------------------------------------------------
        self._key = key
        self._length = length
        self._type = type
        self._dtype = DTYPE[type]
        self._data = data
        self._file = file
        self._startpos = start
        self._end = end
        self._data_start = start + 24
        DEBUG and print(f'Creating {self}')

    #--------------------------------------------------------------------------------
    def __str__(self):                                                  # unfmt_block
    #--------------------------------------------------------------------------------
        return f'key={self.key():8s}, type={self.type():4s}, bytes={self.bytes():8d}, length={self.length():8d}, start={self._startpos:8d}, end={self._end:8d}'

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                  # unfmt_block
    #--------------------------------------------------------------------------------
        return f'<ECL.unfmt_block, {self}>'

    #--------------------------------------------------------------------------------
    def __contains__(self, key):                                        # unfmt_block
    #--------------------------------------------------------------------------------
        return self.key() == key


    #--------------------------------------------------------------------------------
    def __del__(self):                                                  # unfmt_block
    #--------------------------------------------------------------------------------
        DEBUG and print(f'Deleting {self}')


    #--------------------------------------------------------------------------------
    def length(self):                                                   # unfmt_block
    #--------------------------------------------------------------------------------
        return self._length
        #return self._length*(self._type==b"CHAR" and 8 or 1)


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
            s += f' block of {self._length*self._dtype.size} bytes' 
            s += f' holding {self.length()} {self._dtype.name}'
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
    def data(self, *index, raise_error=False, unwrap_tuple=True, nchar=1, strip=False):    # unfmt_block
    #--------------------------------------------------------------------------------
        ### Abort if no data to return
        if self._length == 0:
            return ()
        ### Return all data if no argument given
        if index == ():
            index = ((0, self._length),)
            unwrap_tuple = False
        ### Fix negative positions, and create tuple if not tuple 
        fix_lim = lambda x: x+self._length if x < 0 else x
        index = [[fix_lim(ii) for ii in i] if isinstance(i,(tuple,list)) else [fix_lim(i)+a for a in (0,1)] for i in index]
        ### Out of index error
        if any(i>self._length or i<0 for i in flatten_all(index)):
            raise IndexError(f'index out of range for {self.key()}-block of length {self._length}')
        ### Fix CHAR data for strings > 8 char (nchar > 1)
        if nchar > 1 and self._type == b'CHAR':
            index = [[min(i*nchar,self._length) for i in ind] for ind in index]
        ### List of data chunk start positions, including the 4 byte size int before and after 
        chunk_limits = list(range(self._data_start, self._end, self._dtype.max*self._dtype.size+8))
        chunk_pos = lambda x: chunk_limits[ slice( *(1+(i//self._dtype.max) for i in x) ) ]
        byte_pos = lambda x: self._data_start + x*self._dtype.size + 8*(x//self._dtype.max) + 4
        ### Modify byte_pos by -4/+4 at start/end to match chunk-limits 
        limits = ([byte_pos(a)-4]+chunk_pos((a,b))+[byte_pos(b)+4] for a,b in index)
        ### Compensate for the 4-byte size int
        data_chunks = ([a+4,b-4] for lim in limits for a,b in pairwise(lim))
        try:
            ### CHAR data is an 8 character string
            N = sum(j-i for i,j in index)*(self._type == b'CHAR' and 8 or 1)
            values = unpack(ENDIAN+f'{N}{self._dtype.unpack}', b''.join(self._data[a:b] for a,b in data_chunks))
        except struct_error:
            if raise_error:
                raise SystemError(f'ERROR Unable to read {self.key()} from {self._file.name}')
            return None
        ### Decode string data
        if self._type == b'CHAR':
            values = tuple(string_chunks(values[0].decode(), 8*nchar, strip=strip))
        ### Return value instead of single-value tuple
        if unwrap_tuple and len(values) == 1:
            return values[0]
        return values



#====================================================================================
@dataclass #(init=False)
class keypos:
#====================================================================================
    key: str = ''
    #pos: int = 0
    pos: tuple = (0,)
    name: str = ''

    # def __init__(self, key='', *pos, name=''):
    #     self.key = key
    #     self.pos = pos
    #     self.name = name
    #     print('pos', self.pos)

#====================================================================================
class File:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, filename, suffix, role='', ignore_case=False):          # File
    #--------------------------------------------------------------------------------
        self.file = suffix and Path(filename).with_suffix(suffix) or Path(filename)
        if ignore_case and not self.file.is_file():
            ### Create case-insensitive pattern, e.g. '.[sS][cC][hH]'
            pattern = '*.'+'['+']['.join(c.lower()+c.upper() for c in self.file.suffix[1:])+']'
            self.file = next(filename.parent.glob(pattern), self.file)
        self.role = role.rstrip().lstrip()
        DEBUG and print(f'Creating {repr(self)}')

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                        # File
    #--------------------------------------------------------------------------------
        return f'<ECL.File, file={self.file}, role={self.role}>'

    #--------------------------------------------------------------------------------
    def __str__(self):                                                         # File
    #--------------------------------------------------------------------------------
        return f'{self.file.name}'

    #--------------------------------------------------------------------------------
    def __del__(self):                                                         # File
    #--------------------------------------------------------------------------------
        DEBUG and print(f'Deleting {repr(self)}')

    #--------------------------------------------------------------------------------
    def binarydata(self, raise_error=False):                                    # File
    #--------------------------------------------------------------------------------
        ### Open as binary file to avoid encoding errors
        if self.is_file():
            with open(self.file, 'rb') as f:
                return f.read()
        if raise_error:
            raise SystemError(f'File {self} does not exist')
        return b''
 
    #--------------------------------------------------------------------------------
    def delete(self, raise_error=False, echo=False):                           # File
    #--------------------------------------------------------------------------------
        try:
            self.file.unlink(missing_ok=True)
        except (PermissionError, FileNotFoundError) as e:
            if raise_error:
                raise SystemError(f'Unable to delete {self}: {e}')
            echo and print(f'Deleted {self}')


    #--------------------------------------------------------------------------------
    def is_file(self):                                                         # File
    #--------------------------------------------------------------------------------
        return self.file.is_file()

    #--------------------------------------------------------------------------------
    def with_name(self, file):                                                 # File
    #--------------------------------------------------------------------------------
        return (self.file.parent/file).resolve()

        
    #--------------------------------------------------------------------------------
    def exists(self, raise_error=False):                                       # File
    #--------------------------------------------------------------------------------
        if self.file.is_file():
            return True
        if raise_error:
            raise SystemError(f'ERROR {" ".join((self.role, self.file.name)).lstrip()} is missing in folder {self.file.parent}')
        return False


    #--------------------------------------------------------------------------------
    def size(self):                                                            # File
    #--------------------------------------------------------------------------------
        return self.file.is_file() and self.file.stat().st_size or -1


    #--------------------------------------------------------------------------------
    def name(self):                                                            # File
    #--------------------------------------------------------------------------------
        return self.file.name



#====================================================================================
class unfmt_file(File):
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, filename, suffix, **kwargs):                      # unfmt_file
    #--------------------------------------------------------------------------------
        super().__init__(filename, suffix, **kwargs)
        self.endpos = 0
        DEBUG and print(f'Creating {self}')

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                  # unfmt_file
    #--------------------------------------------------------------------------------
        return f'<ECL.unfmt_file, {self}, endpos={self.endpos}>'

    #--------------------------------------------------------------------------------
    def at_end(self, raise_error=False):                                 # unfmt_file
    #--------------------------------------------------------------------------------
        return self.endpos == self.size()

    #--------------------------------------------------------------------------------
    def offset(self, raise_error=False):                                 # unfmt_file
    #--------------------------------------------------------------------------------
        return self.size() - self.endpos

    #--------------------------------------------------------------------------------
    def blocks(self, only_new=False, start=None):                        # unfmt_file
    #--------------------------------------------------------------------------------
        if not self.is_file():
            return False
        startpos = 0
        if only_new:
            startpos = self.endpos
        if start:
            startpos = start
        if self.size() - startpos < 24: # Header is 24 bytes
            return False
        with open(self.file, mode='rb') as file:
            with mmap(file.fileno(), length=0, access=ACCESS_READ) as data:
                size = data.size()
                pos = startpos
                while pos < size:
                    start = pos
                    ### Header
                    try:
                        ### Header is 24 bytes, we skip int of length 4 before and after
                        key, length, type = unpack(ENDIAN+'8si4s', data[pos+4:pos+20])
                        ### Value array
                        bytes = length*DTYPE[type].size + 8 * -(-length//DTYPE[type].max) # -(-a//b) is the ceil-function
                        pos += 24 + bytes
                    except (ValueError, struct_error): 
                        return False
                    self.endpos = pos
                    yield unfmt_block(key=key, length=length, type=type, start=start, end=pos, 
                                      data=data, file=self.file)
    
    # #--------------------------------------------------------------------------------
    # def blocks(self, only_new=False, start=None):                        # unfmt_file
    # #--------------------------------------------------------------------------------
    #     ' Read blocks without mmap'
    #     if not self.is_file():
    #         return False
    #     startpos = 0
    #     if only_new:
    #         startpos = self.endpos
    #     if start:
    #         startpos = start
    #     if self.size() - startpos < 24: # Header is 24 bytes
    #         return False
    #     size = self.size()
    #     with open(self.file, mode='rb') as file:
    #         pos = startpos
    #         while pos < size:
    #             start = pos
    #             file.seek(start)
    #             ### Header
    #             try:
    #                 ### Header is 24 bytes, we skip int of length 4 before and after
    #                 _, key, length, type = unpack(ENDIAN+'i8si4s', file.read(20))
    #                 ### Value array
    #                 bytes = length*DTYPE[type].size + 8 * -(-length//DTYPE[type].max) # -(-a//b) is the ceil-function
    #                 pos += 24+bytes
    #             except (ValueError, struct_error): 
    #                 return False
    #             self.endpos = pos
    #             yield unfmt_block(key=key, length=length, type=type, start=start, end=pos, 
    #                                 data=file, file=self.file)
    

    #--------------------------------------------------------------------------------
    def tail_blocks(self):                                               # unfmt_file
    #--------------------------------------------------------------------------------
        if not self.is_file() or self.size() < 24: # Header is 24 bytes
            return
        with open(self.file, mode='rb') as file:
            with mmap(file.fileno(), length=0, access=ACCESS_READ) as data:
                # Goto end of file
                data.seek(0, 2)
                while data.tell() > 0:
                    end = data.tell()
                    ### Header
                    ### Rewind until we find a header
                    while data.tell() > 0:
                        try:
                            data.seek(-4, 1)
                            size = unpack(ENDIAN+'i',data.read(4))[0]
                            data.seek(-4-size, 1)
                            # if self.is_header(data, size, data.tell()):
                            pos = data.tell()
                            ### Check if this is a header
                            if size == 16 and data[pos+12:pos+16] in DTYPE.keys():
                                start = data.tell()-4
                                key, length, type = unpack(ENDIAN+'8si4s', data.read(16))
                                data.seek(4, 1)
                                break
                            else:
                                data.seek(-4, 1)
                        except (ValueError, struct_error):
                            return None
                    ### Value array
                    #data_start = data.tell()
                    data.seek(start, 0)
                    yield unfmt_block(key=key, length=length, type=type, start=start, end=end, 
                                    data=data, file=self.file)


    #--------------------------------------------------------------------------------
    def get(self, *var_list, N=0, stop=(), raise_error=True, **kwargs):  # unfmt_file
    #--------------------------------------------------------------------------------
        #print(var_list, N, kwargs)
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
        #print(var_pos)
        values = {v:[] for v in var_list}
        size = lambda : (len(v) for v in values.values())
        for b in blocks(**kwargs):
            if b.key() in var_pos.keys():
                for var, pos in var_pos[b.key()]:
                    #values[var].append( b.data()[pos] )
                    values[var].append( b.data(*pos) )
            if N and set(size()) == set([N]): 
                break
            if stop and stop[1] == values[stop[0]][-1] and len(set(size())) == 1:
                break
        if not all(values.values()):
            if raise_error:
                raise SystemError('ERROR Unable to read ' + list2str(var_pos.keys(), sep="'") + f' from {self.file.name}')
            return []
        return list(values.values())        


    #--------------------------------------------------------------------------------
    def sections(self, begin=0, check_sync=lambda *x:0, init_key=None, start_before=None, 
                 start_after=None, end_before=None, end_after=None):    # unfmt_file
    #--------------------------------------------------------------------------------
        if not self.exists():
            raise SystemError(f'ERROR File {self.file} not found') 
        inside = False
        step = None
        with open(self.file, 'rb') as file:
            for block in self.blocks():
                key = block.key()
                step = check_sync(block, step)
                if inside and key in (end_before, end_after):
                    inside = False
                    if key == end_before:
                        end_pos = block.start()
                    else:
                        end_pos = block.end()
                    yield (n, file.read(end_pos-start_pos))
                if not inside and key in (start_before, start_after):
                    if step < begin:
                        continue
                    inside = True
                    n = step
                    if key == start_before:
                        start_pos = block.start()
                    else:
                        start_pos = block.end()
                    file.seek(start_pos)
            if inside and end_before==init_key:
                yield (n, file.read(self.size()-start_pos))


    #--------------------------------------------------------------------------------
    def create(self, sections=None, progress=lambda x:None, cancel=lambda:None): # unfmt_file
    #--------------------------------------------------------------------------------
        return_value = False
        with open(self.file, 'wb') as out_file, safezip(*sections) as zipper:
            ### Get data from the section generators
            for step_data in zipper:
                steps = []
                for step, data in step_data:
                    out_file.write(data)
                    steps.append(step)
                if len(set(steps)) > 1:
                    raise SystemError(f'ERROR Sections are not synchronized in unfmt_file.create(): {steps}')
                progress(steps[0])
                cancel()
            return_value = self.file
        return return_value



#====================================================================================
class DATA_file(File):
#====================================================================================
    # Sections
    section_names = ['RUNSPEC','GRID','EDIT','PROPS' ,'REGIONS', 'SOLUTION','SUMMARY','SCHEDULE','OPTIMIZE']
    # Global keywords
    global_kw = ['COLUMNS','DEBUG','DEBUG3','ECHO','END', 'ENDINC','ENDSKIP','SKIP','SKIP100','SKIP300','EXTRAPMS','FORMFEED','GETDATA',
                'INCLUDE','MESSAGES','NOECHO','NOWARN','WARN']
    # Common keywords
    common_kw = ['TITLE','CART','DIMENS','FMTIN','FMTOUT','GDFILE',
                'FMTOUT','UNIFOUT','UNIFIN','OIL','WATER','GAS','VAPOIL','DISGAS','FIELD','METRIC','LAB','START','WELLDIMS','REGDIMS','TRACERS',
                'NSTACK','TABDIMS','NOSIM','GRIDFILE','DX','DY','DZ','PORO','BOX','PERMX','PERMY','PERMZ','TOPS',
                'INIT','RPTGRID','PVCDO','PVTW','DENSITY','PVDG','ROCK','SPECROCK','SPECHEAT','TRACER','TRACERKP',
                'TRDIFPAR','TRDIFIDE','SATNUM','FIPNUM','TRKPFPAR','TRKPFIDE','RPTSOL','RESTART','PRESSURE','SWAT',
                'SGAS','RTEMPA','TBLKFA1','TBLKFIDE','TBLKFPAR','FOPR','FOPT','FGPR','FGPT','FWPR','FWPT','FWCT','FWIR',
                'FWIT','FOIP','ROIP','WTPCHEA','WOPR','WWPR','WWIR','WBHP','WWCT','WOPT','WWIT','WTPRA1','WTPTA1','WTPCA1',
                'WTIRA1','WTITA1','WTICA1','CTPRA1','CTIRA1','FOIP','ROIP','FPR','TCPU','TCPUTS','WNEWTON','ZIPEFF','STEPTYPE',
                'NEWTON','NLINEARP','NLINEARS','MSUMLINS','MSUMNEWT','MSUMPROB','WTPRPAR','WTPRIDE','WTPCPAR','WTPCIDE','RUNSUM',
                'SEPARATE','WELSPECS','COMPDAT','WRFTPLT','TSTEP','DATES','SKIPREST','WCONINJE','WCONPROD','WCONHIST','WTEMP','RPTSCHED',
                'RPTRST','TUNING','READDATA', 'ROCKTABH','GRIDUNIT','NEWTRAN','MAPAXES','EQLDIMS','ROCKCOMP','TEMP',
                'GRIDOPTS','VFPPDIMS','VFPIDIMS','AQUDIMS','SMRYDIMS','CPR','FAULTDIM','MEMORY','EQUALS','MINPV',
                'COPY','MULTIPLY']

    #--------------------------------------------------------------------------------
    def __init__(self, file, check=False, **kwargs):      # Input_file
    #--------------------------------------------------------------------------------
        #print(f'Input_file({file}, check={check}, read={read}, reread={reread}, include={include})')
        super().__init__(file, Path(file).suffix or '.DATA', role='Eclipse input-file', **kwargs)
        self._data = None
        # self._sections = None
        self._checked = False
        #self._reread = reread
        #if read or include:
        #    self._data = self.without_comments()
        getter = namedtuple('getter', 'section default convert pattern')
        self._getter = {'TSTEP'   : getter('SCHEDULE', (),      self._convert_float,  r'\bTSTEP\b\s+([0-9*.\s]+)/\s*'),
                        'START'   : getter('RUNSPEC',  (0,),     self._convert_date,   r'\bSTART\b\s+(\d+\s+\'*\w+\'*\s+\d+)'),
                        'DATES'   : getter('SCHEDULE', (),      self._convert_date,   r'\bDATES\b\s+((\d{1,2}\s+\'*\w{3}\'*\s+\d{4}\s*\s*/\s*)+)/\s*'), 
                        'RESTART' : getter('SOLUTION', ('', 0), self._convert_file,   r"\bRESTART\b\s+('*[a-zA-Z0-9_./\\-]+'*\s+[0-9]+)\s*/"),
                        'SUMMARY' : getter('SUMMARY',  (),      self._convert_string, r'\bSUMMARY\b((\s*\w+\s*/*\s*)+)\bSCHEDULE\b'),
                        'WELSPECS': getter('SCHEDULE', (),      self._convert_string, r'\bWELSPECS\b((\s+\'*[A-Za-z0-9_/-]+?.*/\s*)+/)')}
        check and self.check() 
        #(check or include) and self.check() 
        #include and self.with_includes(section=include)
        # Alt. DATES: r'\bDATES\b\s+(\d+\s+\'*\w+\'*\s+\d+)\s*/\s*/\s*')
        # 'INCLUDE' : getter(None,       [''],    self._convert_file,   r"\bINCLUDE\b\s+'*([a-zA-Z0-9_./\\-]+)'*\s*/"), 
        # 'GDFILE'  : getter(None,       [''],    self._convert_file,   r"\bGDFILE\b\s+'*([a-zA-Z0-9_./\\-]+)'*\s*/"), 

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                   # Input_file
    #--------------------------------------------------------------------------------
        return f'<ECL.DATA_file {self.file}>'

    # #--------------------------------------------------------------------------------
    # def __add__(self, other):                                            # Input_file
    # #--------------------------------------------------------------------------------
    #     for obj in (self, other):
    #         if not obj._data:
    #             obj._data = obj.without_comments()
    #     self._data += other._data
    #     return self

    # #--------------------------------------------------------------------------------
    # def __len__(self):                                                   # Input_file
    # #--------------------------------------------------------------------------------
    #     return self._data and len(self._data) or 0

    # #--------------------------------------------------------------------------------
    # def without_comments(self):                                          # Input_file
    # #--------------------------------------------------------------------------------
    #     return remove_comments(self.file, comment='--', end='END')

    #--------------------------------------------------------------------------------
    def __call__(self):                                                  # Input_file
    #--------------------------------------------------------------------------------
        self._data = self.binarydata()
        return self

    #--------------------------------------------------------------------------------
    def remove_comments(self, data=None):                                    # Input_file
    #--------------------------------------------------------------------------------
        #print(list(data))
        data = data or self.binarydata()
        #print([d.split(b'\n') for d in data])
        lines = (l for d in data for l in d.split(b'\n'))
        text = (l.split(b'--')[0].strip() for l in lines)
        text = b'\n'.join(t for t in text if t).decode()
        self._data = text and text+'\n' or ''
        return self

    #--------------------------------------------------------------------------------
    def check(self, include=True):                                       # Input_file
    #--------------------------------------------------------------------------------
        self._checked = True
        ### Check if file exists
        self.exists(raise_error=True)        
        ### Check if included files exists
        if include and (missing := [f for f in self.include_files() if not f.is_file()]):
            raise SystemError(f'ERROR {list2text([f.name for f in missing])} included from {self} is missing in folder {missing[0].parent}')
        return True

    #--------------------------------------------------------------------------------
    def contains(self, key, regex, comments=False):                         # Input_file
    #--------------------------------------------------------------------------------
        #print(list(self.include_files(self._data)))        
        data = self.matching(key)
        if not comments:
            self.remove_comments(data)
        else:
            self._data = b''.join(data).decode()
        #print(self._data)
        return compile(regex).search(self._data)

    #--------------------------------------------------------------------------------
    def is_empty(self):                                                  # Input_file
    #--------------------------------------------------------------------------------
        return self.without_comments() == b''
        #return self.data().strip() == ''

    # #--------------------------------------------------------------------------------
    # def data(self):                                                     # Input_file
    # #--------------------------------------------------------------------------------
    #     #print('data',self)
    #     if not self._data or self._reread:
    #         if self.is_file():
    #             self._data = self.without_comments()
    #         else:
    #             return ()
    #     return self._data

    # #--------------------------------------------------------------------------------
    # def without_comments(self):                                                     # Input_file
    # #--------------------------------------------------------------------------------
    #     if self.is_file():
    #         return self.remove_comments(self.binarydata())
    #     return ()

    #--------------------------------------------------------------------------------
    def lines(self):                                                     # Input_file
    #--------------------------------------------------------------------------------
        return (line for line in self.data().split('\n') if line)

    #--------------------------------------------------------------------------------
    def matching(self, *keys):
    #--------------------------------------------------------------------------------
        keys = [key.encode() for key in keys]
        if any(key in self._data for key in keys):
            yield self._data
        for file in self.include_files(self._data):
            inc_data = File(file,'').binarydata() 
            if any(key in inc_data for key in keys):
                yield inc_data

    # #--------------------------------------------------------------------------------
    # def _includes(self, data:bytes):
    # #--------------------------------------------------------------------------------
    #     regex = rb"(\bINCLUDE\b|\bGDFILE\b)\s*(--)?.*\s+'*(?P<file>[a-zA-Z0-9_./\\-]+)'*\s*/"
    #     return (m.group('file').decode() for m in compile(regex).finditer(data))

    #--------------------------------------------------------------------------------
    def include_files(self, data:bytes=None):                           # Input_file
    #--------------------------------------------------------------------------------
        data = data or self.binarydata()
        #regex = rb"(\bINCLUDE\b|\bGDFILE\b)\s*(--)?.*\s+'*(?P<file>[a-zA-Z0-9_./\\-]+)'*\s*/"
        regex = rb"^[ \t]*(?:\bINCLUDE\b|\bGDFILE\b)(?:.*--.*\s*|\s*)*'*(.*?)['\s]*/\s*(?:--.*)*$"
        files = (m.group(1).decode() for m in compile(regex, flags=MULTILINE).finditer(data))
        for file in files:
            new_file = self.with_name(file)
            yield new_file
            file_data = File(new_file,'').binarydata()
            if b'INCLUDE' in file_data:
                for inc in self.include_files(file_data):
                    yield inc

    # #--------------------------------------------------------------------------------
    # def include_file(self, suffix):                                      # Input_file
    # #--------------------------------------------------------------------------------
    #     ''' Return first included file with given suffix (case-insensitive) or None '''
    #     return next((f for f in self.get('INCLUDE') if suffix in str(f).lower()), None)


    # #--------------------------------------------------------------------------------
    # def include_files(self):                                             # Input_file
    # #--------------------------------------------------------------------------------
    #     '''
    #     Search recursively for include files in the .DATA-file
    #     Return list of full paths
    #     '''
    #     for file in self._include_files_recursive(self.file):
    #         yield file
    #     for file in (f for f in self.get('GDFILE') if f != ''):
    #         yield file


    # #--------------------------------------------------------------------------------
    # def _include_files_recursive(self, file):                           # Input_file
    # #--------------------------------------------------------------------------------
    #     new_files = (f for f in DATA_file(file).get('INCLUDE') if f != '')
    #     for new_file in new_files:
    #         yield new_file
    #         for inc in self._include_files_recursive(new_file):
    #             yield inc

    #--------------------------------------------------------------------------------
    def tsteps(self, start=None, negative_ok=False, missing_ok=False, pos=False, skiprest=False):     # Input_file
    #--------------------------------------------------------------------------------
        'Return timesteps, if DATES are present they are converted to timesteps'
        #self.with_includes(section='SCHEDULE', raise_error=False)
        #dates = self.get('DATES', pos=True)
        _start, tsteps, dates = self.get('START','TSTEP','DATES', pos=True)
        #tsteps = []
        if skiprest:
            tsteps = []
            negative_ok = True
        #else:
        #    tsteps = self.get('TSTEP', pos=True)
        times = sorted(dates+tsteps, key=itemgetter(1))
        #start = start or self.get('START')[0]
        start = start or _start[0][0]
        if not start:
            raise SystemError('ERROR Missing start-date in DATA_file.tsteps()')
        #tsteps = list(self._days(times, start=start))
        tsteps = tuple(self._days(times, start=start))
        ## Checks
        if not negative_ok and any(t<=0 for t,_ in tsteps):
            raise SystemError(f'ERROR Zero or negative timestep in {self} (check if TSTEP or RESTART oversteps a DATES keyword)')
        #if not missing_ok and tsteps == []:
        if not missing_ok and not tsteps:
            raise SystemError(f'ERROR No TSTEP or DATES in {self} (or the included files)')
        return pos and tsteps or tuple(next(zip(*tsteps))) # Do not return positions
        #return pos and tsteps or [t for t,_ in tsteps]

    #--------------------------------------------------------------------------------
    def _days(self, time_pos, start=None):                           # Input_file
    #--------------------------------------------------------------------------------
        'Return relative timestep in days given a timestep or a datetime'
        last_date = start
        for t,p in time_pos:
            if isinstance(t, datetime):
                dt = t
            else:
                dt = last_date + timedelta(hours=t*24)
            yield (dt-last_date).total_seconds()/86400, p
            last_date = dt
            
    #--------------------------------------------------------------------------------
    def _convert_string(self, values, key, raise_error=False):             # Input_file
    #--------------------------------------------------------------------------------
        ret = [v for val in values for v in val.split('\n') if v and v != '/']
        return (ret,)
        #return [ret]

    #--------------------------------------------------------------------------------
    def _convert_float(self, values, key, raise_error=False):            # Input_file
    #--------------------------------------------------------------------------------
        mult = lambda x, y : list(repeat(float(y),int(x))) # Process x*y statements
        values = ([mult(*n.split('*')) if '*' in n else [float(n)] for n in v.split()] for v in values)
        values = tuple(flatten(v) for v in values)
        return values or self._getter[key].default


    #--------------------------------------------------------------------------------
    def _convert_date(self, dates, key, raise_error=False):              # Input_file
    #--------------------------------------------------------------------------------
        ### Remove possible quotes
        ### Extract groups of 3 from the dates strings 
        dates = (grouper(remove_chars("'/\n", v).split(), 3) for v in dates)
        dates = tuple([datetime.strptime(' '.join(d), '%d %b %Y') for d in date] for date in dates)
        return dates or self._getter[key].default


    #--------------------------------------------------------------------------------
    def _convert_file(self, values, key, raise_error=True):              # Input_file
    #--------------------------------------------------------------------------------
        'Return full path of file'
        ### Remove quotes and backslash
        values = (val.replace("'",'').replace('\\','/').split() for val in values)
        ### Unzip values in a files (always) and numbers lists (only for RESTART)
        unzip = zip(*values)
        files = ([(self.file.parent/file).resolve()] for file in next(unzip))
        numbers = [[float(num)] for num in next(unzip, ())]
        files = numbers and tuple([f[0],n[0]] for f,n in zip(files, numbers)) or tuple(files)
        ### Add suffix for RESTART keyword
        if key == 'RESTART' and files:
            files[0][0] = files[0][0].with_suffix('.UNRST')
        return files or self._getter[key].default

    # #--------------------------------------------------------------------------------
    # def get(self, *keywords, **kwargs):                                  # Input_file
    # #--------------------------------------------------------------------------------
    #     #print('get', keywords, kwargs)
    #     #[self.with_includes(section) for key in keywords if (section :=self._getter[key].section)]
    #     ret = [self._get(key, **kwargs) for key in keywords]
    #     if len(ret) == 1:
    #         return ret[0]
    #     return ret

    # #--------------------------------------------------------------------------------
    # def _get(self, keyword, raise_error=False, pos=False, include=True): # Input_file
    # #--------------------------------------------------------------------------------
    #     #print('_get', keyword)
    #     keyword = keyword.upper()
    #     error_msg = f'ERROR Keyword {keyword} not found in {self.file}'
    #     if not keyword in self._getter.keys():
    #         if raise_error:
    #             raise SystemError(f'ERROR Missing get-pattern for {keyword} in Input_file')
    #         return []
    #     key = self._getter[keyword]
    #     if not self._data or self._reread:
    #         if not self.exists(raise_error=raise_error):
    #             return key.default
    #         if keyword.encode() in self.binarydata(): 
    #             self._data = self.without_comments()
    #         else:
    #             if raise_error:
    #                 raise SystemError(error_msg)
    #             return key.default
    #     match_list = compile(key.pattern).finditer(self._data)
    #     val_span = tuple((m.group(1), m.span()) for m in match_list) 
    #     if not val_span:
    #         return key.default
    #     values, span = zip(*val_span)
    #     values = key.convert(values, keyword, raise_error=raise_error)
    #     if pos:
    #         values = (tuple(zip(v,repeat(p))) for v,p in zip(values, span))
    #     if raise_error and not values:
    #         raise SystemError(error_msg)
    #     return flatten(values) 

    #--------------------------------------------------------------------------------
    def get(self, *keywords, raise_error=False, pos=False): # Input_file
    #--------------------------------------------------------------------------------
        #print('get', keywords)
        FAIL = len(keywords)*((),)
        if not self.exists(raise_error=raise_error):
            return FAIL
        keywords = [key.upper() for key in keywords]
        getters = [self._getter.get(key) for key in keywords]
        if missing:=[k for g,k in zip(getters, keywords) if not g]:
            if raise_error:
                raise SystemError(f'ERROR Missing get-pattern for {list2text(missing)} in DATA_file')
            return FAIL
        names = set([g.section for g in getters])
        self.remove_comments( self.section(*names).matching(*keywords) )
        error_msg = f'ERROR Keyword {list2text(keywords)} not found in {self.file}'
        if not self._data:
            if raise_error:
                raise SystemError(error_msg)
            return FAIL 
        result = ()
        for keyword, getter in zip(keywords, getters):
            #key = self._getter[keyword]
            match_list = compile(getter.pattern).finditer(self._data)
            val_span = tuple((m.group(1), m.span()) for m in match_list) 
            if not val_span:
                result += (getter.default,)
                continue
            values, span = zip(*val_span)
            values = getter.convert(values, keyword, raise_error=raise_error)
            if pos:
                values = (tuple(zip(v,repeat(p))) for v,p in zip(values, span))
            result += (flatten(values),)
        if len(result) == 1:
            return result[0]
        return result 

    # #--------------------------------------------------------------------------------
    # def section(self, name, raise_error=True):
    # #--------------------------------------------------------------------------------
    #     print('section',name, )
    #     name = name.upper()
    #     i = self.section_names.index(name)
    #     # if binary:
    #     #     data = self.binarydata()
    #     # else:
    #     data = self.data()
    #     ab = [(a,b) for name,a,b in split_by_words(data, self.section_names[i:i+2])]
    #     if not ab:
    #         if raise_error:
    #             raise SystemError(f'ERROR Section {name} not found in {self}')
    #         return None
    #     a, b = ab[0]
    #     section = namedtuple('section','name data pos')
    #     return section(name, data[a:b], (a,b))

    #--------------------------------------------------------------------------------
    def section(self, *sections, raise_error=True):
    #--------------------------------------------------------------------------------
        self._checked or self.check()
        ### Get section-names 
        #sections = tuple(set([self._getter[key.upper()].section for key in keys]))
        #print(sections)
        data = self.binarydata()
        #ind = sorted(self.section_names.index(section) for section in sections)
        #print(ind)
        #print([self.section_names[i:i+2] for i in ind])
        section_pos = {name.upper():(a,b) for name,a,b in split_by_words(data, self.section_names)}
        #ab = [(a,b) for i in ind for name,a,b in split_by_words(data, self.section_names[i:i+2])]
        #print(section_pos)
        pos = [section_pos[sec.encode()] for sec in sections]
        #print(pos)
        #print(sorted(pos))
        if not pos:
            if raise_error:
                raise SystemError(f'ERROR Section {list2text(sections)} not found in {self}')
            return None
        #a, b = ab[0]
        self._data = b''.join(data[a:b] for a,b in sorted(pos))
        return self
        #return b''.join(data[a:b] for a,b in sorted(pos))
        #return (data[a:b] for a,b in sorted(pos))
        #section = namedtuple('section','name data pos')
        #return section(name, data[a:b], (a,b))

    # #--------------------------------------------------------------------------------
    # def with_includes(self, section=None):             # Input_file
    # #--------------------------------------------------------------------------------
    #     print('with_includes', section)
    #     if section is None:
    #         return self
    #     self._checked or self.check()
    #     if not 'INCLUDE' in self.data():
    #         return self
    #     ### Create dict of section names and positions
    #     # sections = {name.upper():(a,b) for name, a, b in split_by_words(self._data, self.section_names)}
    #     #head = tail = ''
    #     #if isinstance(section,str):
    #     # section = section.upper()
    #     # if section not in sections.keys():
    #     #     if raise_error:
    #     #         raise SystemError(f'ERROR Section {section} not found in {self}')
    #     #     return self 
    #     # a, b = sections[section]
    #     sect = self.section(section)
    #     if not 'INCLUDE' in sect.data:
    #         return self
    #     a, b = sect.pos
    #     head = self._data[:a]
    #     tail = self._data[b:]
    #     self._data = self._data[a:b]
    #     while 'INCLUDE' in self._data:
    #         self._data = self._append_include_files()
    #     self._data = head + self._data + tail
    #     return self


    # #--------------------------------------------------------------------------------
    # def _append_include_files(self):                                     # Input_file
    # #--------------------------------------------------------------------------------
    #     print('_append')
    #     matches = self.get('INCLUDE', pos=True)
    #     out = []
    #     n = 0
    #     for file,(a,b) in matches:
    #         out.append(self._data[n:a])
    #         inc_file = self.file.parent/file
    #         out.append(remove_comments(inc_file, comment='--', end='END'))
    #         n = b
    #     out.append(self._data[n:])
    #     return ''.join(out)

    #--------------------------------------------------------------------------------
    def replace_keyword(self, keyword, new_string):                      # Input_file
    #--------------------------------------------------------------------------------
        ### Get keyword value and position in file
        match = self.get(keyword, pos=True) 
        if match:
            _, pos = match[0] # Get first match
        else:
            raise SystemError(f'ERROR Missing {keyword} in {self}')
        out = self._data[:pos[0]] + new_string + self._data[pos[1]:]
        with open(self.file, 'w') as f:
            f.write(out)


#====================================================================================
class UNRST_file(unfmt_file):
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, file, wait_func=None, end='ENDSOL', **kwargs):    # UNRST_file
    #--------------------------------------------------------------------------------
        super().__init__(file, '.UNRST')
        self.varmap = {'step'  : keypos(key='SEQNUM'),
                       'nwell' : keypos('INTEHEAD', [16] , 'NWELLS'), 
                       'day'   : keypos('INTEHEAD', [64] , 'IDAY'),
                       'month' : keypos('INTEHEAD', [65] , 'IMON'),
                       'year'  : keypos('INTEHEAD', [66] , 'IYEAR'),
                       'hour'  : keypos('INTEHEAD', [206], 'IHOURZ'),
                       'min'   : keypos('INTEHEAD', [207], 'IMINTS'),
                       'sec'   : keypos('INTEHEAD', [410], 'ISECND'),
                       'time'  : keypos(key='DOUBHEAD')}
        self.check = check_blocks(self, start='SEQNUM', end=end, wait_func=wait_func, **kwargs)


    #--------------------------------------------------------------------------------
    def __repr__(self):                                                  # UNRST_file
    #--------------------------------------------------------------------------------
        return f'<ECL.UNRST_file, {self}>'

    #--------------------------------------------------------------------------------
    def last_day(self):                                                # UNRST_file
    #--------------------------------------------------------------------------------
        return (t := self.get('time', N=-1, raise_error=False)) and t[0][0] or 0

    #--------------------------------------------------------------------------------
    def dates(self, N=0):                                                # UNRST_file
    #--------------------------------------------------------------------------------
        year, month, day = self.get('year','month','day', N=N)
        # dates = (datetime.strptime(f'{d} {m} {y}', '%d %m %Y').date() for d,m,y in zip(day, month, year))
        dates = (datetime.strptime(f'{d} {m} {y}', '%d %m %Y') for d,m,y in zip(day, month, year))
        if abs(N) == 1:
            return list(dates)[-1]
        else:
            return dates

    #--------------------------------------------------------------------------------
    def date(self, block, step):                                         # UNRST_file
    #--------------------------------------------------------------------------------
        if block.key() == 'INTEHEAD':
            d, m, y = block.data((64,67)) #[64:66]
            # return datetime.strptime(f'{d} {m} {y}', '%d %m %Y').date()
            return datetime.strptime(f'{d} {m} {y}', '%d %m %Y')
        return step

    #--------------------------------------------------------------------------------
    def step(self, block, step):                                         # UNRST_file
    #--------------------------------------------------------------------------------
        '''
        Used in sections to get step of current section
        '''
        
        if block.key() == 'SEQNUM':
            return block.data(0) #[0]
        return step

    #--------------------------------------------------------------------------------
    def sections(self, **kwargs):                                       # UNRST_file
    #--------------------------------------------------------------------------------
        return super().sections(init_key='SEQNUM', check_sync=self.step, **kwargs)

    #--------------------------------------------------------------------------------
    def data(self, *keys):                                               # UNRST_file
    #--------------------------------------------------------------------------------
        data = {}
        for block in self.blocks():
            if block.key() == 'SEQNUM':
                if data:
                    yield data
                data = {}
                data['SEQNUM'] = block.data(0) #[0]
            if block.key() == 'INTEHEAD':
                data['DATE'] = block.data((64,67)) #[64:67] #data[206:208], data[410] 
            for key in keys:
                if block.key() == key:
                    D = block.data()
                    data[key] = (min(D), max(D))



#====================================================================================
class RFT_file(unfmt_file):                                                # RFT_file
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file, wait_func=None, **kwargs):                    # RFT_file
    #--------------------------------------------------------------------------------
        super().__init__(file, '.RFT')
        self.check = check_blocks(self, start='TIME', end='CONNXT', wait_func=wait_func, **kwargs)

    #--------------------------------------------------------------------------------
    def not_in_sync(self, time, prec=0.1):                                 # RFT_file
    #--------------------------------------------------------------------------------
        data = self.check.data()
        if data and any(abs(nparray(data)-time) > prec):
            return True
        return False
        
    #--------------------------------------------------------------------------------
    def last_day(self):                                                   # RFT_file
    #--------------------------------------------------------------------------------
        return (data := self.check.data()) and data[-1] or 0


#====================================================================================
class UNSMRY_file(unfmt_file):
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file):
    #--------------------------------------------------------------------------------
        super().__init__(file, '.UNSMRY')
        self.spec = SMSPEC_file(file)
        self.key_names = ()
        self.well_names = ()
        self.varmap = {'days'  : keypos('PARAMS'  , [0], 'TIME'), 
                       'years' : keypos('PARAMS'  , [1], ''),
                       'step'  : keypos('MINISTEP', [0], '')}

    #--------------------------------------------------------------------------------
    def init_welldata(self, keys=()):
    #--------------------------------------------------------------------------------
        if self.is_file() and self.spec.read(keys=keys):
            self.varmap['welldata'] = keypos('PARAMS', self.spec.well_pos(), '')
            self.key_names = set(self.keys)
            self.well_names = set(self.wells)
            return True
        return False

    #--------------------------------------------------------------------------------
    def __getattr__(self, item):
    #--------------------------------------------------------------------------------
        return self.spec and getattr(self.spec, item) 


#====================================================================================
class SMSPEC_file(unfmt_file):
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file):
    #--------------------------------------------------------------------------------
        super().__init__(file, '.SMSPEC')
        self._inkeys = ()
        self._index = {}
        self._attr = {}

    #--------------------------------------------------------------------------------
    def read(self, keys=()):
    #--------------------------------------------------------------------------------
        self._inkeys = keys
        if not self.is_file():
            return False
        K, W, M, U = 0, 1, 2, 3   
        data = [b.data(strip=True) for b in self.blocks() if b.key() in ('KEYWORDS', 'WGNAMES', 'MEASRMNT', 'UNITS')]
        ### Fix MEASRMNT by joining substrings (MEASRMNT strings are multiples of 8 chars)
        if all(d for d in data):
            width = len(data[M])//max(len(data[K]), 1)
            data[M] = tuple(''.join(v).lower() for v in grouper(data[M], width))
            keys = keys or data[K]
            ### Dictionary with array index as key and value-tuple (well, varname, fluid type, data type)
            self._index = {i:(w,k,m,u) for i,(w,k,m,u) in enumerate(zip(data[W], data[K], data[M], data[U])) if k in keys and w and not '+' in w}
            self._attr = {a:[v[i] for v in self._index.values()] for i,a in enumerate(('wells', 'keys', 'measures', 'units'))}
            return True
        return False

    #--------------------------------------------------------------------------------
    def __getattr__(self, item):
    #--------------------------------------------------------------------------------
        if (val := self._attr.get(item)) is not None:
            return val
        raise AttributeError(f'{self.__class__.__name__} object has no attribute {item}')

    #--------------------------------------------------------------------------------
    def missing_keys(self):
    #--------------------------------------------------------------------------------
        return [a for a in self._inkeys if not a in self.keys]

    #--------------------------------------------------------------------------------
    def pos(self, keyword:int):
    #--------------------------------------------------------------------------------
        return keyword in self.data[0] and [self.data[0].index(keyword)] or []

    #--------------------------------------------------------------------------------
    def well_pos(self):
    #--------------------------------------------------------------------------------
        pos = tuple(self._index.keys())
        ### Group consecutive indexes into (first, last) limits
        limits = grouper([pos[0]] + flatten((a,b) for a,b in pairwise(pos) if b-a>1) + [pos[-1]], 2)
        limits = [(a,b+1)for a,b in limits]
        return limits



#====================================================================================
class text_file(File):
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file, suffix, **kwargs):
    #--------------------------------------------------------------------------------
        #self.file = Path(file).with_suffix(suffix)
        super().__init__(file, suffix, **kwargs)
        self._pattern = {}
        self._convert = {}

    # #-----------------------------------------------------------------------
    # def size(self):
    # #-----------------------------------------------------------------------
    #     return self.file.stat().st_size

    #-----------------------------------------------------------------------
    def get(self, *var_list, N=0, raise_error=True):
    #-----------------------------------------------------------------------
        values = {}
        for var in var_list:
            match = matches(file=self.file, pattern=self._pattern[var])
            values[var] = [self._convert[var](m.group(1)) for m in match]
        if raise_error and not all(values.values()):
            raise SystemError(f'ERROR Unable to read {var_list} from {self.file.name}')
        if N == 0:
            return list(values.values())
        else:
            if N > 0:
                N -= 1
            return [[v[N]] if v else [] for v in values.values()]



#====================================================================================
class MSG_file(text_file):
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file):
    #--------------------------------------------------------------------------------
        super().__init__(file, '.MSG')
        self._pattern = {'time' : r'<\s*\bmessage\b\s+\bdate\b="[0-9/]+"\s+time="([0-9.]+)"\s*>',
                        'step' : r'\bRESTART\b\s+\bFILE\b\s+\bWRITTEN\b\s+\bREPORT\b\s+([0-9]+)'}
        self._convert = {'time' : float,
                         'step' : int}



#====================================================================================
class PRT_file(text_file):
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file):
    #--------------------------------------------------------------------------------
        #self.file = Path(file).with_suffix('.PRT')
        super().__init__(file, '.PRT')
        self._pattern = {'time' : r'TIME=?\s+([0-9.]+)\s+DAYS',
                        'step' : r'\bSTEP\b\s+([0-9]+)'}
        self._convert = {'time' : float,
                         'step' : int}



#====================================================================================
class check_blocks:                                                    # check_blocks
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, file, start=None, end=None, wait_func=None, warn_offset=True, timer=False):   # check_blocks
    #--------------------------------------------------------------------------------
        if isinstance(file, unfmt_file):
            self._unfmt = file
        else:
            self._unfmt = unfmt_file(file)
        self._keys = [start.ljust(8).encode(), [], end.ljust(8).encode(),  0]
        self._data = None
        self._wait_func = wait_func
        self._warn_offset = warn_offset
        self._timer = timer
        DEBUG and print(f'Creating {self}')

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                 # check_blocks
    #--------------------------------------------------------------------------------
        return f'<ECL.check_blocks(file={self._unfmt}>'

    #--------------------------------------------------------------------------------
    def __del__(self):                                                 # check_blocks
    #--------------------------------------------------------------------------------
        DEBUG and print(f'Deleting {self}')

    #--------------------------------------------------------------------------------
    def data(self):                                                    # check_blocks
    #--------------------------------------------------------------------------------
        return self._data and self._data[1]

    #--------------------------------------------------------------------------------
    def info(self, data=None, count=False):                            # check_blocks
    #--------------------------------------------------------------------------------
        return f"  {self._data[0].decode()} : {list2str(data and data or self._data[1], count=count)}"
        
    #--------------------------------------------------------------------------------
    def _blocks_complete(self, nblocks=1):                             # check_blocks
    #--------------------------------------------------------------------------------
        block = None
        start, start_val, end, end_val = 0, 1, 2, 3
        for block in self._unfmt.blocks(only_new=True):
            if block._key == self._keys[start]:
                if (data := block.data()):
                    self._keys[start_val].append(data[0])
                else:
                    return False    
            if block._key == self._keys[end]:
                self._keys[end_val] += 1 
                if self.steps_complete() and self._keys[end_val] == nblocks:
                    ### nblocks complete blocks read, reset counters and return True
                    self._data = self._keys[:start_val+1]
                    self._keys[start_val], self._keys[end_val] = [], 0
                    return True
        return False

    #--------------------------------------------------------------------------------
    def steps_complete(self):
    #--------------------------------------------------------------------------------
        ### 1: start_list, 3: end_count
        return len(self._keys[1]) == self._keys[3]


    #--------------------------------------------------------------------------------
    def warn_if_offset(self):
    #--------------------------------------------------------------------------------
        msg = ''
        if (offset := self._unfmt.offset()):
            msg = f'WARNING {self._unfmt} not at end after check, offset is {offset}'
        return msg


    #--------------------------------------------------------------------------------
    def data_saved_maxmin(self, nblocks=1, iter=100, **kwargs):      # check_blocks
    #--------------------------------------------------------------------------------
        f'''
            Loop for {iter} iterations until {nblocks} start/end-blocks are found or end-of-file reached.
        '''
        if nblocks == 0:
            return []
        msg = []
        data = []
        n = nblocks
        v = 2
        while n > 0:
            passed = self._wait_func( self._blocks_complete, nblocks=n, limit=iter, timer=self._timer, v=v, **kwargs )
            #msg.append(f'start, end: {self._start, self._end}, at_end: {self.at_end()}, passed: {passed}')
            if self._unfmt.at_end() and self.steps_complete():
                ### blocks <= max_blocks
                break
            elif passed:
                ### Not at end, but check passed: Read one more block!
                n = 1
                data.extend(self.data())
                v = 4
            else:
                ### Not at end, not passed
                n -= 1
                msg.append(f'WARNING Trying to read n - 1 = {n} blocks')
        data.extend(self.data())
        msg.append(self.info(data=data, count=True))
        if not data:
            msg.append(f'WARNING No blocks read in {self._unfmt.file.name}')
        return msg


    #--------------------------------------------------------------------------------
    def data_saved(self, nblocks=1, wait_func=None, **kwargs):               # check_blocks
    #--------------------------------------------------------------------------------        
        msg = ''
        wait_func = self._wait_func or wait_func
        OK = wait_func( self._blocks_complete, nblocks=nblocks, log=self.info, timer=self._timer, **kwargs)
        msg += not OK and f'WARNING Check of {self._unfmt.file.name} failed!' or ''
        msg += self._warn_offset and self.warn_if_offset() or ''
        return msg



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
    def __repr__(self):                                                   # fmt_block                                                           
    #--------------------------------------------------------------------------------                                                            
        return f'<ECL.fmt_block(key={self.keyword:8s}, type={self._dtype.name}, length={self.length:8d}>'

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
        # Consider bytes_.append() here?
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
class fmt_file(File):                                                      # fmt_file
    #
    # Class to handle formatted Eclipse files.
    #
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, filename, suffix, **kwargs):                                  # fmt_file
    #--------------------------------------------------------------------------------
        #self.file = Path(filename)
        super().__init__(filename, suffix, **kwargs)
        self.fh = None


    #--------------------------------------------------------------------------------
    def blocks(self, warn_missing=False):                                  # fmt_file
    #--------------------------------------------------------------------------------
        if not self.is_file():
             return
        with open(self.file) as self.fh:
            for line in self.fh:
                try:
                    keyword, length, dtype = self.read_header(line)
                    data = self.read_data(length, dtype)
                except StopIteration:
                    if warn_missing:
                        print(f"\n  WARNING: Missing data in '{keyword}' block, file {self.file.name} not complete!")
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
            

#====================================================================================
class FUNRST_file(fmt_file):
#====================================================================================
    #----------------------------------------------------------------------------
    def __init__(self, filename):                           # FUNRST_file
    #----------------------------------------------------------------------------
        super().__init__(filename, '.FUNRST')


    #--------------------------------------------------------------------------------
    def data(self, *keys):                                       # FUNRST_file
    #--------------------------------------------------------------------------------
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
    def get_blocks(self, filemap, init_key, rename_duplicate, rename_key): # FUNRST_file
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
    def get_data_pos(self, filemap, size):                             # FUNRST_file
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

    # #----------------------------------------------------------------------------
    # def get_data_pos_v2(self, filemap, size):                             # FUNRST_file
    # #----------------------------------------------------------------------------
    #     dtypes = [("'"+k+"'").encode() for k in DTYPE_LIST]
    #     data_pos = {k:[] for k in DTYPE.keys()}
    #     data_match = (m for m in finditer(b'\S+', filemap[:size]) if m.group(0) in dtypes)
    #     for m in data_match:
    #         print(m)
            
    #         data_pos[m.group(0)[1:-1]].append(m.span()[1]+1)
    #     return data_pos

    #----------------------------------------------------------------------------
    def prepare_helper_arrays(self, blocks, nblocks):                  # FUNRST_file
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
                rename_key=None, echo=False, progress=lambda x:None, cancel=lambda:None):  # FUNRST_file 
    #--------------------------------------------------------------------------------
        outfile = self.file.with_suffix(ext)
        # if self.size() < 1:
        #     return None
        with open(self.file) as f:
            with mmap(f.fileno(), length=0, offset=0, access=ACCESS_READ) as filemap:
                # prepare 
                blocks = self.get_blocks(filemap, init_key, rename_duplicate, rename_key)
                unit_format = ''.join(blocks.format) 
                data_pos, pos_stride = self.get_data_pos(filemap, blocks.size['blocks'])
                #self.get_data_pos_v2(filemap, blocks.size['blocks'])                
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
                        #data = (m.group(0) for m in finditer(b'\S+', filemap[a:b])) # generator version of split
                        a = b
                        buffer = self.string_to_num(nblocks, blocks, data_pos, data, pos_stride)
                        data_chunks = ((*heads[i], *buffer[types[i]][slices[i][0]:slices[i][1]], tails[i]) for i in range(nblocks*blocks.num['chunks']))
                        #out.write(pack(ENDIAN+nblocks*unit_format, *[x for y in data_chunks for x in y]))
                        out.write(pack(ENDIAN+nblocks*unit_format, *(x for y in data_chunks for x in y)))
                        n += nblocks
                        progress(n)
                        cancel()
        return outfile

    #----------------------------------------------------------------------------
    def string_to_num(self, nblocks, blocks, data_pos, data, pos_stride): # FUNRST_file
    #----------------------------------------------------------------------------
        buffer = {}
        # Loop over all datatypes (INTE, REAL, DOUB, etc.)
        for dtyp in blocks.stride.keys():
            # dtype=datatype[dtyp.decode()]
            dtype = DTYPE[dtyp].nptype
            #buf = [data[i+nb*pos_stride:j+nb*pos_stride] for i,j in data_pos[dtyp] for nb in range(nblocks)]
            buf = []
            for i,j in data_pos[dtyp]:
                for nb in range(nblocks):
                    buf.append(data[i+nb*pos_stride:j+nb*pos_stride])
                    #print(f'{dtyp}: data[{i}]={data[i]}, data[{j}]={data[j-1]}')
            try: 
                buffer[dtyp] = nparray([x for y in buf for x in y], dtype=dtype)
            except ValueError:
                #buffer[dtyp] = nparray([x[:15]+b'E'+x[17:] for y in buf for x in y], dtype=dtype)
                buffer[dtyp] = nparray([x.decode().replace('D','E') for y in buf for x in y], dtype=dtype)
        return buffer


    # #----------------------------------------------------------------------------
    # def convert(self, ext='UNRST', init_key='SEQNUM', rename_duplicate=True,
    #             rename_key=None, echo=False, progress=lambda x:None, cancel=lambda:None):  # FUNRST_file
    # #--------------------------------------------------------------------------------
    #     if rename_key and len(rename_key)<2:
    #         raise SystemError(f"ERROR in convert: Format of rename_keyword options is ('old name', 'new name'), but {rename_key} were given")
    #     stem = self.file.stem.upper()
    #     fname = str(self.file.parent/stem)+'.'+ext
    #     out_file = open(fname, 'wb')
    #     bytes_ = bytearray()
    #     n = 0
    #     count = {}
    #     for block in self.blocks():
    #         key = block.key()
    #         if key==init_key and len(bytes_)>0:
    #             # write previous block to file, and reset bytes_
    #             n += 1
    #             out_file.write(bytes_)
    #             # reset bytes for next section
    #             bytes_ = bytearray()
    #             progress(n)
    #             cancel()
    #             count = {}
    #         if rename_duplicate:
    #             if count.get(key):
    #                 # duplicate keyname, rename key
    #                 block.set_key(key[:-1]+str(count[key]+1))
    #             else:
    #                 # create new entry
    #                 count[key] = 0
    #             count[key] += 1
    #         if rename_key and key==rename_key[0]:
    #             block.set_key(rename_key[1])
    #         bytes_ += block.unformatted()
    #     out_file.close()
    #     if echo:
    #         print(f'{self.file.name} converted to {Path(fname)}')
    #     return Path(fname)
            
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
class RSM_file(File):                                                      # RSM_file
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, filename, suffix, **kwargs):
    #--------------------------------------------------------------------------------
        #self.file = Path(filename)
        super().__init__(filename, suffix, **kwargs)
        self.fh = None
        self.tag = '1'
        self.nrow = self.block_length()-10
        self.colpos = None
        
    #--------------------------------------------------------------------------------
    def get_data(self):                                                    # RSM_file
    #--------------------------------------------------------------------------------
        if not self.file.is_file():
            return ()
        with open(self.file) as self.fh:
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
        with open(self.file) as fh: 
            nb, n = 0, 0 
            for line in fh: 
                n += 1 
                if line[0]==self.tag: 
                    nb += 1 
                    if nb==2: 
                        return int(n) 


