
# -*- coding: utf-8 -*-

DEBUG = False
ENDIAN = '>'  # Big-endian

from dataclasses import dataclass
from itertools import chain, repeat, accumulate, groupby, zip_longest
from operator import attrgetter, itemgetter
from pathlib import Path
from platform import system
from numpy import zeros, int32, float32, float64, bool_ as np_bool, array as nparray, append as npappend 
from mmap import mmap, ACCESS_READ
from re import MULTILINE, finditer, compile as re_compile, search
from copy import deepcopy
from collections import namedtuple
from datetime import datetime, timedelta
from struct import unpack, pack, error as struct_error
from locale import getpreferredencoding
from matplotlib.pyplot import figure as pl_figure
#from numba import njit, jit
from .utils import decode, tail_file, index_limits, flatten, flatten_all, groupby_sorted, grouper, list2text, pairwise, remove_chars, safezip, list2str, float_or_str, matches, split_by_words, string_chunks

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
        return f'<{type(self)}, {self}>'

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
        if index == () or () in index:
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
        except struct_error as err:
            if raise_error:
                raise SystemError(f'ERROR Unable to read {self.key()} from {self._file.name}') from err
            return None
        ### Decode string data
        if self._type == b'CHAR':
            values = tuple(string_chunks(values[0].decode(), 8*nchar, strip=strip))
        ### Return value instead of single-value tuple
        if unwrap_tuple and len(values) == 1:
            return values[0]
        return values



#====================================================================================
class File:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, filename, suffix=None, role=None, ignore_suffix_case=False, exists=False):          # File
    #--------------------------------------------------------------------------------
        self.path = Path(filename).resolve() if filename else None
        if suffix:
            self.path = self.with_suffix(suffix, ignore_suffix_case, exists)
        self.role = role.strip()+' ' if role else ''
        self.debug = DEBUG and self.__class__.__name__ == File.__name__
        if self.debug:
            print(f'Creating {repr(self)}')

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                        # File
    #--------------------------------------------------------------------------------
        return f"<{self.__class__.__name__}, file={self.path}, role={self.role or None}>"

    #--------------------------------------------------------------------------------
    def __str__(self):                                                         # File
    #--------------------------------------------------------------------------------
        return f'{self.role}{self.name}'
        #return f'{self.path and self.path.name}'

    #--------------------------------------------------------------------------------
    def __del__(self):                                                         # File
    #--------------------------------------------------------------------------------
        if self.__class__.__name__ == File.__name__ and self.debug:
            print(f'Deleting {repr(self)}')

    #--------------------------------------------------------------------------------
    def __getattr__(self, item):                                               # File
    #--------------------------------------------------------------------------------
        #print('File',self.path, item)
        try:
            attr = getattr(self.path or Path(), item)
        except AttributeError as error:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'") from error
        if self.path:
            return attr
        # self.path is None, return None or None-function
        if callable(attr):
            return lambda: None
        return None

    #--------------------------------------------------------------------------------
    def binarydata(self, raise_error=False):                                    # File
    #--------------------------------------------------------------------------------
        ### Open as binary file to avoid encoding errors
        if self.is_file():
            with open(self.path, 'rb') as f:
                return f.read()
        if raise_error:
            raise SystemError(f'File {self} does not exist')
        return b''
 
    # #--------------------------------------------------------------------------------
    # def decode(self, data):
    # #--------------------------------------------------------------------------------        
    #     encoding = ('utf-8', 'latin1')
    #     for enc in encoding:
    #         try:
    #             return data.decode(encoding=enc)
    #         except UnicodeError:
    #             continue
    #     raise SystemError(f'ERROR decode with {encoding} encoding failed for {self}')

    #--------------------------------------------------------------------------------
    def delete(self, raise_error=False, echo=False):                           # File
    #--------------------------------------------------------------------------------
        if not self.path:
            return
        try:
            self.path.unlink(missing_ok=True)
        except (PermissionError, FileNotFoundError) as error:
            if raise_error:
                raise SystemError(f'Unable to delete {self}: {error}') from error
            if echo:
                print(f'Deleted {self}')

    #--------------------------------------------------------------------------------
    def is_file(self):                                                         # File
    #--------------------------------------------------------------------------------
        if not self.path:
            return False
        return self.path.is_file()

    #--------------------------------------------------------------------------------
    def with_name(self, file):                                                 # File
    #--------------------------------------------------------------------------------
        if not self.path:
            return
        return (self.path.parent/file).resolve()

    #--------------------------------------------------------------------------------
    def with_suffix(self, suffix, ignore_case=False, exists=False):        # File
    #--------------------------------------------------------------------------------
        """
            exists = True:  return first existing file with filename = self.stem + suffix
                   = False: return Path.with_suffix()
        """
        if not self.path:
            return None
        if ignore_case:
            exists = True
        if not exists:
            return self.path.with_suffix(suffix)
        # Require suffix starting with .
        if suffix[0] != '.':
            raise ValueError(f"Invalid suffix '{suffix}'")
        if ignore_case:
            # 'abc' -> '[aA][bB][cC]'
            suffix = '.[' + ']['.join(s+s.swapcase() for s in suffix[1:]) + ']'
        return next(self.glob(suffix), None)
        #return next(self.file.parent.glob(self.file.stem+suffix), None)
        
    #--------------------------------------------------------------------------------
    def glob(self, pattern):                                                   # File
    #--------------------------------------------------------------------------------
        if not self.path:
            return ()
        return self.path.parent.glob(self.path.stem + pattern)

    #--------------------------------------------------------------------------------
    def exists(self, raise_error=False):                                       # File
    #--------------------------------------------------------------------------------
        if self.is_file():
            return True
        if raise_error:
            raise SystemError(f'ERROR {self} is missing in folder {self.parent}')
            #raise SystemError(f'ERROR {" ".join((self.role, str(self.name))).lstrip()} is missing in folder {self.parent}')
        return False


    #--------------------------------------------------------------------------------
    def size(self):                                                            # File
    #--------------------------------------------------------------------------------
        size = -1
        if self.is_file():
            size = self.path.stat().st_size
        return size

    #--------------------------------------------------------------------------------
    def tail(self, **kwargs):
    #--------------------------------------------------------------------------------
        return tail_file(self.path, **kwargs)




#====================================================================================
class unfmt_file(File):
#====================================================================================
    start = None
    end = None
    var_pos = {}

    #--------------------------------------------------------------------------------
    def __init__(self, filename, **kwargs):                      # unfmt_file
    #--------------------------------------------------------------------------------
        super().__init__(filename, **kwargs)
        self.endpos = 0
        # self.varmap = {}
        if DEBUG:
            print(f'Creating {unfmt_file.__repr__(self)}')

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                  # unfmt_file
    #--------------------------------------------------------------------------------
        return f'<{super().__repr__()}, endpos={self.endpos}>'

    #--------------------------------------------------------------------------------
    def at_end(self):                                                    # unfmt_file
    #--------------------------------------------------------------------------------
        return self.endpos == self.size()

    #--------------------------------------------------------------------------------
    def offset(self):                                                    # unfmt_file
    #--------------------------------------------------------------------------------
        return self.size() - self.endpos

    #--------------------------------------------------------------------------------
    def blockdata(self, *keys, strip=True, **kwargs):                        # unfmt_file
    #--------------------------------------------------------------------------------
        return (bl.data(strip=strip) for bl in self.blocks(**kwargs) if bl.key() in keys)


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
        #print(f'{self}, start:{start}, only_new:{only_new}, {self.size()} - {startpos} = {self.size() - startpos}')
        if self.size() - startpos < 24: # Header is 24 bytes
            return False
        with open(self.path, mode='rb') as file:
            try:
                with mmap(file.fileno(), length=0, access=ACCESS_READ) as data:
                    size = data.size()
                    pos = startpos
                    while pos < size:
                        start = pos
                        ### Header
                        try:
                            ### Header is 24 bytes, we skip int of length 4 before and after
                            key, length, typ = unpack(ENDIAN+'8si4s', data[pos+4:pos+20])
                            ### Value array
                            nbytes = length*DTYPE[typ].size + 8 * -(-length//DTYPE[typ].max) # -(-a//b) is the ceil-function
                            pos += 24 + nbytes
                        except (ValueError, struct_error):
                            return False
                        self.endpos = pos
                        yield unfmt_block(key=key, length=length, type=typ, start=start, end=pos, 
                                        data=data, file=self.path)
            except ValueError: # as error: # Catch 'cannot mmap an empty file'
                #print(error)
                return False

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
    def tail_blocks(self, **kwargs):                                               # unfmt_file
    #--------------------------------------------------------------------------------
        if not self.is_file() or self.size() < 24: # Header is 24 bytes
            return ()
        with open(self.path, mode='rb') as file:
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
                            if size == 16 and data[pos+12:pos+16] in DTYPE:
                                start = data.tell()-4
                                key, length, typ = unpack(ENDIAN+'8si4s', data.read(16))
                                data.seek(4, 1)
                                break
                            data.seek(-4, 1)
                        except (ValueError, struct_error):
                            return ()
                    ### Value array
                    #data_start = data.tell()
                    data.seek(start, 0)
                    yield unfmt_block(key=key, length=length, type=typ, start=start, end=end,
                                    data=data, file=self.path)


    #--------------------------------------------------------------------------------
    def read(self, *varnames, tail=False, start=0, stop=None, step=None, drop=None, unpack_single=True, **kwargs):    # unfmt_file
    #--------------------------------------------------------------------------------
        if not self.is_file():
            return
        # Check for wrong value names
        if missing := [val for val in varnames if val not in self.var_pos]:
            err = f'ERROR! Invalid variable names in {self.__class__.__name__}.read() call: {missing}'
            raise SystemError(err)
        # Get order of keywords
        key_order = {v[0]:0 for k,v in self.var_pos.items() if k in varnames}.keys()
        # Make list of [key, pos, var]: ['INTEHEAD', 66, 'year']
        in_order = [self.var_pos[v]+(v,) for v in varnames]
        # Group positions and varnames: 'INTEHEAD': [[207, 'min'],[66, 'year']]
        key_pos_name = dict(groupby_sorted(in_order, key=itemgetter(0)))
        # Sort on positions for each keyword: 'INTEHEAD': [[66, 'year'],[207, 'min']]
        key_pos_name = [(v,sorted(key_pos_name[v], key=itemgetter(0))) for v in key_order]
        blocks = self.blocks
        end_key = self.end
        if tail:
            # Read file reversed from tail to top
            blocks = self.tail_blocks
            # Reverse keyword read-order
            key_pos_name = key_pos_name[::-1]
            # Reverse start/end keywords
            end_key = self.start
        # keyword : ([pos1], size1, name1, [pos2], size2, name2)
        tmp = {k:flatten((v[:-1], len(v[:-1]), v[-1]) for v in pn) for k,pn in key_pos_name}
        values = flatten(tmp.values())
        var_limits = list(pairwise(accumulate((0,)+values[1::3])))
        read_order = values[2::3]
        # Get positions for each keyword: INTEHEAD:[66, 207]
        keypos_to_read = {k:index_limits(flatten(v[0::3])) for k,v in tmp.items()}
        #print(keypos_to_read)
        # Map input to read-order limits: [(1,2), (0,1)] if ('min','year') is input
        out_limits = [var_limits[read_order.index(v)] for v in varnames]
        if unpack_single:
            # Only slice (return a list) for non-single variables
            out_slice = [slice(a,b) if b-a>1 else a for a,b in out_limits]
        else:
            # Always slice
            out_slice = [slice(*i) for i in out_limits]
        section = []
        num = 0
        for block in blocks(**kwargs):
            if end_key in block:
                num += 1
            if num < start:
                continue
            if step and (num-start)%step > 0:
                continue
            if block and (positions:=keypos_to_read.get(block.key())):
                # A pos < 0 in self.var_pos makes index_limits return () 
                # which cause the whole array to be read
                if () in positions:
                    out_slice = [slice(0,block.length())]
                section.extend(block.data(*positions, unwrap_tuple=False))
            if end_key in block and section: # and i > 0:
                data = [section[s] for s in out_slice]
                if not drop or not drop(data):
                    yield data
                section = []
            if stop and num >= stop:
                return


    # #--------------------------------------------------------------------------------
    # def get(self, *var_list, N=0, stop=(), raise_error=True, **kwargs):  # unfmt_file
    # #--------------------------------------------------------------------------------
    #     #print(var_list, N, kwargs)
    #     blocks = self.blocks
    #     if N < 0:
    #         # Read data from end of file
    #         blocks = self.tail_blocks
    #         N = -N
    #     varmap = {k:v for k,v in self.varmap.items() if k in var_list}
    #     # Create dict of keywords with varname and position:
    #     #  {'INTEHEAD':[('day',64), ('month',65), ('year',66)]}
    #     var_pos = {v.key:[] for v in varmap.values()}
    #     for k,v in varmap.items():
    #         var_pos[v.key].append( (k, v.pos) )
    #     #print(var_pos)
    #     values = {v:[] for v in var_list}
    #     def size():
    #         return (len(v) for v in values.values())
    #     for block in blocks(**kwargs):
    #         if block.key() in var_pos.keys():
    #             for var, pos in var_pos[block.key()]:
    #                 #values[var].append( b.data()[pos] )
    #                 values[var].append( block.data(*pos) )
    #         if N and set(size()) == set([N]):
    #             break
    #         if stop and stop[1] == values[stop[0]][-1] and len(set(size())) == 1:
    #             break
    #     if not all(values.values()):
    #         if raise_error:
    #             raise SystemError(
    #                 'ERROR Unable to read ' + list2str(var_pos.keys(), sep="'") + f' from {self}')
    #         return []
    #     return list(values.values())


    #--------------------------------------------------------------------------------
    def sections(self, begin=0, check_sync=lambda *x:0, init_key=None, start_before=None,
                 start_after=None, end_before=None, end_after=None):    # unfmt_file
    #--------------------------------------------------------------------------------
        if not self.exists():
            raise SystemError(f'ERROR File {self.path} not found') 
        inside = False
        step = None
        with open(self.path, 'rb') as file:
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
        with open(self.path, 'wb') as out_file, safezip(*sections) as zipper:
            ### Get data from the section generators
            for step_data in zipper:
                steps = []
                for step, data in step_data:
                    out_file.write(data)
                    steps.append(step)
                if len(set(steps)) > 1:
                    raise SystemError(
                        f'ERROR Sections are not synchronized in unfmt_file.create(): {steps}')
                progress(steps[0])
                cancel()
            return_value = self.path
        return return_value

    #--------------------------------------------------------------------------------
    def assert_no_duplicates(self, raise_error=True):                     # unfmt_file
    #--------------------------------------------------------------------------------
        allowed = (self.start, 'ZTRACER')
        seen = set()
        duplicate = (key for b in self.blocks() if (key:=b.key()) in seen or seen.add(key))
        if (dup:=next(duplicate)) not in allowed:
            msg = f'Duplicate keyword {dup} in {self}'
            if raise_error:
                raise SystemError('ERROR ' + msg)
            print('WARNING ' + msg)

#====================================================================================
class DATA_file(File):
#====================================================================================
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
                 'BOX','PERMX','PERMY','PERMZ','TOPS', 'INIT','RPTGRID','PVCDO','PVTW','SGOF','SWOF',
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
                 'MINPV','COPY','MULTIPLY')

    #--------------------------------------------------------------------------------
    def __init__(self, file, suffix=None, check=False, sections=True, **kwargs):      # DATA_file
    #--------------------------------------------------------------------------------
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
            'TSTEP'   : getter('SCHEDULE', (),      self._convert_float,  r'\bTSTEP\b\s+([0-9*.\s]+)/\s*'),
            'START'   : getter('RUNSPEC',  (0,),    self._convert_date,   r'\bSTART\b\s+(\d+\s+\'*\w+\'*\s+\d+)'),
            'DATES'   : getter('SCHEDULE', (),      self._convert_date,   r'\bDATES\b\s+((\d{1,2}\s+\'*\w{3}\'*\s+\d{4}\s*\s*/\s*)+)/\s*'), 
            'RESTART' : getter('SOLUTION', ('', 0), self._convert_file,   r"\bRESTART\b\s+('*[\w./\\-]+'*\s+[0-9]+)\s*/"),
            'WELSPECS': getter('SCHEDULE', (),      self._convert_string, r'\bWELSPECS\b((\s+\'*[\w/-]+?.*/\s*)+/)')}
        if check:
            self.check()
        # Extract whole record: r"^[ \t]*EQUALS(?:.|[\r\n])*?^[ \t]*/"
        # Remove comments and empty lines: r"^(?:(?!--).)*[\w/']+"
        # Alt. DATES: r'\bDATES\b\s+(\d+\s+\'*\w+\'*\s+\d+)\s*/\s*/\s*')
        # 'INCLUDE' : getter(None,       [''],    self._convert_file,   r"\bINCLUDE\b\s+'*([a-zA-Z0-9_./\\-]+)'*\s*/"), 
        # 'GDFILE'  : getter(None,       [''],    self._convert_file,   r"\bGDFILE\b\s+'*([a-zA-Z0-9_./\\-]+)'*\s*/"), 
        # 'SUMMARY' : getter('SUMMARY',  (),      self._convert_string, r'\bSUMMARY\b((\s*\w+\s*/*\s*)+)\bSCHEDULE\b'),
        # 'RESTART' : getter('SOLUTION', ('', 0), self._convert_file,   r"\bRESTART\b\s+('*[a-zA-Z0-9_./\\-]+'*\s+[0-9]+)\s*/"),

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                   # DATA_file
    #--------------------------------------------------------------------------------
        return f'<{type(self)}, {self.path}>'

    #--------------------------------------------------------------------------------
    def __call__(self):                                                  # DATA_file
    #--------------------------------------------------------------------------------
        self.data = self.binarydata()
        return self

    #--------------------------------------------------------------------------------
    def __contains__(self, key):                                          # DATA_file
    #--------------------------------------------------------------------------------
        self.data = None
        return bool(self.search(key, regex=rf'^[ \t]*{key}', comments=True))
        
    #--------------------------------------------------------------------------------
    def binarydata(self, raise_error=False):                              # DATA_file
    #--------------------------------------------------------------------------------
        self.data = super().binarydata(raise_error)
        end = b'END' in self.data and search(rb'^[ \t]*\bEND\b', self.data, flags=MULTILINE)
        return end and self.data[:end.end()] or self.data

    #--------------------------------------------------------------------------------
    def check(self, include=True):                                        # DATA_file
    #--------------------------------------------------------------------------------
        self._checked = True
        # Check if file exists
        self.exists(raise_error=True)
        # If Linux, check that file name is all capital letters to avoid I/O error in Eclipse
        if self.suffix == '.DATA' and system() == 'Linux' and not self.name.isupper():
            raise SystemError(f"ERROR Invalid filename format' {self.name}'. Under Linux, Eclipse only accepts uppercase letters")
        # Check if included files exists
        if include and (missing := [f for f in self.include_files() if not f.is_file()]):
            raise SystemError(f'ERROR {list2text([f.name for f in missing])} included from {self} is missing in folder {missing[0].parent}')
        return True

    #--------------------------------------------------------------------------------
    def search(self, key, regex, comments=False):                         # DATA_file
    #--------------------------------------------------------------------------------
        data = self._matching(key)
        if not comments:
            self.data = self._remove_comments(data)
        else:
            self.data = decode(b''.join(data))
        return search(regex, self.data, flags=MULTILINE)

    #--------------------------------------------------------------------------------
    def is_empty(self):                                                  # DATA_file
    #--------------------------------------------------------------------------------
        ''' Check if file is empty '''
        return self._remove_comments() == ''

    #--------------------------------------------------------------------------------
    def include_files(self, data:bytes=None):                           # DATA_file
    #--------------------------------------------------------------------------------
        ''' Return full path of INCLUDE files as a generator '''
        return (f[0] for f in self._included_file_data(data))

    #--------------------------------------------------------------------------------
    def including(self, *files):                                          # DATA_file
    #--------------------------------------------------------------------------------
        ''' Add the given files and return self '''
        # Added files must be an iterator to avoid an infinite recursive
        # loop when self._added_files is called in _included_file_data
        self._added_files = iter(files)
        # Disable check to avoid check to consume the above iterator
        self._checked = True
        return self

    #--------------------------------------------------------------------------------
    def tsteps(self, start=None, negative_ok=False, missing_ok=False, pos=False, skiprest=False):     # DATA_file
    #--------------------------------------------------------------------------------
        ''' Return timesteps, if DATES are present they are converted to timesteps '''
        _start, tsteps, dates = self.get('START','TSTEP','DATES', pos=True)
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
            raise SystemError(f'ERROR Zero or negative timestep in {self} (check if TSTEP or RESTART oversteps a DATES keyword)')
        if not missing_ok and not tsteps:
            raise SystemError(f'ERROR No TSTEP or DATES in {self} (or the included files)')
        return tsteps if pos else tuple(next(zip(*tsteps)))
        #return pos and tsteps or tuple(next(zip(*tsteps))) # Do not return positions

    #--------------------------------------------------------------------------------
    def wellnames(self):                                                  # DATA_file
    #--------------------------------------------------------------------------------
        '''
        Return tuple of wellnames from WELSPECS and UNRST file for RESTART runs
        '''
        wells = self.welspecs()
        restart, step = self.get('RESTART')
        if restart and step:
            unrst = UNRST_file(restart, role='RESTART file')
            unrst.exists(raise_error=True)
            wells += unrst.wells(stop=step)
        return tuple(set(wells))

    #--------------------------------------------------------------------------------
    def welspecs(self):                                                  # DATA_file
    #--------------------------------------------------------------------------------
        '''
        Get wellnames from WELSPECS definitions in the DATA-file or in a
        schedule-file
        '''
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

    #--------------------------------------------------------------------------------
    def get(self, *keywords, raise_error=False, pos=False):                # DATA_file
    #--------------------------------------------------------------------------------
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
            match_list = re_compile(getter.pattern).finditer(self.data)
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

    #--------------------------------------------------------------------------------
    def lines(self):                                                       # DATA_file
    #--------------------------------------------------------------------------------
        return (line for line in self._remove_comments(self._matching()).split('\n') if line)

    #--------------------------------------------------------------------------------
    def text(self):                                                       # DATA_file
    #--------------------------------------------------------------------------------
        return self._remove_comments(self._matching())

    #--------------------------------------------------------------------------------
    def summary_keys(self, matching=()):                                # DATA_file
    #--------------------------------------------------------------------------------
        return [k for k in self.section('SUMMARY').text().split() if k in matching]

    #--------------------------------------------------------------------------------
    def section_positions(self, *sections):                               # DATA_file
    #--------------------------------------------------------------------------------
        data = self.data or self.binarydata()
        sec_pos = {sec.upper().decode():(a,b) for sec,a,b in split_by_words(data, self.section_names)}
        if not sections:
            return sec_pos
        return  {sec:pos for sec in sections if (pos := sec_pos.get(sec))}

    #--------------------------------------------------------------------------------
    def section(self, *sections, raise_error=True):                       # DATA_file
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def replace_keyword(self, keyword, new_string):                      # DATA_file
    #--------------------------------------------------------------------------------
        ### Get keyword value and position in file
        match = self.get(keyword, pos=True)
        if match:
            _, pos = match[0] # Get first match
        else:
            raise SystemError(f'ERROR Missing {keyword} in {self}')
        out = self.data[:pos[0]] + new_string + self.data[pos[1]:]
        with open(self.path, 'w') as f:
            f.write(out)

    #--------------------------------------------------------------------------------
    def _remove_comments(self, data=None):                   # DATA_file
    #--------------------------------------------------------------------------------
        data = data or (self.binarydata(),)
        lines = (l for d in data for l in d.split(b'\n'))
        text = (l.split(b'--')[0].strip() for l in lines)
        text = b'\n'.join(t for t in text if t)
        text = decode(text)
        return text+'\n' if text else ''

    #--------------------------------------------------------------------------------
    def _matching(self, *keys):                                           # DATA_file
    #--------------------------------------------------------------------------------
        #print('_matching', keys)
        self.data = self.data or self.binarydata()
        keys = [key.encode() for key in keys]
        if keys == [] or any(key in self.data for key in keys):
            yield self.data
        for file, data in self._included_file_data(self.data):
            if keys == [] or any(key in data for key in keys):
                yield data

    #--------------------------------------------------------------------------------
    def _included_file_data(self, data:bytes=None):                           # DATA_file
    #--------------------------------------------------------------------------------
        ''' Return tuple of filename and binary-data for each include file '''
        #print('_included_file_data', self._added_files)
        data = data or self.binarydata()
        #regex = rb"^[ \t]*(?:\bINCLUDE\b|\bGDFILE\b)(?:.*--.*\s*|\s*)*'*(.*?)['\s]*/\s*(?:--.*)*$"
        # Allow filenames without quotes ('"), i.e. !#%& and ASCII 28 - 126 (7e)
        # Details at: https://regex101.com/r/jTYq16/1
        regex = rb"^[ \t]*(?:\bINCLUDE\b|\bGDFILE\b)(?:\s*--.*\s*|\s*)*'*([!#%&\x28-\x7e]+)['\s]*/.*$"
        files = (m.group(1).decode() for m in re_compile(regex, flags=MULTILINE).finditer(data))
        #print('self.file',self.file)
        for file in chain(files, self._added_files):
            #print('file',file)
            new_filename = self.with_name(file)
            #print('new_file',new_filename)
            file_data = DATA_file(new_filename).binarydata()
            yield (new_filename, file_data)
            if b'INCLUDE' in file_data:
                for inc in self._included_file_data(file_data):
                    yield inc

    #--------------------------------------------------------------------------------
    def _days(self, time_pos, start=None):                           # DATA_file
    #--------------------------------------------------------------------------------
        ''' Return relative timestep in days given a timestep or a datetime '''
        last_date = start
        for t,p in time_pos:
            if isinstance(t, datetime):
                dt = t
            else:
                dt = last_date + timedelta(hours=t*24)
            yield (dt-last_date).total_seconds()/86400, p
            last_date = dt
            
    #--------------------------------------------------------------------------------
    def _convert_string(self, values, key, raise_error=False):             # DATA_file
    #--------------------------------------------------------------------------------
        ret = [v for val in values for v in val.split('\n') if v and v != '/']
        return (ret,)

    #--------------------------------------------------------------------------------
    def _convert_float(self, values, key, raise_error=False):            # DATA_file
    #--------------------------------------------------------------------------------
        #mult = lambda x, y : list(repeat(float(y),int(x))) # Process x*y statements
        def mult(x,y):
            # Process x*y statements
            return list(repeat(float(y),int(x)))
        values = ([mult(*n.split('*')) if '*' in n else [float(n)] for n in v.split()] for v in values)
        values = tuple(flatten(v) for v in values)
        return values or self._getter[key].default

    #--------------------------------------------------------------------------------
    def _convert_date(self, dates, key, raise_error=False):              # DATA_file
    #--------------------------------------------------------------------------------
        ### Remove possible quotes
        ### Extract groups of 3 from the dates strings 
        dates = (grouper(remove_chars("'/\n", v).split(), 3) for v in dates)
        dates = tuple([datetime.strptime(' '.join(d), '%d %b %Y') for d in date] for date in dates)
        return dates or self._getter[key].default

    #--------------------------------------------------------------------------------
    def _convert_file(self, values, key, raise_error=True):              # DATA_file
    #--------------------------------------------------------------------------------
        'Return full path of file'
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



#====================================================================================
class UNRST_file(unfmt_file):
#====================================================================================
    start = 'SEQNUM'
    end = 'ENDSOL'
    var_pos =  {'step'  : ('SEQNUM'  ,  0),
                'nwell' : ('INTEHEAD', 16),
                'day'   : ('INTEHEAD', 64),
                'month' : ('INTEHEAD', 65),
                'year'  : ('INTEHEAD', 66),
                'hour'  : ('INTEHEAD', 206),
                'min'   : ('INTEHEAD', 207),
                'sec'   : ('INTEHEAD', 410),
                'time'  : ('DOUBHEAD', 0),
                'wells' : ('ZWEL'    , -1)}  # -1 = whole array

    #--------------------------------------------------------------------------------
    def __init__(self, file, wait_func=None, end=None, role=None, **kwargs):    # UNRST_file
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.UNRST', role=role)
        self.end = end or self.end
        self.check = check_blocks(self, start=self.start, end=self.end, wait_func=wait_func, **kwargs)

    #--------------------------------------------------------------------------------
    def wells(self, **kwargs):                                           # UNRST_file
    #--------------------------------------------------------------------------------
        wells = flatten_all(self.read('wells', **kwargs))
        unique_wells = set(w for well in wells if (w:=well.strip()))
        return tuple(unique_wells)

    #--------------------------------------------------------------------------------
    def last_day(self):                                                  # UNRST_file
    #--------------------------------------------------------------------------------
        time = next(self.read('time', tail=True), None) or [0]
        return time[0]

    #--------------------------------------------------------------------------------
    def dates(self, **kwargs):                                           # UNRST_file
    #--------------------------------------------------------------------------------
        data = self.read('day','month','year', **kwargs)
        return (datetime.strptime(f'{d} {m} {y}', '%d %m %Y') for d,m,y in data)

    #--------------------------------------------------------------------------------
    def step(self, block, step):                                         # UNRST_file
    #--------------------------------------------------------------------------------
        '''
        Used in sections to get step of current section
        '''
        if block.key() == 'SEQNUM':
            return block.data(0)
        return step

    #--------------------------------------------------------------------------------
    def sections(self, **kwargs):                                        # UNRST_file
    #--------------------------------------------------------------------------------
        return super().sections(init_key=self.start, check_sync=self.step, **kwargs)

    #--------------------------------------------------------------------------------
    def end_key(self):                                                   # UNRST_file
    #--------------------------------------------------------------------------------
        block = next(self.tail_blocks(), None)
        if block:
            return block.key()


#====================================================================================
class RFT_file(unfmt_file):                                                # RFT_file
#====================================================================================
    start = 'TIME'
    end = 'CONNXT'
    var_pos =  {'time'     : ('TIME', 0),
                'wellname' : ('WELLETC', 1)}

    #--------------------------------------------------------------------------------
    def __init__(self, file, wait_func=None, **kwargs):                    # RFT_file
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.RFT')
        self.check = check_blocks(self, start=self.start, end=self.end, wait_func=wait_func, **kwargs)

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
        # Return data from last check if it exists
        if data := self.check.data():
            return data[-1]
        # Return time-value from tail of file
        time = next(self.read('time', tail=True), None) or [0]
        return time[0]
        #return (data := self.check.data()) and data[-1] or 0


#====================================================================================
class UNSMRY_file(unfmt_file):
#====================================================================================
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

    #--------------------------------------------------------------------------------
    def __init__(self, file):                                           # UNSMRY_file
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.UNSMRY')
        self.spec = SMSPEC_file(file)
        self.start = None
        self._plots = None

    #--------------------------------------------------------------------------------
    def welldata(self, keys=(), wells=(), only_new=False, as_array=False, named=False, **kwargs): # UNSMRY_file
    #--------------------------------------------------------------------------------
        """
        named = False   : Returns days, dates and a tuple of key, well, value for each key-well combination
        named = True    : Returns days, dates, and all keys as their names
        only_new = True : Returns only previously un-read data
        as_array = True : Converts key-well data into a numpy array
        """
        if self.is_file() and self.spec.welldata(keys=keys, wells=wells, named=named):
            self.var_pos['welldata'] = ('PARAMS', *self.spec.well_pos())
            try:
                days, data = zip(*self.read('days', 'welldata', only_new=only_new, **kwargs))
            except ValueError:
                days, data = (), ()
            if not data:
                return ()
            # Add dates
            self.start = self.start or self.spec.startdate()
            dates = (self.start + timedelta(days=day) for day in days)
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
            Values = namedtuple('Values','key well data')
            values = (Values(k, w, d) for k,w,d in kwd)
            return namedtuple('Welldata','days dates values')(days, tuple(dates), tuple(values))
        return ()

    #--------------------------------------------------------------------------------
    def plot(self, keys=(), wells=(), ncols=1, date=True, start=0, stop=None, step=None, args=None, **kwargs):                        # UNSMRY_file
    #--------------------------------------------------------------------------------
        if data := self.welldata(keys=keys, wells=wells, start=start, stop=stop, step=step, **kwargs):
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
                fig = pl_figure(1, clear=True, figsize=(8*ncols,4*nrows))
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
                ax.legend(loc='upper left', fontsize='smaller', ncols=-(-len(ax.lines)//7)) # max 7 labels each column
                ax.relim()
                ax.autoscale_view()
            fig.canvas.draw()

    #--------------------------------------------------------------------------------
    def key_units(self):                                                # UNSMRY_file
    #--------------------------------------------------------------------------------
        Var = namedtuple('Var','unit measure')
        kum = zip(*attrgetter('keys', 'units', 'measures')(self.spec))
        var = {k:Var(u, m.split(':')[-1].replace('_',' ')) for k,u,m in set(kum)}
        return namedtuple('Keys', self.keys)(**var)

    #--------------------------------------------------------------------------------
    def __getattr__(self, item):                                        # UNSMRY_file
    #--------------------------------------------------------------------------------
        try:
            # Look for attribute in File-class first
            return super().__getattr__(item)
        except AttributeError:
            return tuple(set(getattr(self.spec, item)))

    #--------------------------------------------------------------------------------
    def energy(self, *wells):                                             # UNSMRY_file
    #--------------------------------------------------------------------------------
        from scipy.integrate import cumtrapz
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


#====================================================================================
class SMSPEC_file(unfmt_file):                                          # SMSPEC_file
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file):                                           # SMSPEC_file
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.SMSPEC')
        self._inkeys = ()
        self._ind = ()
        self.measures = ()
        self.wells = ()
        self.data = ()

    #--------------------------------------------------------------------------------
    def welldata(self, keys=(), wells=(), named=False):                 # SMSPEC_file
    #--------------------------------------------------------------------------------
        self._inkeys = keys
        if not self.is_file():
            return False
        Data = namedtuple('Data','keys wells measures units', defaults=4*(None,))
        # blockdata = (b.data(strip=True) for b in self.blocks() if b.key() in ('KEYWORDS', 'WGNAMES', 'MEASRMNT', 'UNITS'))
        # self.data = Data(*blockdata)
        self.data = Data(*self.blockdata('KEYWORDS', 'WGNAMES', 'MEASRMNT', 'UNITS'))
        if all(self.data):
            width = len(self.data.measures)//max(len(self.data.keys), 1)
            keys = keys or set(self.data.keys)
            all_wells = set(w for w in self.data.wells if w and not '+' in w)
            patterns = (w.split('*')[0] for w in wells if '*' in w)
            matched = [w for p in patterns for w in all_wells if w.startswith(p)]
            wells = [w for w in wells if '*' not in w]
            wells = set(wells+matched) or all_wells
            ikw = enumerate(zip(self.data.keys, self.data.wells))
            # index into UNSMRY arrays
            self._ind = tuple(i for i,(k,w) in ikw if k in keys and w in wells)
            if self._ind:
                getter = itemgetter(*self._ind)
                measure_strings = map(''.join, grouper(self.data.measures, width))
                #self.measures = itemgetter(*self._ind)(tuple(measure_strings))
                self.measures = getter(tuple(measure_strings))
                if named:
                    #self.wells = itemgetter(*self._ind)(tuple(w.replace('-','_') for w in self.data.wells))
                    self.wells = getter(tuple(w.replace('-','_') for w in self.data.wells))
                else:
                    self.wells = getter(tuple(self.data.wells))
                return True
        return False

    #--------------------------------------------------------------------------------
    def startdate(self):                                                # SMSPEC_file
    #--------------------------------------------------------------------------------
        if start := next(self.blockdata('STARTDAT'), None):
            day, month, year, hour, minute, second = start
            return datetime(year, month, day, hour, minute, second)

    #--------------------------------------------------------------------------------
    def __getattr__(self, item):                                        # SMSPEC_file
    #--------------------------------------------------------------------------------
        """
        Read attributes from the named-tuple Data
        """
        if (val := getattr(self.data, item, None)) is not None:
            return itemgetter(*self._ind)(val) if self._ind else ()
        return super().__getattr__(item)

    #--------------------------------------------------------------------------------
    def missing_keys(self):                                             # SMSPEC_file
    #--------------------------------------------------------------------------------
        return [a for a in self._inkeys if not a in self.keys]

    #--------------------------------------------------------------------------------
    def well_pos(self):                                                 # SMSPEC_file
    #--------------------------------------------------------------------------------
        return self._ind


#====================================================================================
class text_file(File):
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                   # text_file
    #--------------------------------------------------------------------------------
        super().__init__(file, **kwargs)
        self._pattern = {}
        self._convert = {}

    #--------------------------------------------------------------------------------
    def __contains__(self, key):                                          # text_file
    #--------------------------------------------------------------------------------
        return key.encode() in self.binarydata()
        
    #--------------------------------------------------------------------------------
    def get(self, *var_list, N=0, raise_error=True):                      # text_file
    #--------------------------------------------------------------------------------
        values = {}
        for var in var_list:
            match = matches(file=self.path, pattern=self._pattern[var])
            values[var] = [self._convert[var](m.group(1)) for m in match]
        if raise_error and not all(values.values()):
            raise SystemError(f'ERROR Unable to read {var_list} from {self.path.name}')
        if N == 0:
            return list(values.values())
        if N > 0:
            N -= 1
        return [[v[N]] if v else [] for v in values.values()]



#====================================================================================
class MSG_file(text_file):
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file):
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.MSG')
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
        super().__init__(file, suffix='.PRT')
        self._pattern = {'time' : r'TIME=?\s+([0-9.]+)\s+DAYS',
                        'step' : r'\bSTEP\b\s+([0-9]+)'}
        self._convert = {'time' : float,
                         'step' : int}

    #--------------------------------------------------------------------------------
    def last_day(self):
    #--------------------------------------------------------------------------------
        text = (txt for txt in self.tail(size=10*1024) if 'TIME=' in txt)
        data = next(text, None)
        if data:
            days = list(m.group(1) for m in re_compile(self._pattern['time']).finditer(data))
            return float(days[-1]) if days else 0


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
        if DEBUG:
            print(f'Creating {self}')

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                 # check_blocks
    #--------------------------------------------------------------------------------
        return f'<{type(self)}, file={self._unfmt}>'

    #--------------------------------------------------------------------------------
    def __del__(self):                                                 # check_blocks
    #--------------------------------------------------------------------------------
        if DEBUG:
            print(f'Deleting {self}')

    #--------------------------------------------------------------------------------
    def data(self):                                                    # check_blocks
    #--------------------------------------------------------------------------------
        return self._data and self._data[1]

    #--------------------------------------------------------------------------------
    def info(self, data=None, count=False):                            # check_blocks
    #--------------------------------------------------------------------------------
        return f"  {self._data[0].decode()} : {list2str(data and data or self._data[1], count=count)}"
        
    #--------------------------------------------------------------------------------
    def blocks_complete(self, nblocks=1, only_new=True):                             # check_blocks
    #--------------------------------------------------------------------------------
        block = None
        start, start_val, end, end_val = 0, 1, 2, 3
        for block in self._unfmt.blocks(only_new=only_new):
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
        f"""
            Loop for {iter} iterations until {nblocks} start/end-blocks are found or end-of-file reached.
        """
        if nblocks == 0:
            return []
        msg = []
        data = []
        n = nblocks
        v = 2
        while n > 0:
            passed = self._wait_func( self.blocks_complete, nblocks=n, limit=iter, timer=self._timer, v=v, **kwargs )
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
            msg.append(f'WARNING No blocks read in {self._unfmt.path.name}')
        return msg


    #--------------------------------------------------------------------------------
    def data_saved(self, nblocks=1, wait_func=None, **kwargs):         # check_blocks
    #--------------------------------------------------------------------------------
        msg = ''
        wait_func = self._wait_func or wait_func
        OK = wait_func( self.blocks_complete, nblocks=nblocks, log=self.info, timer=self._timer, **kwargs)
        msg += not OK and f'WARNING Check of {self._unfmt.path.name} failed!' or ''
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
        return f'<{type(self)}, key={self.keyword:8s}, type={self._dtype.name}, length={self.length:8d}>'

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
    def __init__(self, filename, **kwargs):                                  # fmt_file
    #--------------------------------------------------------------------------------
        #self.file = Path(filename)
        super().__init__(filename, **kwargs)
        self.fh = None


    #--------------------------------------------------------------------------------
    def blocks(self, warn_missing=False):                                  # fmt_file
    #--------------------------------------------------------------------------------
        keyword = ''
        if not self.is_file():
             return
        with open(self.path, encoding=getpreferredencoding()) as self.fh:
            for line in self.fh:
                try:
                    keyword, length, dtype = self.read_header(line)
                    data = self.read_data(length, dtype)
                except StopIteration:
                    if warn_missing:
                        print(f"\n  WARNING: Missing data in '{keyword}' block, file {self.path.name} not complete!")
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
        super().__init__(filename, suffix='.FUNRST')


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
                if n > 1:
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
    #def fast_convert(self, nblocks=1, ext='.UNRST', init_key='SEQNUM', rename_duplicate=True,
    def as_UNRST(self, nblocks=1, ext='.UNRST', init_key='SEQNUM', rename_duplicate=True,
                rename_key=None, progress=lambda x:None, cancel=lambda:None):  # FUNRST_file 
    #--------------------------------------------------------------------------------
        outfile = self.path.with_suffix(ext)
        # if self.size() < 1:
        #     return None
        with open(self.path) as f:
            with mmap(f.fileno(), length=0, offset=0, access=ACCESS_READ) as filemap:
                # prepare
                blocks = self.get_blocks(filemap, init_key, rename_duplicate, rename_key)
                #print(blocks)
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
                        out.write(pack(ENDIAN+nblocks*unit_format, *(x for y in data_chunks for x in y)))
                        n += nblocks
                        progress(n)
                        cancel()
        return UNRST_file(outfile, end=UNRST_file(outfile).end_key())

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
    def __init__(self, filename, **kwargs):
    #--------------------------------------------------------------------------------
        #self.file = Path(filename)
        super().__init__(filename, **kwargs)
        self.fh = None
        self.tag = '1'
        self.nrow = self.block_length()-10
        self.colpos = None
        
    #--------------------------------------------------------------------------------
    def get_data(self):                                                    # RSM_file
    #--------------------------------------------------------------------------------
        if not self.path.is_file():
            return ()
        with open(self.path) as self.fh:
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
        with open(self.path) as fh: 
            nb, n = 0, 0 
            for line in fh: 
                n += 1 
                if line[0]==self.tag: 
                    nb += 1 
                    if nb==2: 
                        return int(n) 


