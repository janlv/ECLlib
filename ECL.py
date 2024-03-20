
# -*- coding: utf-8 -*-

from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import chain, repeat, accumulate, groupby, zip_longest, islice
from operator import attrgetter, itemgetter, sub as subtract
from pathlib import Path
from platform import system
from mmap import mmap, ACCESS_READ, ACCESS_WRITE
from re import MULTILINE, finditer, findall, compile as re_compile, search as re_search
from subprocess import Popen, STDOUT
from time import sleep
from collections import namedtuple
from datetime import datetime, timedelta
from struct import unpack, pack, error as struct_error
#from locale import getpreferredencoding
from shutil import copy
from numpy import int32, float32, float64, bool_ as np_bool, array as nparray
from matplotlib.pyplot import figure as pl_figure
from .utils import (batched, batched_when, cumtrapz, decode, flatten, group_indices, last_line, match_in_wildlist, pad, tail_file, head_file,
                    flat_list, flatten_all, grouper, list2text, pairwise, remove_chars,
                    list2str, float_or_str, matches, split_by_words, string_split, split_in_lines, take)
from .runner import Process


DEBUG = False
ENDIAN = '>'  # Big-endian
ECL2IX_LOG = 'ecl2ix.log'

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


#====================================================================================
@dataclass
class Dtyp:
#====================================================================================
    name     : str = ''     # ECL type name
    unpack   : str = ''     # Char used by struct.unpack/pack to read/write binary data
    size     : int = 0      # Bytesize 
    max      : int = 0      # Maximum number of data records in one block
    nptype   : type = None  # Type used in numpy arrays 
    max_bytes: int = field(init=False) # max * size

    #--------------------------------------------------------------------------------
    def __post_init__(self):
    #--------------------------------------------------------------------------------
        self.max_bytes = self.max * self.size


#                        name  unpack size max  nptype
DTYPE = {b'INTE' : Dtyp('INTE', 'i',   4, 1000, int32),
         b'REAL' : Dtyp('REAL', 'f',   4, 1000, float32),
         b'DOUB' : Dtyp('DOUB', 'd',   8, 1000, float64),
         b'LOGI' : Dtyp('LOGI', 'i',   4, 1000, np_bool),
         b'CHAR' : Dtyp('CHAR', 's',   8, 105 , str),
         b'C008' : Dtyp('C008', 's',   8, 105 , str),
         b'C009' : Dtyp('C009', 's',   9, 105 , str),
         b'MESS' : Dtyp('MESS', ' ',   1, 1   , str)}

DTYPE_LIST = [v.name for v in DTYPE.values()]
        

#====================================================================================
@dataclass
class Restart:
#====================================================================================
    days: float = 0
    step: int = 0
    file: str = ''
    run : bool = False


#====================================================================================
class File:                                                                    # File
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, filename, suffix=None, role=None, ignore_suffix_case=False, exists=False):          # File
    #--------------------------------------------------------------------------------
        #print('init',filename, suffix)
        self.path = Path(filename).resolve() if filename else None
        if suffix:
            self.path = self.with_suffix(suffix, ignore_suffix_case, exists)
        self.role = role.strip()+' ' if role else ''
        self.debug = DEBUG and self.__class__.__name__ == File.__name__
        if self.debug:
            print(f'Creating {repr(self)}')
        #print('path', self.path)

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

    @contextmanager
    #--------------------------------------------------------------------------------
    def mmap(self, write=False):                                               # File
    #--------------------------------------------------------------------------------
        filemap = None
        mode = 'rb'
        access = ACCESS_READ
        if write:
            mode += '+'
            access = ACCESS_WRITE
        try:
            with open(self.path, mode=mode) as file:
                filemap = mmap(file.fileno(), length=0, access=access)
                yield filemap
        finally:
            if filemap:
                filemap.close()

    #--------------------------------------------------------------------------------
    def resize(self, start=0, end=0):                                   # File
    #--------------------------------------------------------------------------------
        # NB Very slow for large files, use with caution! 
        if start >= end:
            raise SyntaxError("'start' must be less than 'end'")
        length = end - start
        size = self.size()
        newsize = size - length
        if newsize == 0:
            self.touch()
            return
        with self.mmap(write=True) as infile:
            # This is the slow operation
            infile.move(start, end, size-end)
            infile.flush()
            infile.resize(size - length)

    #--------------------------------------------------------------------------------
    def binarydata(self, raise_error=False):                                   # File
    #--------------------------------------------------------------------------------
        # Open as binary file to avoid encoding errors
        if self.is_file():
            with open(self.path, 'rb') as f:
                return f.read()
        if raise_error:
            raise SystemError(f'File {self} does not exist')
        return b''
 
    #--------------------------------------------------------------------------------
    def as_text(self, **kwargs):                                               # File
    #--------------------------------------------------------------------------------
        return self.binarydata(**kwargs).decode()

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
    def with_tag(self, head:str='', tail:str=''):                              # File
    #--------------------------------------------------------------------------------
        if not self.path:
            return
        return self.path.parent/(head + self.path.stem + tail + self.path.suffix)

    #--------------------------------------------------------------------------------
    def with_suffix(self, suffix, ignore_case=False, exists=False):            # File
    #--------------------------------------------------------------------------------
        """
            exists = True:  return first existing file with filename = self.stem + suffix or None
                   = False: return Path.with_suffix()
        """

        if not self.path:
            return None
        # Require suffix starting with .
        if suffix[0] != '.':
            raise ValueError(f"Invalid suffix '{suffix}'")
        ext = suffix
        if ignore_case:
            # 'abc' -> '[aA][bB][cC]'
            ext = '.[' + ']['.join(s+s.swapcase() for s in suffix[1:]) + ']'
        path = next(self.glob(ext), None)
        if not exists and path is None:
            path = self.path.with_suffix(suffix)
        return path

        
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
        return False

    #--------------------------------------------------------------------------------
    def __stat(self, attr):                                                    # File
    #--------------------------------------------------------------------------------
        if self.is_file():
            return getattr(self.path.stat(), attr)
        return -1

    #--------------------------------------------------------------------------------
    def size(self):                                                            # File
    #--------------------------------------------------------------------------------
        return self.__stat('st_size')

    #--------------------------------------------------------------------------------
    def creation_time(self):                                                   # File
    #--------------------------------------------------------------------------------
        return self.__stat('st_ctime')

    #--------------------------------------------------------------------------------
    def tail(self, **kwargs):                                                  # File
    #--------------------------------------------------------------------------------
        return next(tail_file(self.path, **kwargs), '')

    #--------------------------------------------------------------------------------
    def reversed(self, **kwargs):                                             # File
    #--------------------------------------------------------------------------------
        return tail_file(self.path, **kwargs)

    #--------------------------------------------------------------------------------
    def head(self, **kwargs):                                                  # File
    #--------------------------------------------------------------------------------
        return next(head_file(self.path, **kwargs), '')

    #--------------------------------------------------------------------------------
    def lines(self):                                                           # File
    #--------------------------------------------------------------------------------
        if self.is_file():
            with open(self.path, 'r', encoding='utf-8') as file:
                while line:=file.readline():
                    yield line
        return ()

    #--------------------------------------------------------------------------------
    def line_matching(self, word):                                             # File
    #--------------------------------------------------------------------------------
        return next((line for line in self.lines() if word in line), None)

    #--------------------------------------------------------------------------------
    def last_line(self):                                                       # File
    #--------------------------------------------------------------------------------
        return last_line(self.path)

    #--------------------------------------------------------------------------------
    def backup(self, tag, overwrite=False):                                    # File
    #--------------------------------------------------------------------------------
        backup_file = self.path.with_name(f'{self.stem}{tag}{self.suffix}')
        if overwrite or not backup_file.exists():
            copy(self.path, backup_file)
            return backup_file


    #--------------------------------------------------------------------------------
    def replace_text(self, text=(), pos=()):                                   # File
    #--------------------------------------------------------------------------------
        """
        Replace or append text. Replace text if pos is a (start, len+start) tuple, 
        append '\n'+text if pos is a (start, start) tuple
        
        text : tuple of strings
        pos  : tuple of tuples of length 2
        """
        data = self.binarydata().decode()
        size = len(data)
        # Sort on ascending position
        for txt, (a,b) in sorted(zip(text, pos), key=lambda x:x[1][0]):
            if b - a == 0:
                # Append text
                txt = '\n' + txt
            # Shift pos because pos is relative to the input text
            shift = len(data) - size
            new_data = data[:a+1+shift] + txt + data[b+shift:]
            data = new_data
        self.write_text(data)
        


#====================================================================================
class unfmt_header:                                                    # unfmt_header
#====================================================================================
    #         | h e a d e r  |     d a t a     |     d a t a     |    d a t a      |
    #         |4i|8s|4i|4s|4i|4i| 1000 data |4i|4i| 1000 data |4i|4i| 1000 data |4i| 
    #  bytes  |      24      |4 | 1000*size |  8  | 1000*size |  8  | 1000*size |4 |

    #--------------------------------------------------------------------------------
    def __init__(self, key:str=b'', length:int=0, type:str=b'', 
                 startpos:int=0, endpos:int=0):                        # unfmt_header
    #--------------------------------------------------------------------------------
        self.key = key
        self.length = length  # datalength
        self.type = type
        self.startpos = startpos
        self.endpos = endpos
        self.dtype = DTYPE[self.type]
        self.bytes = self.length*self.dtype.size # databytes
        if not endpos:
            if self.length:
                # Add the last payload int of 4 bytes to reach the end
                self.endpos = self._data_pos(self.length) + 4
            else:
                # No data, only header
                self.endpos = self.startpos + 24

    #--------------------------------------------------------------------------------
    def __str__(self):                                                 # unfmt_header
    #--------------------------------------------------------------------------------
        return (f'key={self.key.decode():8s}, type={self.type.decode():4s}, bytes={self.bytes:8d},' 
                f'length={self.length:8d}, start={self.startpos:8d}, end={self.endpos:8d}')

    #--------------------------------------------------------------------------------
    def _data_pos(self, pos):                                          # unfmt_header
    #--------------------------------------------------------------------------------
        """
        Return absolute file-position of given the relative index of the data-array 
        """
        #  | h e a d e r  |   d a t a     |   d a t a     |   d a t a     |
        #  |4i|8s|4i|4s|4i|4i|1000 data|4i|4i|1000 data|4i|4i|1000 data|4i| 
        #  |    24 bytes  |
        # Add 8 payload bytes (two ints) at the transition between data chunks
        return self.startpos + 24 + 4 + pos*self.dtype.size + 8*(pos//self.dtype.max)

    #--------------------------------------------------------------------------------
    def _data_slices(self, limits=((None,),)):                          # unfmt_header
    #--------------------------------------------------------------------------------
        dtype = self.dtype
        # Extend limit to whole range if None is given
        flat_lim = tuple(flatten(((0, self.length) if None in l else l for l in limits)))
        # Check for out-of-bounds limits
        if oob_err := [l for l in flat_lim if l<0 or l>self.length]:
            raise SyntaxWarning(
                f'{self.key.decode().strip()}: index {oob_err} is out of bounds {(0, self.length)}')
        # First and last file (byte) position of the slices 
        first_last = batched((self._data_pos(l) for l in flat_lim), 2)
        # The number of the first and last data chunk
        num = list(batched((l//dtype.max for l in flat_lim), 2))
        # The start position of the first payload  
        shift = (self._data_pos((n+1)*dtype.max)-8 for n,_ in num)
        # The distance in bytes between consecutive payloads 
        step = 8 + dtype.max_bytes
        # The list of payload start-positions between data start and stop
        pay_ran = (tuple(s+r*step for r in range(b-a)) for s,(a,b) in zip(shift, num))
        # Add first and last index to the ends of the payload ranges
        lims = ((f,*((r, r+8) for r in ran),l) for (f, l), ran in zip(first_last, pay_ran))
        # lims = list(lims)
        # print(lims)
        # Pull pairs of indices, and return slices 
        return (slice(*l) for l in batched(flatten_all(lims), 2))
        #return (slice(*l) for l in batched(flatten(lims), 2))
        #for (first, last), ran in zip(first_last, pay_ran):
        #    lims = chain([first], *((r, r+8) for r in ran), [last])
        #    yield (slice(*l) for l in batched(lims, 2))

    # #--------------------------------------------------------------------------------
    # def file_slices(self, limits=None):                                # unfmt_header
    # #--------------------------------------------------------------------------------
    #     dtype = self.dtype
    #     start = self.startpos + 24 + 4
    #     # A is the start positions of the data pieces
    #     dmax = dtype.max * dtype.size
    #     A = (pos+i*8 for i,pos in enumerate(range(start, start+self.bytes, dmax)))
    #     pos = ([a, a+dmax] for a in islice(A, self.bytes//dmax))
    #     if rest:=self.bytes%dmax:
    #         pos = chain(pos, ([a, a+rest] for a in A))
    #         #pos = pos + [[a, a+rest] for a in A]
    #     pos = (tuple(pos),)
    #     # Modify the positions if a limit is given
    #     if not any(l[0] is None for l in limits): 
    #         limits = list(flatten(limits))
    #         if any(l>self.length for l in limits):
    #             raise SyntaxWarning(
    #                 f'{self.key.decode().strip()} data index out of range ({self.length})')
    #         ind = (l//dtype.max for l in limits)
    #         # Use index to get the relevant slices-tuples
    #         pos = [[pos[0][i] for i in range(a,b+1)] for a,b in batched(ind,2)]
    #         #print('POS', pos)
    #         # Add lower and upper absolute limit to the (flattened) slice-tuples
    #         #abs_lim = [start + l*dtype.size for l in flatten(limits)]
    #         abs_lim = list( map(self.data_pos, limits) )
    #         #print('LIM', limits)
    #         #print('ABS', abs_lim)
    #         pos = list([a]+list(flatten(p))[1:-1]+[b] for (a,b),p in zip(batched(abs_lim,2), pos))
    #         #print('POS', pos)
    #         # Batch flat pos-list into slice-tuples
    #         pos = ((batched(p, 2)) for p in pos)
    #     return [[slice(*i) for i in p] for p in pos]

    # #--------------------------------------------------------------------------------
    # def unpack_format(self, limit=None):                               # unfmt_header
    # #--------------------------------------------------------------------------------
    #     #if any(l[0] is None for l in limit):
    #     if any(None in l for l in limit):
    #         # No limit, extract all
    #         length = self.bytes if self.is_char() else self.length
    #     else:
    #         #print(limit)
    #         length = sum(-subtract(*l) for l in limit) * (self.dtype.size if self.is_char() else 1)
    #     return ENDIAN+f'{length}{self.dtype.unpack}'

    # #--------------------------------------------------------------------------------
    # def unpack_format2(self, slices):                               # unfmt_header
    # #--------------------------------------------------------------------------------
    #     length = sum(sl.stop-sl.start for sl in flatten(slices))//(1 if self.is_char() else self.dtype.size)
    #     return ENDIAN+f'{length}{self.dtype.unpack}'

    # #--------------------------------------------------------------------------------
    # def data_positions(self, index):                                   # unfmt_header
    # #--------------------------------------------------------------------------------
    #     # List of [start, end] positions of the data-chunks
    #     # Data layout: |4i|0..999 elements|4i||4i|999..1999 elements|4i|...|4i|rest data|4i|
    #     # The 4i byte size int's are skipped
    #     data_start = self.startpos + 24
    #     data_limits = list(range(data_start, self.endpos, self.dtype.max*self.dtype.size+8))
    #     data_pos = lambda x: data_limits[ slice( *(1+(i//self.dtype.max) for i in x) ) ]
    #     byte_pos = lambda x: data_start + x*self.dtype.size + 8*(x//self.dtype.max) + 4
    #     # Modify byte_pos by -4/+4 at start/end to match chunk-limits 
    #     limits = ([byte_pos(a)-4]+data_pos((a,b))+[byte_pos(b)+4] for a,b in index)
    #     # Compensate for the 4-byte size int
    #     return ([a+4,b-4] for lim in limits for a,b in pairwise(lim))

    #--------------------------------------------------------------------------------
    def is_char(self):                                                 # unfmt_header
    #--------------------------------------------------------------------------------
        return self.type[0:1] == b'C'

    # #--------------------------------------------------------------------------------
    # def number_of_elements(self, index):                               # unfmt_header
    # #--------------------------------------------------------------------------------
    #     # CHAR data needs special care since it consists of 8 elements
    #     return sum(j-i for i,j in index)*(self.dtype.size if self.is_char() else 1)


#====================================================================================
class unfmt_block:                                                     # unfmt_block
#====================================================================================
    #
    # Block of unformatted Eclipse data
    #
    #  | h e a d e r  |   d a t a     |
    #  |4i|8s|4i|4s|4i|4i|1000 data|4i| 
    #  |    24 bytes  |
    #  |              |4i|8d| 

    #--------------------------------------------------------------------------------
    def __init__(self, header:unfmt_header=None, data=None, 
                 file=None, file_obj=None):                             # unfmt_block
    #--------------------------------------------------------------------------------
        self.header = header
        self._data = data
        self._file = file
        self._file_obj = file_obj
        if DEBUG:
            print(f'Creating {self}')

    #--------------------------------------------------------------------------------
    def __str__(self):                                                  # unfmt_block
    #--------------------------------------------------------------------------------
        return str(self.header)

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                 # unfmt_block
    #--------------------------------------------------------------------------------
        return f'<{self}>'

    #--------------------------------------------------------------------------------
    def __contains__(self, key):                                        # unfmt_block
    #--------------------------------------------------------------------------------
        return self.key() == key

    #--------------------------------------------------------------------------------
    def __del__(self):                                                  # unfmt_block
    #--------------------------------------------------------------------------------
        if DEBUG:
            print(f'Deleting {self}')

    #--------------------------------------------------------------------------------
    def __getattr__(self, item):                                        # unfmt_block
    #--------------------------------------------------------------------------------
        return getattr(self.header, item)

    #--------------------------------------------------------------------------------
    def key(self):                                                      # unfmt_block
    #--------------------------------------------------------------------------------
        return self.header.key.decode().strip()

    #--------------------------------------------------------------------------------
    def type(self):                                                     # unfmt_block
    #--------------------------------------------------------------------------------
        return self.header.type.decode()
        
    #--------------------------------------------------------------------------------
    def read_file(self, sl:slice):                                      # unfmt_block
    #--------------------------------------------------------------------------------        
        self._file_obj.seek(sl.start)
        return self._file_obj.read(sl.stop - sl.start)

    #--------------------------------------------------------------------------------
    def fix_payload_errors(self):                                       # unfmt_block
    #--------------------------------------------------------------------------------
        # Payload positions
        start = self.header.startpos
        #slices = self.header.file_slices()
        slices = self.header._data_slices()
        #print([type(s) for s in slices])
        data_pos = (((s.start-4, s.start), (s.stop, s.stop+4)) for s in slices)
        header_pos = ((start, start+4), (start+20, start+24))
        # Prepend the header positions to the data postions 
        pos = list(chain(header_pos, flatten(data_pos)))
        # Payload sizes of this block
        data = b''.join(self._data[slice(*p)] for p in pos)
        read_sizes = (unpack(ENDIAN + f'{len(data)//4}i', data))
        # Correct payload sizes
        sizes = flatten(2*[b[0]-a[1]] for a,b in batched(pos, 2))
        # Update with correct payload sizes
        sizes_pos = [(s, p) for r,s,p in zip(read_sizes, sizes, pos) if r != s]
        if sizes_pos:
            sizes, pos = zip(*sizes_pos)
            data = pack(ENDIAN + f'{len(sizes)}i', *sizes)
            for i, p in enumerate(pos):
                self._data[slice(*p)] = data[i*4:i*4+4]
        return len(sizes_pos)

    #--------------------------------------------------------------------------------
    def _read_data(self, limit):                                        # unfmt_block
    #--------------------------------------------------------------------------------
        #slices = tuple(flatten(self.header._data_slices(limit)))
        slices = tuple(self.header._data_slices(limit))
        #print(self.key(), slices)
        if self._data:
            # File is mmap'ed 
            data = (self._data[sl] for sl in slices)
        else:
            # File object
            data = (self.read_file(sl) for sl in slices)
        nbytes = sum(sl.stop-sl.start for sl in slices)
        length = nbytes//(1 if self.is_char() else self.dtype.size)
        return unpack(ENDIAN+f'{length}{self.dtype.unpack}', b''.join(data))

    #--------------------------------------------------------------------------------
    def data(self, *index, limit=((None,),), strip=False, unpack=True):  # unfmt_block
    #--------------------------------------------------------------------------------
        if self.header.length == 0:
            return ()
        if index:
            #limit = [index] if len(index)==2 else [(index[0],index[0]+1)]
            limit = [pad(index, 2, fill=index[0]+1)]
        #print(limit, limit2)
        values = iter(self._read_data(limit))
        if self.header.is_char():
            values = (string_split(next(values).decode(), self.header.dtype.size))
            if strip:
                values = (v.strip() for v in values)
        if index:
            pos = slice(*index) if len(index)>1 else index[0]
            return tuple(values)[pos]
        if None in limit[0]:
            return tuple(values) if unpack else (tuple(values),)
        ndata = (-subtract(*l) for l in limit)
        return tuple(take(n, values) for n in ndata)
 
    # #--------------------------------------------------------------------------------
    # def data2(self, *index, raise_error=False, unwrap_tuple=True, nchar=1, strip=False):    # unfmt_block
    # #--------------------------------------------------------------------------------
    #     length = self.header.length
    #     # Abort if no data to return
    #     if length == 0:
    #         return ()
    #     # Return all data if no argument given
    #     if index == () or () in index:
    #         index = ((0, length),)
    #         unwrap_tuple = False
    #     # Fix negative positions, and create tuple if not tuple
    #     fix_lim = lambda x: x+length if x < 0 else x
    #     index = [[fix_lim(ii) for ii in i] if isinstance(i,(tuple,list)) else [fix_lim(i)+a for a in (0,1)] for i in index]
    #     # Out of index error
    #     if any(i>length or i<0 for i in flatten_all(index)):
    #         raise IndexError(f'index out of range for {self.key()}-block of length {length}')
    #     # Fix CHAR data for strings > 8 char (nchar > 1)
    #     if nchar > 1 and self.header.is_char():
    #         index = [[min(i*nchar, length) for i in ind] for ind in index]
    #     # List of data chunk [start,end] positions, stepping over the 4 byte size int before and after 
    #     data_chunks = self.header.data_positions(index)
    #     try:
    #         # Join chunks of data
    #         if self._data:
    #             # Join mmap'ed data
    #             data = b''.join(self._data[a:b] for a,b in data_chunks)
    #         else:
    #             # Read data from file-object
    #             start = self.header.startpos
    #             self._file_obj.seek(start)
    #             bytedata = self._file_obj.read(self.header.endpos - start)
    #             # Shift positions from absolute to relative
    #             data = b''.join(bytedata[a-start:b-start] for a,b in data_chunks)      
    #         # Get number of data elements from the index-list. 
    #         num = self.header.number_of_elements(index)
    #         values = unpack(ENDIAN+f'{num}{self.header.dtype.unpack}', data)
    #     except struct_error as err:
    #         if raise_error:
    #             raise SystemError(f'ERROR Unable to read {self.key()} from {self._file.name}') from err
    #         return None
    #     # Decode string data
    #     if self.header.is_char():
    #         values = tuple(string_split(values[0].decode(), self.header.dtype.size*nchar, strip=strip))
    #     # Return value instead of single-value tuple
    #     if unwrap_tuple and len(values) == 1:
    #         return values[0]
    #     return values



#====================================================================================
class unfmt_file(File):                                                  # unfmt_file
#====================================================================================
    start = None
    end = None
    var_pos = {}

    #--------------------------------------------------------------------------------
    def __init__(self, filename, **kwargs):                              # unfmt_file
    #--------------------------------------------------------------------------------
        super().__init__(filename, **kwargs)
        self.endpos = 0
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
    def __prepare_limits(self, *keylim):                                 # unfmt_file
    #--------------------------------------------------------------------------------
        # Examples of keylim tuple is: ('KEY1', 10, 20, 'KEY2', 'KEY3', 5)
        # Batch the input on keywords (str)
        batch = list(batched_when(keylim, lambda x: isinstance(x, str)))
        # Add last index for keys with only first index, and None for keys with no index
        B = list(pad(a, min(len(a)+1,3), fill=a[1]+1 if len(a)==2 else None) for a in batch)
        # Make indices after key a list [('INTEHEAD', 0, 10)] -> [('INTEHEAD', [0, 10])]
        B = list((a,b) for a,*b in B)
        # Define unique keywords using the first index. Otherwise, values from same keyword 
        # but at different locations will be grouped toghether
        dictkeys = [f'{b[0]}_{b[1][0]}' for b in B]
        # Group on keywords
        groups = list((k, list(zip(*sorted(g)))) for k,g in groupby(B, lambda x:x[0]))
        keys, lims = zip(*groups)
        # Variables from the same keyword must be listed together as input args
        if len(set(keys)) != len(keys):
            raise SyntaxWarning(f'Wrong input: similar keywords must be listed together: {keys}')
        return keys, [l[-1] for l in lims], dictkeys

    #--------------------------------------------------------------------------------
    def blockdata(self, *keylim, limits=None, strip=True, 
                  tail=False, singleton=False, **kwargs):                # unfmt_file
    #--------------------------------------------------------------------------------
        """ 
        Return data in the order of the given keys, not the reading order.
        The keys-list may contain wildcards (*, ?, [seq], [!seq]) 
        """
        keys, limits, dictkeys = self.__prepare_limits(*keylim)
        limits = dict(zip(keys, limits))
        data = {key:None for key in dictkeys}
        blocks = self.blocks
        if tail:
            blocks = self.tail_blocks
        for block in blocks(**kwargs):
            if key:=match_in_wildlist(block.key(), keys):
                limit = limits[key]
                values = block.data(strip=strip, limit=limit, unpack=False)
                dkeys = (f'{key}_{l[0]}' for l in limit)
                for i,dk in enumerate(dkeys):
                    data[dk] = values[i]
            #if not any(val is None for val in data.values()):
            if all(data.values()):
                if singleton:
                    yield tuple(data.values()) 
                else:
                    # Unpack single values
                    values = tuple(v if len(v)>1 else v[0] for v in data.values())
                    yield values if len(values)>1 else values[0]
                data = {key:None for key in dictkeys}

    #--------------------------------------------------------------------------------
    def read2(self, *varnames, **kwargs):                                # unfmt_file
    #--------------------------------------------------------------------------------
        """
        Read data block by block using variable names defined in self.var_pos. 
        The number of returned values match the number of input variables. 
        Single values are unpacked. Use zip to collect values across blocks.
        """
        if missing := [var for var in varnames if var not in self.var_pos]:
            raise SyntaxWarning(f'Missing variable definitions for {type(self).__name__}: {missing}')
        var_pos = list(self.var_pos[var] for var in varnames)
        keylim = flatten_all(zip(repeat(v[0]), group_indices(v[1:])) for v in var_pos)
        nvar = [len(pos) for _,*pos in var_pos]
        for values in self.blockdata(*keylim, **kwargs):
            if any(n>1 for n in nvar):
                # Split values to match number of input variables
                values = iter(values)
                yield [take(n,values)[0] if n==1 else take(n,flatten(values)) for n in nvar]
            else:
                yield values

    #--------------------------------------------------------------------------------
    def read_header(self, data, startpos):                               # unfmt_file
    #--------------------------------------------------------------------------------
        try:
            # Header is 24 bytes, but we skip size int of length 4 before and after
            # Length of data must be 24 - 8 = 16 
            key, length, typ = unpack(ENDIAN+'8si4s', data)
            #key, length, typ = unpack(ENDIAN+'4x8si4s', data) # x is pad byte
            return unfmt_header(key, length, typ, startpos)
        except (ValueError, struct_error):
            return False

    #--------------------------------------------------------------------------------
    def blocks(self, only_new=False, start=None, use_mmap=True, 
               **kwargs):                                                # unfmt_file
    #--------------------------------------------------------------------------------
        if not self.is_file():
            return ()
        startpos = 0
        if only_new:
            startpos = self.endpos
        if start:
            startpos = start
        if self.size() - startpos < 24: # Header is 24 bytes
            return ()
        if use_mmap:
            return self.blocks_from_mmap(startpos, **kwargs)
        return self.blocks_from_file(startpos)


    #--------------------------------------------------------------------------------
    def blocks_from_mmap(self, startpos, write=False):                   # unfmt_file
    #--------------------------------------------------------------------------------
        try:
            with self.mmap(write=write) as data:
                size = data.size()
                pos = startpos
                while pos < size:
                    header = self.read_header(data[pos+4:pos+20], pos)
                    #header = self.read_header(data[pos:pos+20], pos)
                    if not header:
                        return #() #False
                    pos = self.endpos = header.endpos
                    yield unfmt_block(header=header, data=data, file=self.path)
        except ValueError: # Catch 'cannot mmap an empty file'
            return #() #False

    #--------------------------------------------------------------------------------
    def blocks_from_file(self, startpos):                                # unfmt_file
    #--------------------------------------------------------------------------------
        with open(self.path, mode='rb') as file:
            size = self.size()
            pos = startpos
            while pos < size:
                file.seek(pos+4) # +4 to skip size int
                header = self.read_header(file.read(16), pos)
                #header = self.read_header(file.read(20), pos)
                if not header:
                    return #() #False
                pos = self.endpos = header.endpos
                yield unfmt_block(header=header, file_obj=file, file=self.path)


    #--------------------------------------------------------------------------------
    def tail_blocks(self, **kwargs):                                     # unfmt_file
    #--------------------------------------------------------------------------------
        if not self.is_file() or self.size() < 24: # Header is 24 bytes
            return ()
        with open(self.path, mode='rb') as file:
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
                            ### Check if this is a header
                            if size == 16 and data[pos+12:pos+16] in DTYPE:
                                start = data.tell()-4
                                key, length, typ = unpack(ENDIAN+'8si4s', data.read(16))
                                data.seek(4, 1)
                                # Found header
                                break 
                            data.seek(-4, 1)
                        except (ValueError, struct_error):
                            return #()
                    ### Value array
                    #data_start = data.tell()
                    data.seek(start, 0)
                    yield unfmt_block(header=unfmt_header(key, length, typ, start, end), data=data, file=self.path)
                    # yield unfmt_block(key=key, length=length, type=typ, start=start, end=end,
                    #                 data=data, file=self.path)

    #--------------------------------------------------------------------------------
    def fix_errors(self):                                                # unfmt_file
    #--------------------------------------------------------------------------------
        return sum(b.fix_payload_errors() for b in self.blocks(write=True))
        #for block in self.blocks(write=True):
        #    block.fix_payload_errors()

    # #--------------------------------------------------------------------------------
    # def read(self, *varnames, tail=False, start=0, stop=None, step=None, 
    #          drop=None, unpack_single=True, **kwargs):                   # unfmt_file
    # #--------------------------------------------------------------------------------
    #     if not self.is_file():
    #         return
    #     # Check for wrong value names
    #     if missing := [val for val in varnames if val not in self.var_pos]:
    #         err = f'ERROR! Invalid variable names in {self.__class__.__name__}.read() call: {missing}'
    #         raise SystemError(err)
    #     # Get order of keywords
    #     key_order = {v[0]:0 for k,v in self.var_pos.items() if k in varnames}.keys()
    #     # Make list of [key, pos, var]: ['INTEHEAD', 66, 'year']
    #     in_order = [self.var_pos[v]+(v,) for v in varnames]
    #     # Group positions and varnames: 'INTEHEAD': [[207, 'min'],[66, 'year']]
    #     key_pos_name = dict(groupby_sorted(in_order, key=itemgetter(0)))
    #     # Sort on positions for each keyword: 'INTEHEAD': [[66, 'year'],[207, 'min']]
    #     key_pos_name = [(v,sorted(key_pos_name[v], key=itemgetter(0))) for v in key_order]
    #     blocks = self.blocks
    #     end_key = self.end
    #     if tail:
    #         # Read file reversed from tail to top
    #         blocks = self.tail_blocks
    #         # Reverse keyword read-order
    #         key_pos_name = key_pos_name[::-1]
    #         # Reverse start/end keywords
    #         end_key = self.start
    #     # keyword : ([pos1], size1, name1, [pos2], size2, name2)
    #     tmp = {k:flat_list((v[:-1], len(v[:-1]), v[-1]) for v in pn) for k,pn in key_pos_name}
    #     values = flat_list(tmp.values())
    #     var_limits = list(pairwise(accumulate((0,)+values[1::3])))
    #     read_order = values[2::3]
    #     # Get positions for each keyword: INTEHEAD:[66, 207]
    #     keypos_to_read = {k:index_limits(flat_list(v[0::3])) for k,v in tmp.items()}
    #     #print(keypos_to_read)
    #     # Map input to read-order limits: [(1,2), (0,1)] if ('min','year') is input
    #     out_limits = [var_limits[read_order.index(v)] for v in varnames]
    #     if unpack_single:
    #         # Only slice (return a list) for non-single variables
    #         out_slice = [slice(a,b) if b-a>1 else a for a,b in out_limits]
    #     else:
    #         # Always slice
    #         out_slice = [slice(*i) for i in out_limits]
    #     section = []
    #     num = 0
    #     for block in blocks(**kwargs):
    #         if end_key in block:
    #             num += 1
    #         if num < start:
    #             continue
    #         if step and (num-start)%step > 0:
    #             continue
    #         if block and (positions:=keypos_to_read.get(block.key())):
    #             # A pos < 0 in self.var_pos makes index_limits return () 
    #             # which cause the whole array to be read
    #             if () in positions:
    #                 out_slice = [slice(0,block.length)]
    #             section.extend(block.data2(*positions, unwrap_tuple=False))
    #         if end_key in block and section: # and i > 0:
    #             data = [section[s] for s in out_slice]
    #             if not drop or not drop(data):
    #                 yield data
    #             section = []
    #         if stop and num >= stop:
    #             return

    #--------------------------------------------------------------------------------
    def count_sections(self):                                            # unfmt_file
    #--------------------------------------------------------------------------------
        return sum(1 for block in self.blocks() if self.start in block)
        #return len([i for i,block in enumerate(self.blocks()) if self.start in block])

    #--------------------------------------------------------------------------------
    def keys_matching(self, *pattern, sec=0):                           # unfmt_file
    #--------------------------------------------------------------------------------
        """
        Return matching keywords from the first section of blocks
        """
        #blocks = next(self.section_blocks())
        blocks = next(islice(self.section_blocks(), sec, sec+1))
        return [b.key() for b in blocks if match_in_wildlist(b.key(), pattern)]

    #--------------------------------------------------------------------------------
    def blocks_matching(self, *keys):                                    # unfmt_file
    #--------------------------------------------------------------------------------
        step = -1
        for b in self.blocks():
            if self.start in b:
                step = b.data()[0]
            if any(key in b for key in keys):
                yield (step, b)

    #--------------------------------------------------------------------------------
    def section_blocks(self, tail=False):                                # unfmt_file
    #--------------------------------------------------------------------------------
        """
        Return blocks one section at a time
        """
        self.exists(raise_error=True)
        blocks_func = self.blocks
        if tail:
            blocks_func = self.tail_blocks
        step_gen = (i for i,b in enumerate(blocks_func()) if self.start in b)
        step = batched(step_gen, 2)
        a, b = next(step)
        blocks = blocks_func()
        while batch:=tuple(islice(blocks, b-a)):
            yield batch
            a, b = next(step,(0,0))

    #--------------------------------------------------------------------------------
    def section_data(self, start=(), end=(), rename=(), begin=0):        # unfmt_file
    #--------------------------------------------------------------------------------
        
        # Example of start and end format: 
        # start=('SEQNUM', 'startpos'), end=('ENDSOL', 'endpos') 
        # Start by splitting args in keys=('SEQNUM', 'ENDSOL') and attrs=('startpos', 'endpos')
        keys, attrs = zip(start, end)
        pairs = batched(self.blocks_matching(*keys), 2)
        with self.mmap() as filemap:
            for step_pair in pairs:
                step, pair = zip(*step_pair)
                if step[0] < begin:
                    continue
                _slice = slice(*(getattr(p,a) for p,a in zip(pair, attrs)))
                data = filemap[_slice]
                if rename and (names:=[rn for rn in rename if rn[0] in data]):
                    for old, new in names:
                        data = data.replace(old.ljust(8), new.ljust(8))
                yield (step[0], data)
                # yield (step[0], filemap[_slice])

    #--------------------------------------------------------------------------------
    def merge(self, *section_data, progress=lambda x:None, 
              cancel=lambda:None):                                       # unfmt_file
    #--------------------------------------------------------------------------------
        with open(self.path, 'wb') as merge_file:
            n = 0
            for steps_data in zip(*section_data):
                cancel()
                steps, data = zip(*steps_data)
                if len(set(steps)) > 1:
                    raise SystemError(f'ERROR Merged steps are different: {steps}')
                for d in data:
                    merge_file.write(d)
                n += 1
                progress(n)
        return self.path

    #--------------------------------------------------------------------------------
    def assert_no_duplicates(self, raise_error=True):                     # unfmt_file
    #--------------------------------------------------------------------------------
        allowed = (self.start, 'ZTRACER')
        seen = set()
        duplicate = (key for b in self.blocks() if (key:=b.key()) in seen or seen.add(key))
        if (dup:=next(duplicate, None)) and dup not in allowed:
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
    def mode(self):                                                       # DATA_file
    #--------------------------------------------------------------------------------
        return 'backward' if ('READDATA' in self) else 'forward'


    #--------------------------------------------------------------------------------
    def restart(self):                                                  # DATA_file
    #--------------------------------------------------------------------------------
        # Check if this is a restart-run
        file, step = self.get('RESTART')
        if file and step:
            # Get time and step from the restart-file
            file = UNRST_file(file)
            if not file.is_file():
                raise SystemError(f'ERROR Restart file {file.path} is missing')
            days, n = next(((t,s) for t,s in file.read2('time', 'step') if s >= step), (-1,-1))
            #days, n = next(file.read('time', 'step', drop=lambda x:x[1]<step))
            if n != step:
                raise SystemError(f'ERROR Step {step} is missing in restart file {file}')
            return Restart(days=days, step=n, file=file, run=True)
        return Restart()


    #--------------------------------------------------------------------------------
    def binarydata(self, raise_error=False):                              # DATA_file
    #--------------------------------------------------------------------------------
        self.data = super().binarydata(raise_error)
        end = b'END' in self.data and re_search(rb'^[ \t]*\bEND\b', self.data, flags=MULTILINE)
        return end and self.data[:end.end()] or self.data

    #--------------------------------------------------------------------------------
    def check(self, include=True, uppercase=False):                          # DATA_file
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def search(self, key, regex, comments=False):                         # DATA_file
    #--------------------------------------------------------------------------------
        data = self._matching(key)
        if not comments:
            self.data = self._remove_comments(data)
        else:
            self.data = decode(b''.join(data))
        return re_search(regex, self.data, flags=MULTILINE)

    #--------------------------------------------------------------------------------
    def is_empty(self):                                                  # DATA_file
    #--------------------------------------------------------------------------------
        """ Check if file is empty """
        return self._remove_comments() == ''

    #--------------------------------------------------------------------------------
    def include_files(self, data:bytes=None):                           # DATA_file
    #--------------------------------------------------------------------------------
        """ Return full path of INCLUDE files as a generator """
        return (f[0] for f in self._included_file_data(data))

    #--------------------------------------------------------------------------------
    def _included_file_data(self, data:bytes=None):                           # DATA_file
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def grid_files(self, data:bytes=None):                           # DATA_file
    #--------------------------------------------------------------------------------
        data = data or self.binarydata()
        regex = rb"^\s*(?:\bGDFILE\b)(?:\s*--.*\s*|\s*)*'*([^' ]+)['\s]*/.*$"
        files = (self.with_name(m.group(1).decode()) for m in finditer(regex, data, flags=MULTILINE))
        return (file.with_suffix(file.suffix or '.EGRID') for file in files)

    #--------------------------------------------------------------------------------
    def including(self, *files):                                          # DATA_file
    #--------------------------------------------------------------------------------
        """ Add the given files and return self """
        # Added files must be an iterator to avoid an infinite recursive
        # loop when self._added_files is called in _included_file_data
        self._added_files = iter(files)
        # Disable check to avoid check to consume the above iterator
        self._checked = True
        return self

    #--------------------------------------------------------------------------------
    def start(self):                                                      # DATA_file
    #--------------------------------------------------------------------------------
        return self.get('START')[0]

    #--------------------------------------------------------------------------------
    def timesteps(self, start=None, negative_ok=False, missing_ok=False, pos=False, skiprest=False):     # DATA_file
    #--------------------------------------------------------------------------------
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
            raise SystemError(f'ERROR Zero or negative timestep in {self} (check if TSTEP or RESTART oversteps a DATES keyword)')
        if not missing_ok and not tsteps:
            raise SystemError(f'ERROR No TSTEP or DATES in {self} (or the included files)')
        return tsteps if pos else tuple(next(zip(*tsteps)))
        #return pos and tsteps or tuple(next(zip(*tsteps))) # Do not return positions

    #--------------------------------------------------------------------------------
    def report_dates(self):                                               # DATA_file
    #--------------------------------------------------------------------------------
        return [self.start() + timedelta(days=days) for days in accumulate(self.timesteps())]
    
    #--------------------------------------------------------------------------------
    def wellnames(self):                                                  # DATA_file
    #--------------------------------------------------------------------------------
        """
        Return tuple of wellnames from WELSPECS and UNRST file for RESTART runs
        """
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
        with open(self.path, 'w', encoding='utf-8') as f:
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
    def _days(self, time_pos, start=None):                           # DATA_file
    #--------------------------------------------------------------------------------
        """ Return relative timestep in days given a timestep or a datetime """
        last_date = start
        for t,p in time_pos:
            if isinstance(t, datetime):
                dt = t
            else:
                dt = last_date + timedelta(hours=t*24)
            yield (dt-last_date).total_seconds()/86400, p
            last_date = dt
            
    #--------------------------------------------------------------------------------
    def _convert_string(self, values, key):                               # DATA_file
    #--------------------------------------------------------------------------------
        ret = [v for val in values for v in val.split('\n') if v and v != '/']
        return (ret,)

    #--------------------------------------------------------------------------------
    def _convert_float(self, values, key):                                # DATA_file
    #--------------------------------------------------------------------------------
        #mult = lambda x, y : list(repeat(float(y),int(x))) # Process x*y statements
        def mult(x,y):
            # Process x*y statements
            return list(repeat(float(y),int(x)))
        values = ([mult(*n.split('*')) if '*' in n else [float(n)] for n in v.split()] for v in values)
        values = tuple(flat_list(v) for v in values)
        return values or self._getter[key].default

    #--------------------------------------------------------------------------------
    def _convert_date(self, dates, key):                                  # DATA_file
    #--------------------------------------------------------------------------------
        ### Remove possible quotes
        ### Extract groups of 3 from the dates strings 
        dates = (grouper(remove_chars("'/\n", v).split(), 3) for v in dates)
        dates = tuple([datetime.strptime(' '.join(d), '%d %b %Y') for d in date] for date in dates)
        return dates or self._getter[key].default

    #--------------------------------------------------------------------------------
    def _convert_file(self, values, key):                                 # DATA_file
    #--------------------------------------------------------------------------------
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


#====================================================================================
class EGRID_file(unfmt_file):                                            # EGRID_file
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                  # EGRID_file
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.EGRID', **kwargs)
        self.var_pos = {'nx': ('GRIDHEAD', 1),
                        'ny': ('GRIDHEAD', 2),
                        'nz': ('GRIDHEAD', 3)}


#====================================================================================
class INIT_file(unfmt_file):                                              # INIT_file
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                   # INIT_file
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.INIT', **kwargs)
        self.var_pos = {'nx'       : ('INTEHEAD',  8),
                        'ny'       : ('INTEHEAD',  9),
                        'nz'       : ('INTEHEAD', 10),
                        'simulator': ('INTEHEAD', 94)}

    #--------------------------------------------------------------------------------
    def simulator(self):                                                  # INIT_file
    #--------------------------------------------------------------------------------
        sim_codes = {100:'ecl', 300:'ecl', 500:'ecl', 700:'ix', 800:'FrontSim'}
        if sim:=next(self.read2('simulator'), None):
            if sim < 0:
                return 'other simulator'
            return sim_codes[sim]


#====================================================================================
class UNRST_file(unfmt_file):                                            # UNRST_file
#====================================================================================
    start = 'SEQNUM'
    end = 'ENDSOL'
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
                'wells' : ('ZWEL'    , None)}  # First ZWEL in second section 
                                                
    #--------------------------------------------------------------------------------
    def __init__(self, file, wait_func=None, end=None, role=None, 
                 **kwargs):                                              # UNRST_file
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.UNRST', role=role)
        self.end = end or self.end
        self.check = check_blocks(self, start=self.start, end=self.end, wait_func=wait_func, **kwargs)
        self._dim = None
        #self._dates = None

    #--------------------------------------------------------------------------------
    def dim(self):                                                       # UNRST_file
    #--------------------------------------------------------------------------------
        self._dim = self._dim or next(self.read2('nx', 'ny', 'nz'))
        return self._dim

    #--------------------------------------------------------------------------------
    def _cellnr(self, coord):                                            # UNRST_file
    #--------------------------------------------------------------------------------
        dim = self.dim()
        return coord[0]-1 + dim[0]*(coord[1]-1) + dim[0]*dim[1]*(coord[2]-1)

    #--------------------------------------------------------------------------------
    def celldata(self, coord, *keywords):                                # UNRST_file
    #--------------------------------------------------------------------------------
        #self._dates = self._dates or 
        cellnr = self._cellnr(coord)
        args = ((key, cellnr) for key in keywords)
        data = list(zip(*self.blockdata(*args, singleton=True)))
        if missing:=[k for k,d in zip(keywords, data) if not d]:
            raise RuntimeWarning(f'Missing keywords in {self.path}: {missing}')
        return (list(self.dates()), data)

    #--------------------------------------------------------------------------------
    def cellarray(self, *keys, start=0, stop=None, step=None, skip=1):   # UNRST_file
    #--------------------------------------------------------------------------------
        step = step or self.count_sections()
        keys = self.keys_matching(*keys)
        names = [remove_chars('+-', str(k).lower()) for k in keys]
        cellarray = namedtuple('cellarray', ['days', 'dates'] + names)
        dim = self.dim()
        ddd = zip(self.days(), self.dates(), self.blockdata(*keys))
        day_date_data = islice(ddd, start, stop, skip)
        while (batch := tuple(islice(day_date_data, step))):
            days, dates, data = [nparray(d) for d in zip(*batch)]
            data = data.transpose((1,0,2))
            yield cellarray(days, dates, *data.reshape(data.shape[:-1]+dim, order='F'))

    # #--------------------------------------------------------------------------------
    # def cellarray2(self, *keys): # UNRST_file
    # #--------------------------------------------------------------------------------
    #     keys = self.keys_matching(*keys)
    #     names = [remove_chars('+-', str(k).lower()) for k in keys]
    #     cellarray = namedtuple('cellarray', ['days', 'dates'] + names)
    #     shape = (len(keys), -1,) + self.dim()
    #     data = nparray([*zip(*self.blockdata(*keys))]).reshape(shape, order='F')
    #     days = nparray([*self.days()])
    #     dates = nparray([*self.dates()])
    #     return cellarray(days, dates, *data)

    #--------------------------------------------------------------------------------
    def wells(self, **kwargs):                                           # UNRST_file
    #--------------------------------------------------------------------------------
        wells = flatten_all(self.read2('wells', **kwargs))
        unique_wells = set(w for well in wells if (w:=well.strip()))
        return tuple(unique_wells)

    #--------------------------------------------------------------------------------
    def steps(self):                                                     # UNRST_file
    #--------------------------------------------------------------------------------
        return flatten_all(self.read2('step'))
        #return flatten_all(self.blockdata('SEQNUM'))

    #--------------------------------------------------------------------------------
    def end_value(self, var:str):                                        # UNRST_file
    #--------------------------------------------------------------------------------
        #value = next(self.read(var, tail=True), None) or [0]
        return next(self.read2(var, tail=True), None) or 0
        #return value[0]

    #--------------------------------------------------------------------------------
    def end_step(self):                                                  # UNRST_file
    #--------------------------------------------------------------------------------
        return self.end_value('step')

    #--------------------------------------------------------------------------------
    def end_time(self):                                                  # UNRST_file
    #--------------------------------------------------------------------------------
        return self.end_value('time')

    #--------------------------------------------------------------------------------
    def dates(self, **kwargs):                                           # UNRST_file
    #--------------------------------------------------------------------------------
        #data = self.read('day','month','year', **kwargs)
        data = self.read2('day','month','year', **kwargs)
        return (datetime.strptime(f'{d} {m} {y}', '%d %m %Y') for d,m,y in data)

    #--------------------------------------------------------------------------------
    def days(self, **kwargs):                                           # UNRST_file
    #--------------------------------------------------------------------------------
        start = next(self.dates(**kwargs))
        return ((date-start).days for date in self.dates(**kwargs))

    # #--------------------------------------------------------------------------------
    # def step(self, block, step):                                         # UNRST_file
    # #--------------------------------------------------------------------------------
    #     """
    #     Used in sections to get step of current section
    #     """
    #     if block.key() == 'SEQNUM':
    #         return block.data(0)
    #     return step

    # #--------------------------------------------------------------------------------
    # def sections(self, **kwargs):                                        # UNRST_file
    # #--------------------------------------------------------------------------------
    #     return super().sections(init_key=self.start, check_sync=self.step, **kwargs)

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
    def end_time(self):                                                   # RFT_file
    #--------------------------------------------------------------------------------
        # Return data from last check if it exists
        if data := self.check.data():
            return data[-1]
        # Return time-value from tail of file
        return next(self.read2('time', tail=True), None) or 0
        #return time
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
    def welldata(self, keys=(), wells=(), only_new=False, as_array=False, 
                 named=False, start=0, stop=None, step=1, **kwargs):    # UNSMRY_file
    #--------------------------------------------------------------------------------
        """
        named = False   : Returns days, dates and a tuple of key, well, value for each key-well combination
        named = True    : Returns days, dates, and all keys as their names
        only_new = True : Returns only previously un-read data
        as_array = True : Converts key-well data into a numpy array
        """
        if self.is_file() and self.spec.welldata(keys=keys, wells=wells, named=named):
            self.var_pos['welldata'] = ('PARAMS', *self.spec.well_pos())
            reader = self.read2('days', 'welldata', only_new=only_new, **kwargs)
            try:
                #days, data = zip(*self.read('days', 'welldata', only_new=only_new, **kwargs))
                #days, data = zip(*self.read2('days', 'welldata', only_new=only_new, **kwargs))
                days, data = zip(*islice(reader, start, stop, step))
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
    def plot(self, keys=(), wells=(), ncols=1, date=True, 
             args=None, **kwargs):                                      # UNSMRY_file
    #--------------------------------------------------------------------------------
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
            fig.canvas.draw_idle()# draw()

    #--------------------------------------------------------------------------------
    def key_units(self):                                                # UNSMRY_file
    #--------------------------------------------------------------------------------
        Var = namedtuple('Var','unit measure')
        # If 'measures' is None, just use empty string
        kum = zip(*attrgetter('keys', 'units')(self.spec), self.spec.measures or repeat(''))
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
    def energy(self, *wells):                                           # UNSMRY_file
    #--------------------------------------------------------------------------------
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
        #self.wellkey = None

    #--------------------------------------------------------------------------------
    def welldata(self, keys=(), wells=(), named=False):                 # SMSPEC_file
    #--------------------------------------------------------------------------------
        # print('SMSPEC WELLDATA', self.is_file())
        self._inkeys = keys
        if not self.is_file():
            return False
        Data = namedtuple('Data','keys wells measures units', defaults=4*(None,))
        # Do not use mmap here because the SMSPEC-file might 
        # get truncated while mmap'ed causing a bus-error
        self.data = Data(*next(self.blockdata('KEYWORDS', '*NAMES', 'MEASRMNT', 'UNITS', use_mmap=False), ()))
        #if all(self.data):
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
                return True
        return False

    #--------------------------------------------------------------------------------
    def startdate(self):                                                # SMSPEC_file
    #--------------------------------------------------------------------------------
        if start := next(self.blockdata('STARTDAT'), None):
            day, month, year, hour, minute, second = start
            #day, month, year, hour, minute, second = start[0]
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
class text_file(File):                                                    # text_file
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                   # text_file
    #--------------------------------------------------------------------------------
        super().__init__(file, **kwargs)
        self._pattern = {} 
        self._convert = {}
        # self._flavor = None          # 'ecl' or 'ix'

    #--------------------------------------------------------------------------------
    def __contains__(self, key):                                          # text_file
    #--------------------------------------------------------------------------------
        return key.encode() in self.binarydata()
        
    #--------------------------------------------------------------------------------
    def contains_any(self, *keys, head=None, tail=None):                  # text_file
    #--------------------------------------------------------------------------------
        if head:
            data = self.head(size=head)
        elif tail:
            data = self.tail(size=tail)
        else:
            data = self.binarydata()
            keys = (key.encode() for key in keys)
        return any(key in data for key in keys)

    #--------------------------------------------------------------------------------
    def read(self, *var_list):                                            # text_file
    #--------------------------------------------------------------------------------
        #if not self.is_ready():
        #    return ()
        values = []
        #pattern = self._pattern[self._flavor] if self._flavor else self._pattern
        for var in var_list:
            match = matches(file=self.path, pattern=self._pattern[var])
            values.append([self._convert[var](m.group(1)) for m in match])
        return list(zip(*values))



#====================================================================================
class MSG_file(text_file):
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file):
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.MSG')
        #'time' : r'<\s*\bmessage\b\s+\bdate\b="[0-9/]+"\s+time="([0-9.]+)"\s*>',
        self._pattern = {'date' : r'<message date="([0-9/]+)"',
                         'time' : r'<message date="[0-9/]+" time="([0-9.]+)"',
                         'step' : r'\bRESTART\b\s+\bFILE\b\s+\bWRITTEN\b\s+\bREPORT\b\s+([0-9]+)'}
        self._convert = {'date' : lambda x: datetime.strptime(x.decode(),'%d/%m/%Y'),
                         'time' : float,
                         'step' : int}



#====================================================================================
class PRT_file(text_file):                                                # PRT_file
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                    # PRT_file
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.PRT', **kwargs)
        self._pattern['time'] = r'TIME(?:[ a-zA-Z\s/%-]+;|=) +([\d.]+)'
        #self._pattern['time'] = r' (?:Rep    ;|Init   ;|TIME=)\s*([0-9.]+)\s+'
        self._pattern['days'] = self._pattern['time']
        self._convert = {key:float for key in self._pattern}

    #--------------------------------------------------------------------------------
    def end_time(self):                                                    # PRT_file
    #--------------------------------------------------------------------------------
        default = 0
        # timetags = ('TIME=', 'Rep    ;', 'Init   ;')
        # chunks = (txt for txt in self.reversed(size=10*1024) if any(tag in txt for tag in timetags))
        chunks = (txt for txt in self.reversed(size=10*1024) if 'TIME' in txt)
        if data:=next(chunks, None):
            days = findall(self._pattern['time'], data)
            return float(days[-1]) if days else default


#====================================================================================
class PRTX_file(text_file):                                                # PRTX_file
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                    # PRTX_file
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.PRTX', **kwargs)
        self._var_index = {}

    #--------------------------------------------------------------------------------
    def var_index(self):                                                   # PRTX_file
    #--------------------------------------------------------------------------------
        if not self._var_index:
            names = next(self.lines(), '').split(',')
            self._var_index = {name:i for i,name in enumerate(names)}
        return self._var_index

    #--------------------------------------------------------------------------------
    def end_time(self):                                                   # PRTX_file
    #--------------------------------------------------------------------------------
        """
        Note that time in PRTX seems to be delayed compared to PRT and RFT
        """
        time = 0
        if (line:=self.last_line()) and (index:=self.var_index()):
            time = line.split(',')[index['Simulation Time']]
            time = float(time) if time[0] != 'S' else 0 
        return time

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
    def blocks_complete(self, nblocks=1, only_new=True):               # check_blocks
    #--------------------------------------------------------------------------------
        block = None
        start, start_val, end, end_val = 0, 1, 2, 3
        for block in self._unfmt.blocks(only_new=only_new):
            if block.header.key == self._keys[start]:
                #if (data := block.data()):
                if (data := block.data()[0]):
                    self._keys[start_val].append(data[0])
                else:
                    return False
            if block.header.key == self._keys[end]:
                self._keys[end_val] += 1
                if self.steps_complete() and self._keys[end_val] == nblocks:
                    # nblocks complete blocks read, reset counters and return True
                    self._data = self._keys[:start_val+1]
                    self._keys[start_val], self._keys[end_val] = [], 0
                    return True
        return False

    #--------------------------------------------------------------------------------
    def steps_complete(self):
    #--------------------------------------------------------------------------------
        # 1: start_list, 3: end_count
        return len(self._keys[1]) == self._keys[3]


    #--------------------------------------------------------------------------------
    def warn_if_offset(self):
    #--------------------------------------------------------------------------------
        msg = ''
        if (offset := self._unfmt.offset()):
            msg = f'WARNING {self._unfmt} not at end after check, offset is {offset}'
        return msg


    #--------------------------------------------------------------------------------
    def data_saved_maxmin(self, nblocks=1, niter=100, **kwargs):      # check_blocks
    #--------------------------------------------------------------------------------
        """
            Loop for 'niter' iterations until 'nblocks' start/end-blocks are found or end-of-file reached.
        """
        if nblocks == 0:
            return []
        msg = []
        data = []
        n = nblocks
        v = 2
        while n > 0:
            passed = self._wait_func( self.blocks_complete, nblocks=n, limit=niter, timer=self._timer, v=v, **kwargs )
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
    def __init__(self, key=None, length=None, datatype=None, data=(), filemap:mmap=None, start=0, size=0): # fmt_block
    #--------------------------------------------------------------------------------
        self._key = key
        self._length = length
        self._dtype = DTYPE[datatype]
        self.data = data
        self.filemap = filemap
        self.startpos = start
        self.size = size
        self.endpos = start + size

    #--------------------------------------------------------------------------------
    def __str__(self):                                                    # fmt_block
    #--------------------------------------------------------------------------------
        return (f'key={self.key():8s}, type={self._dtype.name:4s},' 
                f'length={self._length:8d}, start={self.startpos:8d}, end={self.endpos:8d}')

    #--------------------------------------------------------------------------------                                                            
    def __repr__(self):                                                   # fmt_block                                                           
    #--------------------------------------------------------------------------------                                                            
        return f'<{type(self)}, key={self.key():8s}, type={self._dtype.name}, length={self._length:8d}>'

    #--------------------------------------------------------------------------------
    def __contains__(self, key:str):                                      # fmt_block
    #--------------------------------------------------------------------------------
        return self.key() == key

    #--------------------------------------------------------------------------------
    def is_last(self):                                                    # fmt_block
    #--------------------------------------------------------------------------------
        return self.endpos == self.filemap.size()

    #--------------------------------------------------------------------------------
    def formatted(self):                                                  # fmt_block
    #--------------------------------------------------------------------------------
        return self.filemap[self.startpos:self.endpos]

    #--------------------------------------------------------------------------------
    def key(self):                                                        # fmt_block
    #--------------------------------------------------------------------------------
        return self._key.decode().strip()
        #return self.keyword.strip()
    
    # #--------------------------------------------------------------------------------
    # def position(self):                                                   # fmt_block
    # #--------------------------------------------------------------------------------
    #     return (self.startpos, self.endpos)

    #--------------------------------------------------------------------------------
    def as_binary(self):                                                  # fmt_block
    #--------------------------------------------------------------------------------
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
                
    #--------------------------------------------------------------------------------
    def print(self):                                                      # fmt_block
    #--------------------------------------------------------------------------------
        print(self._key, self._length, self._dtype.name)

        
#====================================================================================
class fmt_file(File):                                                      # fmt_file
    #
    # Class to handle formatted Eclipse files.
    #
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, filename, **kwargs):                                # fmt_file
    #--------------------------------------------------------------------------------
        super().__init__(filename, **kwargs)
        self.start = None 

    #--------------------------------------------------------------------------------
    def blocks(self):                                                      # fmt_file
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def first_section(self):                                               # fmt_file
    #--------------------------------------------------------------------------------
        # Get number of blocks and size of first section
        secs = ((i,b) for i, b in enumerate(self.blocks()) if 'SEQNUM' in b)
        count, first_block_next_section = tuple(islice(secs, 2))[-1]
        return namedtuple('section','count size')(count, first_block_next_section.startpos)

    #--------------------------------------------------------------------------------
    def section_blocks(self, count=None, with_attr:str=None):              # fmt_file
    #--------------------------------------------------------------------------------
        count = count or self.section_count()
        if with_attr:
            return batched((getattr(b, with_attr)() for b in self.blocks()), count)    
        return batched(self.blocks(), count)

    #--------------------------------------------------------------------------------
    def as_binary(self, outfile, stop:int=None, buffer=100, rename=(),
                  progress=lambda x:None, cancel=lambda:None):             # fmt_file
    #--------------------------------------------------------------------------------
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
                # This is a slow operation for large files
                if (end:=m*section.size) > buffer:
                    resized = True
                    m = 0
                    self.resize(start=0, end=end)
                    sectiondata = self.section_blocks(count=section.count, with_attr='as_binary')
        # Delete the rest of the file if it has been resized
        if resized:
            self.delete()
        return Path(outfile)


#====================================================================================
class FUNRST_file(fmt_file):
#====================================================================================
    #----------------------------------------------------------------------------
    def __init__(self, filename):                           # FUNRST_file
    #----------------------------------------------------------------------------
        super().__init__(filename, suffix='.FUNRST')
        self.start = 'SEQNUM'

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
    def as_unrst(self, outfile=None, **kwargs):  # FUNRST_file 
    #----------------------------------------------------------------------------
        outfile = Path(outfile) if outfile else self.path
        outfile = outfile.with_suffix('.UNRST')
        return UNRST_file( super().as_binary(outfile, **kwargs) )



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
        with open(self.path, 'r', encoding='utf-8') as self.fh:
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
        next(islice(self.fh, n, n), None)
        # for i in range(n):
        #     next(self.fh)
        
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
        """
        Return None if column is empty
        """
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
        with open(self.path, 'r', encoding='utf-8') as fh:
            nb, n = 0, 0
            for line in fh:
                n += 1
                if line[0]==self.tag:
                    nb += 1
                    if nb==2:
                        return int(n)



#####################################################################################
#                                                                                   #
#                                 INTERSECT FILES                                   #
#                                                                                   #
#####################################################################################


#====================================================================================
class AFI_file(File):                                                      # AFI_file
#====================================================================================
    include_regex = rb'^[ \t]*\bINCLUDE\b\s*"*([\w.-]+)"*'

    #--------------------------------------------------------------------------------
    def __init__(self, file, check=False, **kwargs):                       # AFI_file
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.afi', role='Top level Intersect input-file', 
                         ignore_suffix_case=True, **kwargs)
        self.data = None
        if check:
            self.exists(raise_error=True)

    #--------------------------------------------------------------------------------
    def ixf_files(self):
    #--------------------------------------------------------------------------------
        self.data = self.data or self.binarydata()
        matches_ = findall(self.include_regex, self.data, flags=MULTILINE)
        files = (Path(m.decode()) for m in matches_)
        return (self.with_name(file) for file in files if file.suffix.lower() == '.ixf')

    #--------------------------------------------------------------------------------
    def include_files(self):                                               # AFI_file
    #--------------------------------------------------------------------------------
        return (f[0] for f in self._included_file_data(self.data))

    #--------------------------------------------------------------------------------
    def _included_file_data(self, data:bytes=None):                        # AFI_file
    #--------------------------------------------------------------------------------
        """ Return tuple of filename and binary-data for each include file """
        data = data or self.binarydata()
        regex = self.include_regex
        files = (m.group(1).decode() for m in re_compile(regex, flags=MULTILINE).finditer(data))
        # Loop over files included in self
        for file in files:
            inc_file = self.with_name(file)
            inc_data = File(inc_file).binarydata()
            yield (inc_file, inc_data)
            # Recursive call for files included deeper down
            if b'INCLUDE' in inc_data:
                for inc_inc in self._included_file_data(inc_data):
                    yield inc_inc

#====================================================================================
@dataclass
class IXF_node:                                                            # IXF_node
#====================================================================================
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

    #--------------------------------------------------------------------------------
    def __post_init__(self):                                               # IXF_node
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def __str__(self):                                                     # IXF_node
    #--------------------------------------------------------------------------------
        return f'{self.type} \"{self.name}\" {self.brace[0]}{self.content}{self.brace[1]}'

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                     # IXF_node
    #--------------------------------------------------------------------------------
        return (f'<IXF_node type={self.type}, name={self.name}, '
                f'is_table={self.is_table}, is_context={self.is_context} >')

    #--------------------------------------------------------------------------------
    def copy(self):                                                        # IXF_node
    #--------------------------------------------------------------------------------
        return IXF_node(self.type, self.name, self.brace[0]+self.content+self.brace[1])

    #--------------------------------------------------------------------------------
    def set_content(self, rows):                                           # IXF_node
    #--------------------------------------------------------------------------------
        if self.is_context:
            lines = (f'    {k}{"=" if v else ""}{v}' for k,v in rows)
        else:
            width = [max(map(len, col)) for col in zip(*rows)]
            lines = (''.join(f'    {v:>{w}s}' for v,w in zip(row, width)) for row in rows)
        self.content = '\n' + '\n'.join(lines) + '\n'

    #--------------------------------------------------------------------------------
    def lines(self):                                                       # IXF_node
    #--------------------------------------------------------------------------------
        return tuple(split_in_lines(self.content))

    #--------------------------------------------------------------------------------
    def columns(self):                                                     # IXF_node
    #--------------------------------------------------------------------------------
        delimiter = '=' if self.is_context else None # None equals any whitespace
        data = (row for line in self.lines() if (row:=line.split(delimiter)))
        # Use repeat('', 2) to get two columns also if data = (('key',),)
        # instead of data = (('key','value'),)
        return tuple(v[:-1] for v in zip_longest(*data, ('', ''), fillvalue=''))

    #--------------------------------------------------------------------------------
    def rows(self):                                                        # IXF_node
    #--------------------------------------------------------------------------------
        return tuple(zip(*self.columns()))

    #--------------------------------------------------------------------------------
    def as_dict(self):                                                     # IXF_node
    #--------------------------------------------------------------------------------
        return {k:v for k,*v in self.rows()}

    #--------------------------------------------------------------------------------
    def get(self, item):                                                     # IXF_node
    #--------------------------------------------------------------------------------
        return self.as_dict().get(item)

    #--------------------------------------------------------------------------------
    def update(self, node=None): #, on_top=False):                              # IXF_node
    #--------------------------------------------------------------------------------
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
        

#====================================================================================
class IXF_file(File):                                                      # IXF_file
#====================================================================================

    """
    Intersect Input Format (IXF):

    type "name" { (or [)
        content
    } (or ])

    """

    #--------------------------------------------------------------------------------
    def __init__(self, file, check=False, **kwargs):                       # IXF_file
    #--------------------------------------------------------------------------------
        super().__init__(file, role='Intersect input-file', **kwargs)
        self.data = None
        if check:
            self.exists(raise_error=True)
            

    #--------------------------------------------------------------------------------
    def __contains__(self, key):                                           # IXF_file
    #--------------------------------------------------------------------------------
        self.data = self.data or self.binarydata()
        return bool(re_search(rf'^[ \t]*\b{key}\b'.encode(), self.data, flags=MULTILINE))


    #--------------------------------------------------------------------------------
    def binarydata(self, raise_error=False):                               # IXF_file
    #--------------------------------------------------------------------------------
        self.data = super().binarydata(raise_error)
        end_key = b'END_INPUT'
        if end_key in self.data and not b'#'+end_key in self.data:
            self.data, _ = self.data.split(end_key, maxsplit=1)
        return self.data
        # end = b'END' in self.data and re_search(rb'^[ \t]*\bEND\b', self.data, flags=MULTILINE)
        # return end and self.data[:end.end()] or self.data

    #--------------------------------------------------------------------------------
    def node(self, *nodes, convert=(), brace=(rb'{',rb'}')):               # IXF_file
    #--------------------------------------------------------------------------------
        begin, end = brace
        self.data = self.data or self.binarydata()
        if nodes[0] == 'all':
            keys = rb'\w+'
        else:
            keys = '|'.join(nodes).encode()
        # The pattern is explained at https://regex101.com/r/4FVBxU/5
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


#====================================================================================
class IX_input:                                                            # IX_input
#====================================================================================
    STAT_FILE = '.ecl2ix' # Check if ECL-input has changed and new conversion is needed

    #--------------------------------------------------------------------------------
    def __init__(self, case, check=False, **kwargs):         # IX_input
    #--------------------------------------------------------------------------------
        self.afi = AFI_file(case)
        self.path = self.afi.path
        self.ixf_files = [IXF_file(file) for file in self.afi.ixf_files()]
        self._checked = False
        if check:
            self.check()

    #--------------------------------------------------------------------------------
    def __str__(self):                                                     # IX_input
    #--------------------------------------------------------------------------------
        return f'{self.afi}'

    #--------------------------------------------------------------------------------
    def __getattr__(self, item):                                           # IX_input
    #--------------------------------------------------------------------------------
        #print('IX_INPUT GETATTR')
        return getattr(self.path, item)

    #--------------------------------------------------------------------------------
    def __contains__(self, key):                                           # IX_input
    #--------------------------------------------------------------------------------
        return any(key in ixf for ixf in self.ixf_files)

    @classmethod
    #--------------------------------------------------------------------------------
    def need_convert(self, path):                                          # IX_input
    #--------------------------------------------------------------------------------
        path = Path(path)
        afi_file = AFI_file(path)
        data_file = DATA_file(path)
        if afi_file.is_file() and not data_file.is_file():
            # No need for convert
            return
        if not afi_file.is_file():
            if not data_file.is_file():
                raise SystemError('ERROR Eclipse input is missing, unable to create Intersect input.')
            return 'Intersect input is missing for this case, but can be created from the Eclipse input.'
        # Check if input is complete
        if any(file for file in afi_file.include_files() if not file.is_file()):
            return 'Intersect input is incomplete for this case (missing include files).'
        # Check if DATA-file has changed since last convert
        stat_file = path.with_suffix(self.STAT_FILE)
        mtime, size = attrgetter('st_mtime_ns', 'st_size')(data_file.stat())
        if stat_file.is_file():
            old_mtime, old_size = map(int, stat_file.read_text(encoding='utf-8').split())
            if mtime > old_mtime and size > old_size:
                return 'Intersect input exists for this case, but the Eclipse input has changed since the previous convert.'
        else:
            stat_file.write_text(f'{mtime} {size}', encoding='utf-8')
            #stat_file.write_text(f'{data_file.name} {mtime} {size}')


    @classmethod
    #--------------------------------------------------------------------------------
    def from_eclipse(self, path, progress=None, abort=None, freq=20):      # IX_input
    #--------------------------------------------------------------------------------
        # Create IX input from Eclipse input
        if not DATA_file(path).is_file():
            raise SystemError('ERROR Eclipse input is missing, convert aborted...')
        path = Path(path)
        cmd = ['eclrun', 'ecl2ix', path]
        #msg = 'Creating Intersect input from Eclipse input'
        # How often to check if convert is completed
        sec = 1/freq
        logfile = path.with_name(ECL2IX_LOG)
        with open(logfile, 'w', encoding='utf-8') as log:
            popen = Popen(cmd, stdout=log, stderr=STDOUT)
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
            mtime, size = attrgetter('st_mtime_ns', 'st_size')(DATA_file(path).stat())
            path.with_name(self.STAT_FILE).write_text(f'{mtime} {size}', encoding='utf-8')
            return True, logfile


    #--------------------------------------------------------------------------------
    def check(self, include=True):                                         # IX_input
    #--------------------------------------------------------------------------------
        self._checked = True
        # Check if top level afi-file exist
        self.afi.exists(raise_error=True)
        # Check if included files exists
        if include and (missing := [f for f in self.include_files() if not f.is_file()]):
            raise SystemError(f'ERROR {list2text([f.name for f in missing])} '
                              'included from {self} is missing in folder {missing[0].parent}')
        return True

    #--------------------------------------------------------------------------------
    def files_matching(self, *keys):                                        # IX_input
    #--------------------------------------------------------------------------------
        """ Return only ixf-files that match the given keys """
        return (ixf for ixf in self.ixf_files if any(key in ixf for key in keys))

    #--------------------------------------------------------------------------------
    def include_files(self):                                               # IX_input
    #--------------------------------------------------------------------------------
        return self.afi.include_files()

    #--------------------------------------------------------------------------------
    def including(self, *files):                                           # IX_input
    #--------------------------------------------------------------------------------
        """ For compatibility with DATA_file """
        return self

    #--------------------------------------------------------------------------------
    def nodes(self, *types, files=None, table=False, context=False, **kwargs): # IX_input
    #--------------------------------------------------------------------------------
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
        contexts = flatten(file.node(*types, **kwargs) for file in files)
        tables = flatten(file.node(*types, brace=(b'\\[',b'\\]'), **kwargs) for file in files)
        if table and context:
            return (node for node in chain(contexts, tables) if node.content)
        if table:
            return (table for table in tables if table.content)
        if context:
            return (context for context in contexts if context.content)
        return contexts
            

    #--------------------------------------------------------------------------------
    def get_node(self, node):                                              # IX_input
    #--------------------------------------------------------------------------------
        return next(self.nodes(node.type, table=node.is_table, context=node.is_context), None)


    #--------------------------------------------------------------------------------
    def start(self):                                                       # IX_input
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def _timestep_files(self):                                                  # IX_input
    #--------------------------------------------------------------------------------
        date_files = [ixf for ixf in self.ixf_files if 'DATE' in ixf]
        field_files = [ixf for ixf in date_files if next(ixf.node('FieldManagement'), None)]
        if field_files:
            return field_files[0]
        if date_files:
            return date_files[-1]
            #return date_files[0]

    #--------------------------------------------------------------------------------
    def timesteps(self, start=None, **kwargs):                             # IX_input
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def report_dates(self):                                                # IX_input
    #--------------------------------------------------------------------------------
        return [self.start() + timedelta(days=days) for days in accumulate(self.timesteps())]
    
    #--------------------------------------------------------------------------------
    def wellnames(self):                                                   # IX_input
    #--------------------------------------------------------------------------------
        return tuple(set(node.name for node in self.nodes('Well')))
        #return tuple(set(node.name for node in self.nodes('WellDef')))

    #--------------------------------------------------------------------------------
    def wells(self):                                                   # IX_input
    #--------------------------------------------------------------------------------
        return tuple(set((well.name, _type[0]) for well in self.nodes('Well') if (_type:=well.get('Type'))))

    #--------------------------------------------------------------------------------
    def summary_keys(self, matching=()):                                   # IX_input
    #--------------------------------------------------------------------------------
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


    #--------------------------------------------------------------------------------
    def make_iorsim_compatible(self, backup='_NO_IORSIM_'):                # IX_input
    #--------------------------------------------------------------------------------
        """
        Update nodes in IXF-files with values found in 'iorsim_ix_fix.ixf'
        """
        ior = IX_input('IORlib/iorsim_ix_fix')
        ior_nodes = list(ior.nodes('all', context=True, table=True))
        # Get relevant nodes from IX files or None if missing 
        ix_nodes = [self.get_node(node) for node in ior_nodes]
        # Syntax with 'add_property' currently not supported
        if any('add_property' in n.content for n in ix_nodes if n):
            raise SystemError(
                'ERROR Currently unable to update nodes with *add_property* statements')
        # Use node name from ix_nodes if node type exists
        node_name = {n.type:n.name for n in ior_nodes}
        node_name.update({n.type:n.name for n in ix_nodes if n})
        # Update ix_nodes with values from ior_nodes
        for i,ior in enumerate(ior_nodes):
            if ix_nodes[i]:
                ix_nodes[i].update(ior)
            else:
                ix_nodes[i] = ior.copy()
            ix_nodes[i].name = node_name[ix_nodes[i].type]
        # Get the name of the file holding the relevant nodes
        filename = list(set(n.file for n in ix_nodes if n.file))
        if len(filename) > 1:
            print(f'WARNING! More than one filename in IX_input.make_iorsim_compatible(): {filename}')
        file = File(filename[0])
        if backup:
            # Returns None if backup-file already exists
            backup = file.backup(tag=backup)
        # Place new nodes just after the last updated nodes
        max_pos = 2*[max(max(n.pos for n in ix_nodes if n.pos))]
        # Split text and pos in separate tuples
        text, pos = zip(*[(str(n), n.pos or max_pos) for n in ix_nodes])
        file.replace_text(text, pos)
        if backup:
            # Add comment at top of updated file
            comment = ( '#----------------------------------------------------\n'
                        '# This file is updated to be compatible with IORSim.\n'
                       f"# The original file is renamed to '{backup.name}'\n"
                        '#----------------------------------------------------\n\n')
            file.write_text(comment + file.as_text())
        return (file, backup)


    #--------------------------------------------------------------------------------
    def mode(self):                                                        # IX_input
    #--------------------------------------------------------------------------------
        return 'forward'

    #--------------------------------------------------------------------------------
    def restart(self):                                                     # IX_input
    #--------------------------------------------------------------------------------
        return Restart()
