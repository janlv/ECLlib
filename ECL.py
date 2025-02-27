
# -*- coding: utf-8 -*-

from contextlib import contextmanager
from curses.ascii import RS
from dataclasses import dataclass, field
from fnmatch import fnmatch
from hmac import new
from itertools import chain, product, repeat, accumulate, groupby, zip_longest, islice
from math import hypot, prod
from operator import attrgetter, itemgetter, neg, sub as subtract
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
from numpy import (fromstring, int32, float32, float64, bool_ as np_bool, array as nparray, roll, stack, sum as npsum,
                   zeros, ones, atleast_1d, argsort, newaxis, zeros_like, abs as npabs, any as npany)
from matplotlib.pyplot import figure as pl_figure
from pandas import DataFrame
from pyvista import CellType, UnstructuredGrid

from .utils import (batched, batched_when, cumtrapz, date_range, decode, ensure_bytestring, expand_pattern, flatten,
                    index_limits, last_line, match_in_wildlist, nth, pad, slice_range, tail_file, head_file,
                    flat_list, flatten_all, grouper, list2text, pairwise, remove_chars,
                    list2str, float_or_str, matches, split_by_words, string_split, split_in_lines, take,
                    missing_elements)
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

#------------------------------------------------------------------------------------
def check_flowvol(root, phase='WAT', resrate=True):
#------------------------------------------------------------------------------------
    dt_accuracy = 1e-6
    rft = RFT_file(root)
    unrst = UNRST_file(root)
    phlow = phase.lower()
    Volumes = namedtuple('Volumes', 'time dt dQdt dV diff')
    for (bflow, pvt), wflow in zip(unrst.block_flows(phase, resrate), rft.well_flows()):
        if any(abs(wf.days - bflow.days) > dt_accuracy for wf in wflow):
            raise ValueError('The timesteps of the UNRST- and RFT-file dont match: '
                             f'{bflow.days}, {[wf.days for wf in wflow]}')
        #print('unrst:', bflow.days, 'rft:', [wf.days for wf in wflow])
        Qin = bflow.Qin
        # Add inflow from injectors
        for flow in wflow:
            Qw = getattr(flow, f'{phlow}rate')
            if any(q < 0 for q in Qw):
                # Well penetrates in z-dir (2)
                Qin[*flow.pos, 2] += npabs(Qw)
        #print('dQ:', (Qin - bflow.Qout)[...,0].squeeze(), ', dt:', bflow.dt)
        dQdt = (Qin - bflow.Qout) * bflow.dt
        dQdt = npsum(dQdt, axis=-1)
        diff = zeros_like(dQdt)
        mask = npabs(dQdt) > 1e-9
        diff[mask] = 100*(bflow.dVol[mask]-dQdt[mask])/dQdt[mask]
        yield Volumes(bflow.days, bflow.dt, dQdt, bflow.dVol, diff)


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
    start: datetime = None
    days: float = 0
    step: int = 0
    # file: str = ''
    # run : bool = False


# #--------------------------------------------------------------------------------
# def catch_error_write_log(error=(Exception,), error_msg=None, echo=None, echo_msg=None): 
# #--------------------------------------------------------------------------------
#     def decorator(func):
#         def inner(*args, **kwargs):
#             try:
#                 return func(*args, **kwargs)
#             except error as err:
#                 if error_msg:
#                     raise SystemError(error_msg) from err
#             if echo:
#                 if callable(echo):
#                     echo(echo_msg)
#                 else:
#                     print(echo_msg)
#             return inner
#         return decorator



#====================================================================================
class File:                                                                    # File
#====================================================================================
    """
    A class representing a file, allowing various operations such as reading, writing,
    memory mapping, and manipulation of file metadata. This class also supports file
    backup, text replacement, and more.

    Attributes:
        path (Path): The resolved path of the file.
        role (str): Optional role identifier for the file.
        debug (bool): Debug flag, prints debug messages when True.
    """

    #--------------------------------------------------------------------------------
    def __init__(self, filename, suffix=None, role=None, ignore_suffix_case=False, exists=False):          # File
    #--------------------------------------------------------------------------------
        """
        Initializes a File object with a given filename and optional suffix, role,
        and case sensitivity for suffix matching.

        Args:
            filename (str): The name or path of the file.
            suffix (str, optional): A suffix to apply to the filename.
            role (str, optional): A role description for the file.
            ignore_suffix_case (bool, optional): Whether to ignore case when matching suffix.
            exists (bool, optional): If True, will only set the path if the file exists.
        """        
        if isinstance(filename, File):
            filename = filename.path
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
        """
        Returns a string representation of the File object.

        Returns:
            str: String representation of the File object including its path and role.
        """
        return f"<{self.__class__.__name__}, file={self.path}, role={self.role or None}>"

    #--------------------------------------------------------------------------------
    def __str__(self):                                                         # File
    #--------------------------------------------------------------------------------
        """
        Returns a human-readable string representation of the file.

        Returns:
            str: The role and name of the file.
        """
        return f'{self.role}{self.name}'

    #--------------------------------------------------------------------------------
    def __del__(self):                                                         # File
    #--------------------------------------------------------------------------------
        """
        Destructor for the File object, optionally printing debug information if enabled.
        """
        if self.__class__.__name__ == File.__name__ and self.debug:
            print(f'Deleting {repr(self)}')

    #--------------------------------------------------------------------------------
    def __getattr__(self, item):                                               # File
    #--------------------------------------------------------------------------------
        """
        Attempts to retrieve the requested attribute from the internal path object,
        or returns None if the file path is not set.

        Args:
            item (str): Attribute name.

        Returns:
            Any: The value of the requested attribute or a function returning None if callable.
        """
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
        """
        Memory-maps the file for efficient file I/O operations.

        Args:
            write (bool, optional): If True, opens the file in write mode.

        Yields:
            mmap: A memory-mapped object for the file.
        """
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
        """
        Resizes the file by removing content between the given start and end positions.
        NB! Very slow for large files, use with caution!
        
        Args:
            start (int): Start position in the file.
            end (int): End position in the file.

        Raises:
            SyntaxError: If start is not less than end.
        """
        # NB! Very slow for large files, use with caution!
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
    def binarydata(self, pos=None, raise_error=False):                                   # File
    #--------------------------------------------------------------------------------
        """
        Reads the file as binary data.

        Args:
            pos (tuple, optional): Only read from pos[0] to pos[1] 
            raise_error (bool, optional): If True, raises an error if the file does not exist.

        Returns:
            bytes: The binary content of the file.
        """
        # Open as binary file to avoid encoding errors
        if self.is_file():
            with open(self.path, 'rb') as f:
                if pos:
                    f.seek(pos[0])
                    return f.read(pos[1]-pos[0])
                return f.read()
        if raise_error:
            raise SystemError(f'File {self} does not exist')
        return b''
 
    #--------------------------------------------------------------------------------
    def as_text(self, **kwargs):                                               # File
    #--------------------------------------------------------------------------------
        """
        Reads the file and returns its content as text.

        Returns:
            str: The textual content of the file.
        """
        return self.binarydata(**kwargs).decode()

    #--------------------------------------------------------------------------------
    def delete(self, raise_error=False, echo=False):                           # File
    #--------------------------------------------------------------------------------
        """
        Deletes the file.

        Args:
            raise_error (bool, optional): If True, raises an error if deletion fails.
            echo (bool, optional): If True, prints a message upon successful deletion.
        """
        if not self.path:
            return
        try:
            self.path.unlink(missing_ok=True)
        except (PermissionError, FileNotFoundError) as error:
            if raise_error:
                raise SystemError(f'Unable to delete {self}: {error}') from error
        if echo:
            msg = f'Deleted {self}'
            if callable(echo):
                echo(msg)
            else:
                print(msg)

    #--------------------------------------------------------------------------------
    def rename(self, newname, raise_error=False, echo=False):                  # File
    #--------------------------------------------------------------------------------
        """
        Rename the file.

        Args:
            newname (str, Path, File): Name of the new file 
            raise_error (bool, optional): If True, raises an error if deletion fails.
            echo (bool, optional): If True, prints a message upon successful renaming.
        """
        if not self.path:
            return
        if isinstance(newname, File):
            newname = newname.path
        try:
            self.path.rename(newname)
        except (PermissionError, FileNotFoundError) as error:
            if raise_error:
                raise SystemError(f'Unable to rename {self}: {error}') from error
        if echo:
            msg = f'Renamed {self.path} --> {newname}'
            if callable(echo):
                echo(msg)
            else:
                print(msg)


    #--------------------------------------------------------------------------------
    def is_file(self):                                                         # File
    #--------------------------------------------------------------------------------
        """
        Checks if the path points to a file.

        Returns:
            bool: True if the path is a file, False otherwise.
        """
        if not self.path:
            return False
        return self.path.is_file()

    #--------------------------------------------------------------------------------
    def with_name(self, file):                                                 # File
    #--------------------------------------------------------------------------------
        """
        Returns a new file path with the given filename.

        Args:
            file (str): The new filename.

        Returns:
            Path: A new path object with the updated filename.
        """
        if not self.path:
            return
        return (self.path.parent/file).resolve()

    #--------------------------------------------------------------------------------
    def with_tag(self, head:str='', tail:str=''):                              # File
    #--------------------------------------------------------------------------------
        """
        Creates a new file path with the given head and tail added to the stem of the file.

        Args:
            head (str): String to prepend to the file's stem.
            tail (str): String to append to the file's stem.

        Returns:
            Path: A new path object with the updated name.
        """
        if not self.path:
            return
        return self.path.parent/(head + self.path.stem + tail + self.path.suffix)

    #--------------------------------------------------------------------------------
    def with_suffix(self, suffix, ignore_case=False, exists=False):            # File
    #--------------------------------------------------------------------------------
        """
        Adds a suffix to the file's name, with optional case-insensitive matching and 
        checking if the file exists.

        Args:
            suffix (str): The suffix to append to the file's name.
            ignore_case (bool, optional): If True, ignores case when matching suffix.
            exists (bool, optional): If True, only returns an existing file path with the suffix.
                                     If False, return path with the suffix (ignore existance) 
            
        Returns:
            Path: A new path object with the suffix applied, or None if no matching file is found.
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
        """
        Performs a glob pattern search in the file's directory.

        Args:
            pattern (str): The glob pattern to match.

        Returns:
            generator: A generator yielding paths that match the pattern.
        """
        if not self.path:
            return ()
        return self.path.parent.glob(self.path.stem + pattern)

    #--------------------------------------------------------------------------------
    def exists(self, raise_error=False):                                       # File
    #--------------------------------------------------------------------------------
        """
        Checks if the file exists.

        Args:
            raise_error (bool, optional): If True, raises an error if the file does not exist.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        if self.is_file():
            return True
        if raise_error:
            if self.path.parent.is_dir():
                raise SystemError(f'ERROR {self} is missing in folder {self.path.parent}')
            raise SystemError(f'ERROR {self} not found because folder {self.path.parent} is missing')
        return False

    #--------------------------------------------------------------------------------
    def __stat(self, attr):                                                    # File
    #--------------------------------------------------------------------------------
        """
        Retrieves a specific file metadata attribute.

        Args:
            attr (str): The name of the attribute to retrieve.

        Returns:
            Any: The value of the specified attribute, or -1 if the file does not exist.
        """
        if self.is_file():
            return getattr(self.path.stat(), attr)
        return -1

    #--------------------------------------------------------------------------------
    def size(self):                                                            # File
    #--------------------------------------------------------------------------------
        """
        Returns the size of the file in bytes.

        Returns:
            int: The file size in bytes, or -1 if the file does not exist.
        """
        return self.__stat('st_size')

    #--------------------------------------------------------------------------------
    def creation_time(self):                                                   # File
    #--------------------------------------------------------------------------------
        """
        Returns the file creation time.

        Returns:
            float: The file's creation time as a timestamp, or -1 if the file does not exist.
        """
        return self.__stat('st_ctime')

    #--------------------------------------------------------------------------------
    def tail(self, **kwargs):                                                  # File
    #--------------------------------------------------------------------------------
        """
        Returns the last few lines of the file.

        Returns:
            str: The last lines of the file.
        """
        return next(tail_file(self.path, **kwargs), '')

    #--------------------------------------------------------------------------------
    def reversed(self, **kwargs):                                             # File
    #--------------------------------------------------------------------------------
        """
        Reads the file's content in reverse order.

        Returns:
            generator: A generator yielding lines in reverse order.
        """
        return tail_file(self.path, **kwargs)

    #--------------------------------------------------------------------------------
    def head(self, **kwargs):                                                  # File
    #--------------------------------------------------------------------------------
        """
        Returns the first few lines of the file.

        Returns:
            str: The first lines of the file.
        """
        return next(head_file(self.path, **kwargs), '')

    #--------------------------------------------------------------------------------
    def lines(self):                                                           # File
    #--------------------------------------------------------------------------------
        """
        Reads the file line by line.

        Returns:
            generator: A generator yielding lines from the file.
        """
        if self.is_file():
            with open(self.path, 'r', encoding='utf-8') as file:
                while line:=file.readline():
                    yield line
        return ()

    #--------------------------------------------------------------------------------
    def line_matching(self, word):                                             # File
    #--------------------------------------------------------------------------------
        """
        Finds the first line in the file that contains the specified word.

        Args:
            word (str): The word to search for in the file.

        Returns:
            str: The first line that contains the word, or None if not found.
        """
        return next((line for line in self.lines() if word in line), None)

    #--------------------------------------------------------------------------------
    def last_line(self):                                                       # File
    #--------------------------------------------------------------------------------
        """
        Returns the last line of the file.

        Returns:
            str: The last line of the file.
        """
        return last_line(self.path)

    #--------------------------------------------------------------------------------
    def backup(self, tag, overwrite=False):                                    # File
    #--------------------------------------------------------------------------------
        """
        Creates a backup of the file with an optional tag appended to the filename.

        Args:
            tag (str): The tag to append to the filename.
            overwrite (bool, optional): If True, overwrites an existing backup file.

        Returns:
            Path: The path to the backup file.
        """
        backup_file = self.path.with_name(f'{self.stem}{tag}{self.suffix}')
        if overwrite or not backup_file.exists():
            copy(self.path, backup_file)
            return backup_file


    #--------------------------------------------------------------------------------
    def replace_text(self, text=(), pos=()):                                   # File
    #--------------------------------------------------------------------------------
        """
        Replaces or appends text in the file at specified positions.

        Args:
            text (tuple): A tuple of strings to replace or append.
            pos (tuple): A tuple of (start, end) positions for replacement or appending.
                         Replace text if pos is a (start, len+start) tuple, append 
                         '\n'+text if pos is a (start, start) tuple
        Returns:
            None
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

    #--------------------------------------------------------------------------------
    def append_bytes(self, data):                                              # File
    #--------------------------------------------------------------------------------
        with open(self.path, 'ab') as file:
            file.write(data)

    #--------------------------------------------------------------------------------
    def write_bytes(self, data):                                              # File
    #--------------------------------------------------------------------------------
        with open(self.path, 'wb') as file:
            file.write(data)


#====================================================================================
class unfmt_header:                                                    # unfmt_header
#====================================================================================
    #         | h e a d e r  |     d a t a     |     d a t a     |    d a t a      |
    #         |4i|8s|4i|4s|4i|4i| 1000 data |4i|4i| 1000 data |4i|4i| 1000 data |4i|
    #  bytes  |      24      |4 | 1000*size |  8  | 1000*size |  8  | 1000*size |4 |

    #pack_format = 'i8si4si'

    #--------------------------------------------------------------------------------
    def __init__(self, key:str=b'', length:int=0, type:str=b'',
                 startpos:int=0, endpos:int=0):                        # unfmt_header
    #--------------------------------------------------------------------------------
        """
        Initialize the unfmt_header object, which represents the header for unformatted data.

        Parameters:
        key (str): The identifier for the data block.
        length (int): The length of the data array.
        type (str): The data type.
        startpos (int): Starting position of the data block.
        endpos (int): Ending position of the data block. 
                      If not provided, it is calculated from the data length and type.
        """
        self._key = key
        self.length = length  # datalength
        self.type = type
        self.startpos = startpos
        self.endpos = endpos
        self.dtype = DTYPE[self.type]
        self.bytes = self.length*self.dtype.size # databytes
        if not endpos:
            if self.length:
                # self._data_pos() gives start of last data value
                self.endpos = self._data_pos(self.length-1) + self.dtype.size + 4
            else:
                # No data, only header
                self.endpos = self.startpos + 24

    @classmethod
    #--------------------------------------------------------------------------------
    def from_bytes(cls, _bytes, startpos=0):                           # unfmt_header
    #--------------------------------------------------------------------------------
        try:
            # Header is 24 bytes, but we skip size int of length 4 before and after
            # Length of data must be 24 - 8 = 16
            key, length, typ = unpack(ENDIAN+'8si4s', _bytes)
            return cls(key, length, typ, startpos)
        except (ValueError, struct_error):
            return False

    #--------------------------------------------------------------------------------
    def as_bytes(self):                                               # unfmt_header
    #--------------------------------------------------------------------------------
        #  | h e a d e r  |
        #  |4i|8s|4i|4s|4i|
        #key = self.key if isinstance(self.key, bytes) else self.key.encode()
        #data = (16, ensure_bytes(self.key), self.length, ensure_bytes(self.type), 16)
        return pack(ENDIAN+'i8si4si', 16, self._key, self.length, ensure_bytestring(self.type), 16)

    #--------------------------------------------------------------------------------
    def __str__(self):                                                 # unfmt_header
    #--------------------------------------------------------------------------------
        """
        Return a string representation of the unfmt_header object.
        """
        return (f'key={self._key.decode():8s}, type={self.type.decode():4s}, bytes={self.bytes:8d},' 
                f'length={self.length:8d}, start={self.startpos:8d}, end={self.endpos:8d}')

    #--------------------------------------------------------------------------------
    def _data_pos(self, pos):                                          # unfmt_header
    #--------------------------------------------------------------------------------
        """
        Return the absolute file position for the given relative index in the data array.

        Parameters:
        pos (int): The index of the data item in the array.

        Returns:
        int: The absolute byte position in the file.
        """
        #  | h e a d e r  |   d a t a     |   d a t a     |   d a t a     |
        #  |4i|8s|4i|4s|4i|4i|1000 data|4i|4i|1000 data|4i|4i|1000 data|4i| 
        #  |    24 bytes  |
        # Add 8 payload bytes (two ints) at the transition between data chunks
        return self.startpos + 24 + 4 + pos*self.dtype.size + 8*(pos//self.dtype.max)

    #--------------------------------------------------------------------------------
    def _data_slices(self, limits=((None,),)):                          # unfmt_header
    #--------------------------------------------------------------------------------
        """
        Calculate byte slices for accessing the data between the given limits.

        Parameters:
        limits (tuple): A tuple representing start and end indices for slicing the data.

        Returns:
        generator: A generator yielding slice objects representing byte positions.
        """
        dtype = self.dtype
        # Extend limit to whole range if None is given
        flat_lim = tuple(flatten(((0, self.length) if None in l else l for l in limits)))
        # Check for out-of-bounds limits
        if oob_err := [l for l in flat_lim if l<0 or l>self.length]:
            raise SyntaxWarning(
                f'{self._key.decode().strip()}: index {oob_err} is out of bounds {(0, self.length)}')
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
        # Pull pairs of indices, and return slices 
        return (slice(*l) for l in batched(flatten_all(lims), 2))

    #--------------------------------------------------------------------------------
    def is_char(self):                                                 # unfmt_header
    #--------------------------------------------------------------------------------
        """
        Check if the data type is a character (string) type.

        Returns:
        bool: True if the data type is character-based, False otherwise.
        """
        return self.type[0:1] == b'C'




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
        """
        Initialize the unfmt_block object, which represents a block of unformatted data.

        Parameters:
        header (unfmt_header): The header associated with the block.
        data (optional): The data associated with the block.
        file (optional): The file where the block data is stored.
        file_obj (optional): A file object for accessing the block data.
        """
        self.header = header
        self._data = data
        self._file = file
        self._file_obj = file_obj
        if DEBUG:
            print(f'Creating {self}')

    @classmethod
    #--------------------------------------------------------------------------------
    def from_data(cls, key:str, data, _dtype):                          # unfmt_block
    #--------------------------------------------------------------------------------
        dtype = {'int':b'INTE', 'float':b'REAL', 'double':b'DOUB', 'bool':b'LOGI', 'char':b'CHAR'}[_dtype]
        header = unfmt_header(ensure_bytestring(key.ljust(8)), len(data), dtype)
        return cls(header, data)

    #--------------------------------------------------------------------------------
    def binarydata(self):                                               # unfmt_block
    #--------------------------------------------------------------------------------
        sl = slice(self.header.startpos, self.header.endpos)
        if self._data:
            return self._data[sl]
        return self.read_file(sl)


    #--------------------------------------------------------------------------------
    def __str__(self):                                                  # unfmt_block
    #--------------------------------------------------------------------------------
        """
        Return a string representation of the unfmt_block object.
        """
        return str(self.header)

    #--------------------------------------------------------------------------------
    def __repr__(self):                                                 # unfmt_block
    #--------------------------------------------------------------------------------
        """
        Return a formal string representation of the unfmt_block object.
        """
        return f'<{self}>'

    #--------------------------------------------------------------------------------
    def __contains__(self, key):                                        # unfmt_block
    #--------------------------------------------------------------------------------
        """
        Check if a specific key is present in the block.

        Parameters:
        key (str): The key to check.

        Returns:
        bool: True if the key is present, False otherwise.
        """
        return self.key() == key

    #--------------------------------------------------------------------------------
    def __del__(self):                                                  # unfmt_block
    #--------------------------------------------------------------------------------
        """
        Destructor for unfmt_block object. Prints debug information if DEBUG is enabled.
        """
        if DEBUG:
            print(f'Deleting {self}')

    #--------------------------------------------------------------------------------
    def __getattr__(self, item):                                        # unfmt_block
    #--------------------------------------------------------------------------------
        """
        Delegate attribute access to the associated unfmt_header if the attribute is not found.

        Parameters:
        item (str): The attribute name.

        Returns:
        object: The value of the attribute from the umfmt_header.
        """
        return getattr(self.header, item)

    #--------------------------------------------------------------------------------
    def key(self):                                                      # unfmt_block
    #--------------------------------------------------------------------------------
        """
        Return the decoded key of the block.

        Returns:
        str: The key as a string.
        """
        return self.header._key.decode().strip()

    #--------------------------------------------------------------------------------
    def type(self):                                                     # unfmt_block
    #--------------------------------------------------------------------------------
        """
        Return the decoded type of the block.

        Returns:
        str: The type of the block.
        """
        return self.header.type.decode()
        
    #--------------------------------------------------------------------------------
    def read_file(self, sl:slice):                                      # unfmt_block
    #--------------------------------------------------------------------------------        
        """
        Read data from a file within the given slice range.

        Parameters:
        sl (slice): A slice object specifying the range of bytes to read.

        Returns:
        bytes: The read data.
        """
        self._file_obj.seek(sl.start)
        return self._file_obj.read(sl.stop - sl.start)

    #--------------------------------------------------------------------------------
    def fix_payload_errors(self):                                       # unfmt_block
    #--------------------------------------------------------------------------------
        """
        Fix errors in the payload sizes by comparing the stored sizes with calculated sizes.

        Returns:
        int: The number of errors fixed.
        """
        # Payload positions
        start = self.header.startpos
        slices = self.header._data_slices()
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
        values = iter(self._read_data(limit))
        if self.header.is_char():
            values = (string_split(next(values).decode(), self.header.dtype.size))
            if strip:
                values = (v.strip() for v in values)
        if index:
            pos = slice(*index) if len(index)>1 else index[0]
            return tuple(values)[pos]
        #print('LIMIT[0]', limit[0])
        if None in limit[0]:
            return tuple(values) if unpack else (tuple(values),)
        ndata = (-subtract(*l) for l in limit)
        return tuple(take(n, values) for n in ndata)

    #--------------------------------------------------------------------------------
    def _pack_data(self):                                               # unfmt_block
    #--------------------------------------------------------------------------------
        # 4i| 1000 data |4i|4i| 1000 data |4i|4i| 1000 data |4i|...|4i| 1000 data |4i|
        dtype = self.header.dtype
        for a,b in slice_range(0, len(self._data), dtype.max):
            length = b - a
            size = length * dtype.size
            yield pack(ENDIAN + f'i{length}{dtype.unpack}i', size, *self._data[a:b], size)
        
    #--------------------------------------------------------------------------------
    def as_bytes(self):                                               # unfmt_block
    #--------------------------------------------------------------------------------
        return self.header.as_bytes() + b''.join(self._pack_data())

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
    def is_flushed(self, endkey):                                        # unfmt_file
    #--------------------------------------------------------------------------------
        if self.is_file():
            last_block = next(self.tail_blocks(), None)
            if last_block and last_block.key() == endkey:
                return True

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
        # groups = list((k, list(zip(*sorted(g)))) for k,g in groupby(B, lambda x:x[0]))
        groups = ((k, list(zip(*sorted(g)))) for k,g in groupby(B, lambda x:x[0]))
        keys, lims = zip(*groups)
        # Variables from the same keyword must be listed together as input args
        if len(set(keys)) != len(keys):
            raise SyntaxWarning(f'Wrong input: similar keywords must be listed together: {keys}')
        return keys, [l[-1] for l in lims], dictkeys

    #--------------------------------------------------------------------------------
    def blocks_to_file(self, filename, keys=None, invert=False, append=False): # unfmt_file
    #--------------------------------------------------------------------------------
        """
        Write binary blocks with keys matching the given keys to file.

        Args:
            filename (str,Path): The name or path of the file.
            keys (list,tuple): Keywords of the blocks to write, wildcard patterns 
                               are allowed. If empty, all keywords are selected. 
            invert (bool): Write all blocks except the given keywords.
            append (bool): Append to file instead of creating new file
        """
        mode = 'wb'
        if append:
            mode = 'ab'
        if keys:
            keylist = expand_pattern(keys, self.section_keys(), invert=invert)
        else:
            keylist = self.section_keys()
        if isinstance(filename, UNRST_file):
            filename = filename.path
        with open(filename, mode) as file:
            for block in self.blocks():
                if block.key() in keylist:
                    file.write(block.binarydata())


    #--------------------------------------------------------------------------------
    def blockdata(self, *keylim, limits=None, strip=True, 
                  tail=False, singleton=False, **kwargs):                # unfmt_file
    #--------------------------------------------------------------------------------
        """ 
        Return data in the order of the given keys, not the reading order.
        The keys-list may contain wildcards (*, ?, [seq], [!seq]) 
        A key may be followed by zero, one, or two index values. 
        Two indices are interpreted as a slice, zero indices are 
        interpreted as the whole array.

        Example: ('KEY1', 10, 20, 'KEY2', 'KEY3', 5)
        """
        #print(keylim)
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
    def read(self, *varnames, **kwargs):                                # unfmt_file
    #--------------------------------------------------------------------------------
        """
        Read data block by block using variable names defined in self.var_pos. 
        The number of returned values match the number of input variables. 
        Single values are unpacked. Use zip to collect values across blocks.
        """
        #print(varnames)
        if missing := [var for var in varnames if var not in self.var_pos]:
            raise SyntaxWarning(f'Missing variable definitions for {type(self).__name__}: {missing}')
        var_pos = list(self.var_pos[var] for var in varnames)
        #keylim = flatten_all(zip(repeat(v[0]), group_indices(v[1:])) for v in var_pos)
        keylim = flatten_all(zip(repeat(v[0]), index_limits([-1 if i is None else i for i in v[1:]])) for v in var_pos)
        nvar = [len(pos) for _,*pos in var_pos]
        for values in self.blockdata(*keylim, **kwargs):
            if any(n>1 for n in nvar):
                # Split values to match number of input variables
                values = iter(values)
                yield [take(n,values)[0] if n==1 else take(n,flatten(values)) for n in nvar]
            else:
                yield values


    #--------------------------------------------------------------------------------
    def last_value(self, var:str):                                        # unfmt_file
    #--------------------------------------------------------------------------------
        return next(self.read(var, tail=True), None) or 0

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
    def blocks(self, only_new=False, start=None, use_mmap=True, **kwargs):  # unfmt_file
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
            return self.blocks_from_mmap(startpos, only_new=only_new, **kwargs)
        return self.blocks_from_file(startpos, only_new=only_new)
        #     yield from self.blocks_from_mmap(startpos, **kwargs)
        # yield from self.blocks_from_file(startpos)

    #--------------------------------------------------------------------------------
    def blocks_from_mmap(self, startpos, only_new=False, write=False):    # unfmt_file
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
                    # pos = self.endpos = header.endpos
                    pos = header.endpos
                    if only_new:
                        self.endpos = pos
                    yield unfmt_block(header=header, data=data, file=self.path)
        except ValueError: # Catch 'cannot mmap an empty file'
            return #() #False

    #--------------------------------------------------------------------------------
    def blocks_from_file(self, startpos, only_new=False):                # unfmt_file
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
                # pos = self.endpos = header.endpos
                pos = header.endpos
                if only_new:
                    self.endpos = pos
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
        # If reading from tail does not work we need to fix block payload errors
        if not next(self.tail_blocks(), False):
            # Fix errors in-place
            return sum(b.fix_payload_errors() for b in self.blocks(write=True))
        return 0

    #--------------------------------------------------------------------------------
    def count_sections(self):                                            # unfmt_file
    #--------------------------------------------------------------------------------
        return sum(1 for block in self.blocks() if self.start in block)

    #--------------------------------------------------------------------------------
    def count_blocks(self):                                              # unfmt_file
    #--------------------------------------------------------------------------------
        return sum(1 for _ in self.blocks())

    #--------------------------------------------------------------------------------
    def section_keys(self, n=0):                                         # unfmt_file
    #--------------------------------------------------------------------------------
        """ 
        Return keywords from section n, n=0 is default
        """
        return [bl.key() for bl in nth(self.section_blocks(), n)]

    #--------------------------------------------------------------------------------
    def check_missing_keys(self, *keys, raise_error=True):               # unfmt_file
    #--------------------------------------------------------------------------------
        """
        Check for missing keys in the section keys of the unrst attribute.

        Args:
            *keys: Variable length argument list of keys to check.
            raise_error (bool): If True, raises a ValueError if any keys are missing. Default is True.

        Returns:
            list: A list of missing keys.

        Raises:
            ValueError: If any keys are missing and raise_error is True.
        """
        missing = set(keys) - set(self.section_keys())
        if missing and raise_error:
            raise ValueError(f'Missing keywords in {self}: {list(missing)}')
        return list(missing)

    #--------------------------------------------------------------------------------
    def find_keys(self, *keys, sec=0):                                   # unfmt_file
    #--------------------------------------------------------------------------------
        """
        Return matching keywords from section sec, sec=0 is default
        """
        #block_keys = [bl.key() for bl in nth(self.section_blocks(), sec)]
        return expand_pattern(keys, self.section_keys(sec))

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
    def section_blocks(self, tail=False):                      # unfmt_file
    #--------------------------------------------------------------------------------
        """
        Return blocks one section at a time
        """
        self.exists(raise_error=True)
        blocks_func = self.blocks
        if tail:
            blocks_func = self.tail_blocks
        step_gen = (i for i,b in enumerate(blocks_func()) if self.start in b)
        # Need to add the total number of blocks to include the last section
        step = pairwise(chain(step_gen, [self.count_blocks()]))
        blocks = blocks_func()
        a, b = next(step)
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
                # Get 'endpos' or 'startpos' for the start/end blocks
                _slice = slice(*(getattr(p,a) for p,a in zip(pair, attrs)))
                data = filemap[_slice]
                if rename and (names:=[rn for rn in rename if rn[0] in data]):
                    for old, new in names:
                        data = data.replace(old.ljust(8), new.ljust(8))
                yield (step[0], data)
                # yield (step[0], filemap[_slice])

    #--------------------------------------------------------------------------------
    def section_slices(self, start=(), end=()):                          # unfmt_file
    #--------------------------------------------------------------------------------
        """
        Get the file-position slice defined by the 'start' and 'end' block-keywords. 
        """
        # Example of start and end format: 
        # start=('SEQNUM', 'startpos'), end=('ENDSOL', 'endpos') 
        # Start by splitting args in keys=('SEQNUM', 'ENDSOL') and attrs=('startpos', 'endpos')
        keys, attrs = zip(start, end)
        step = -1
        _matches = {k:None for k in keys}
        for section in self.section_blocks():
            for block in section:
                if self.start in block:
                    step = block.data()[0]
                if any(key in block for key in keys):
                    _matches[block.key()] = block
            # Get 'endpos' or 'startpos' for the start/end blocks
            _slice = slice(*(getattr(p,a) for p,a in zip(_matches.values(), attrs)))
            yield (step, _slice)
            _matches = {k:None for k in keys}

    #--------------------------------------------------------------------------------
    def section_data2(self, start=(), end=(), rename=(), begin=0):       # unfmt_file
    #--------------------------------------------------------------------------------        
        with self.mmap() as filemap:
            for step, _slice in self.section_slices(start, end):
                if step < begin:
                    continue
                data = filemap[_slice]
                if rename and (names:=[rn for rn in rename if rn[0] in data]):
                    for old, new in names:
                        data = data.replace(old.ljust(8), new.ljust(8))
                yield (step, data)

    #--------------------------------------------------------------------------------
    def section_ends(self, **kwargs):                                    # unfmt_file
    #--------------------------------------------------------------------------------
        endwords = [self.start, self.end]
        ends = (bl for bl in self.blocks(**kwargs) if bl.key() in endwords)
        for first, last in batched(ends, 2):
            if first.key() == last.key():
                endwords.remove(first.key())
                raise ValueError(f"Incomplete section: '{endwords[0]}' keyword is missing")
            yield (first, last)
        #return batched(ends, 2)

    #--------------------------------------------------------------------------------
    def merge(self, *section_data, progress=lambda x:None, 
              cancel=lambda:None):                                       # unfmt_file
    #--------------------------------------------------------------------------------
        skipped = []
        with open(self.path, 'wb') as merge_file:
            n = 0
            for steps_data in zip(*section_data):
                cancel()
                steps, data = zip(*steps_data)
                while len(set(steps)) > 1:
                    # Sync sections if steps don't match
                    if steps[0] < steps[1]:
                        skipped.append(steps[0])
                        steps, data = zip(next(section_data[0]), (steps[1], data[1]))
                    else:
                        skipped.append(steps[1])
                        steps, data = zip((steps[0], data[0]), next(section_data[1]))
                    #raise SystemError(f'ERROR Merged steps are different: {steps}')
                for d in data:
                    merge_file.write(d)
                n += 1
                progress(n)
        if skipped:
            print(f'WARNING! Some steps were skipped: {skipped}')
        return self.path

    #--------------------------------------------------------------------------------
    def remove_sections(self, nsec):                                     # unfmt_file
    #--------------------------------------------------------------------------------
        """
        For positive values, remove leading sections
        For negtive values, remove tailing sections
        """
        if nsec > 0:
            # Remove nsec leading sections
            end = next(islice(self.section_blocks(), nsec, None))[-1].endpos
            self.resize(start=0, end=end)
        else:
            # Remove nsec tailing sections
            start = next(islice(self.section_blocks(tail=True), abs(nsec), None))[0].startpos
            self.resize(start=start, end=self.size)


    #--------------------------------------------------------------------------------
    def assert_no_duplicates(self, raise_error=True):                    # unfmt_file
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
            days, n = next(((t,s) for t,s in file.read('time', 'step') if s >= step), (-1,-1))
            if n != step:
                raise SystemError(f'ERROR Step {step} is missing in restart file {file}')
            start = next(file.dates())
            return Restart(start=start, days=days, step=n)
        # Get start from DATA-file
        start = self.start()
        return Restart(start=self.start(), step=step)


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
            raise SystemError(
                f'ERROR Zero or negative timestep in {self} (check if TSTEP or RESTART oversteps a DATES keyword)')
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
            wells += unrst.wells(stop=int(step))
        return list(set(wells))

    #--------------------------------------------------------------------------------
    def welspecs(self):                                                   # DATA_file
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
    start = 'FILEHEAD'
    end = 'ENDGRID'
    #--------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                  # EGRID_file
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.EGRID', **kwargs)
        self.var_pos = {'nx': ('GRIDHEAD', 1),
                        'ny': ('GRIDHEAD', 2),
                        'nz': ('GRIDHEAD', 3),
                        'unit': ('MAPUNITS', 0)}
        self._nijk = None
        self._coord_zcorn = None

    #--------------------------------------------------------------------------------
    def length(self):                                                    # EGRID_file
    #--------------------------------------------------------------------------------
        convert = {'METRES':1.0, 'FEET':0.3048, 'CM':0.01}
        unit = next(self.blockdata('MAPUNITS'), None)
        if unit:
            return convert[unit]
        raise SystemError(f'ERROR Missing MAPUNITS in {self}')

    #--------------------------------------------------------------------------------
    def axes(self):                                                      # EGRID_file
    #--------------------------------------------------------------------------------
        ax = next(self.blockdata('MAPAXES'), None)
        origin = (ax[2], ax[3])
        unit_x = (ax[4]-ax[2], ax[5]-ax[3])
        unit_y = (ax[0]-ax[2], ax[1]-ax[3])
        norm_x = 1 / hypot(*unit_x)
        norm_y = 1 / hypot(*unit_y)
        return origin, (unit_x[0]*norm_x, unit_x[1]*norm_x), (unit_y[0]*norm_y, unit_y[1]*norm_y)

    #--------------------------------------------------------------------------------
    def nijk(self):                                           # EGRID_file
    #--------------------------------------------------------------------------------
        self._nijk = self._nijk or next(self.read('nx', 'ny', 'nz'))
        return self._nijk        

    #--------------------------------------------------------------------------------
    def coord_zcorn(self):                                           # EGRID_file
    #--------------------------------------------------------------------------------
        self._coord_zcorn = self._coord_zcorn or list(map(nparray, next(self.blockdata('COORD', 'ZCORN'))))
        return self._coord_zcorn        

    #--------------------------------------------------------------------------------
    def _indices(self, ijk):                                         # EGRID_file
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def cell_corners(self, ijk_iter):                                     # EGRID_file
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def grid(self, i=None, j=None, k=None, scale=(1,1,1)):                     # EGRID_file
    #--------------------------------------------------------------------------------
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


#====================================================================================
class INIT_file(unfmt_file):                                              # INIT_file
#====================================================================================
    start = 'INTEHEAD'
    #--------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                   # INIT_file
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def dim(self):                                                       # INIT_file
    #--------------------------------------------------------------------------------
        self._dim = self._dim or next(self.read('nx', 'ny', 'nz'))
        return self._dim
    
    #--------------------------------------------------------------------------------
    def reshape_dim(self, *data):                                         # INIT_file
    #--------------------------------------------------------------------------------
        return [nparray(d).reshape(self.dim(), order='F') for d in data]

    #--------------------------------------------------------------------------------
    def simulator(self):                                                  # INIT_file
    #--------------------------------------------------------------------------------
        sim_codes = {100:'ecl', 300:'ecl', 500:'ecl', 700:'ix', 800:'FrontSim'}
        if sim:=next(self.read('simulator'), None):
            if sim < 0:
                return 'other simulator'
            return sim_codes[sim]

    #--------------------------------------------------------------------------------
    def start_date(self):                                                  # INIT_file
    #--------------------------------------------------------------------------------
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

#====================================================================================
class UNRST_file(unfmt_file):                                            # UNRST_file
#====================================================================================
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
                                                
    #--------------------------------------------------------------------------------
    def __init__(self, file, suffix='.UNRST', wait_func=None, end=None, role=None, 
                 **kwargs):                                              # UNRST_file
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix=suffix, role=role)
        self.end = end or self.end
        self.check = check_blocks(self, start=self.start, end=self.end, wait_func=wait_func, **kwargs)
        self._dim = None
        self._units = None
        #self._dates = None

    #--------------------------------------------------------------------------------
    def __len__(self):                                                   # UNRST_file
    #--------------------------------------------------------------------------------
        return len(list(self.steps()))

    #--------------------------------------------------------------------------------
    def dim(self):                                                       # UNRST_file
    #--------------------------------------------------------------------------------
        self._dim = self._dim or next(self.read('nx', 'ny', 'nz'))
        return self._dim
    
    #--------------------------------------------------------------------------------
    def reshape_dim(self, *data): #, flip=True):                         # UNRST_file
    #--------------------------------------------------------------------------------
        return [nparray(d).reshape(self.dim(), order='F') for d in data]
        # if flip:
        #     return [npflip(a, axis=-1) for a in arr]
        # return arr

    #--------------------------------------------------------------------------------
    def _check_for_missing_keys(self, *in_keys, keys=None):              # UNRST_file
    #--------------------------------------------------------------------------------
        keys = keys or self.find_keys(*in_keys)
        if missing := [ik for ik in in_keys if not any(fnmatch(k, ik) for k in keys)]:
            raise ValueError(f'The following keywords are missing in {self}: {missing}')
        return keys

    #--------------------------------------------------------------------------------
    def _cellnr(self, coord, base=0):                                    # UNRST_file
    #--------------------------------------------------------------------------------
        """
        Return position in 1D array given 3D coordinate of base=0 (default) or base=1
        """
        dim = self.dim()
        # Apply negative index from the end
        coord = [c if c>=0 else dim[i]+c+base for i,c in enumerate(coord)]
        return coord[0]-base + dim[0]*(coord[1]-base) + dim[0]*dim[1]*(coord[2]-base)

    #--------------------------------------------------------------------------------
    #def celldata(self, coord, *keywords, base=0, time_res='day'):        # UNRST_file
    def celldata(self, coord, *keywords, base=0):                         # UNRST_file
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def celldata_as_dataframe(self, *args, **kwargs):        # UNRST_file
    #--------------------------------------------------------------------------------
        """
        Write given keywords for the given cell-coordinate to a tab-separated file
        """
        # data = self.celldata(coord, *keywords, base=base, time_res=time_res)
        data = self.celldata(*args, **kwargs)
        return DataFrame(data._asdict())

    #--------------------------------------------------------------------------------
    def cellarray(self, *in_keys, start=None, stop=None, step=1, warn_missing=True):   # UNRST_file                  
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def wells(self, stop=None):                                          # UNRST_file
    #--------------------------------------------------------------------------------
        wells = flatten_all(islice(self.read('wells'), 0, stop))
        unique_wells = set(w for well in wells if (w:=well.strip()))
        return tuple(unique_wells)

    #--------------------------------------------------------------------------------
    def open_wells(self):                                                # UNRST_file
    #--------------------------------------------------------------------------------
        for ihead, icon in self.blockdata('INTEHEAD', 'ICON'):
            niconz, ncwmax, nwells  = ihead[32], ihead[17], ihead[16]
            icon = nparray(icon).reshape((niconz, ncwmax, nwells), order='F')
            yield sum(npsum(icon[5,:,:], axis=0) > 0)


    #--------------------------------------------------------------------------------
    def steps(self):                                                     # UNRST_file
    #--------------------------------------------------------------------------------
        return flatten_all(self.read('step'))

    #--------------------------------------------------------------------------------
    def end_step(self):                                                  # UNRST_file
    #--------------------------------------------------------------------------------
        return self.last_value('step')

    #--------------------------------------------------------------------------------
    def end_time(self):                                                  # UNRST_file
    #--------------------------------------------------------------------------------
        return self.last_value('time')

    #--------------------------------------------------------------------------------
    def end_date(self):                                                  # UNRST_file
    #--------------------------------------------------------------------------------
        return next(self.dates(tail=True), None)

    #--------------------------------------------------------------------------------
    def dates(self, resolution='day', **kwargs):                         # UNRST_file
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def units(self):                                                     # UNRST_file
    #--------------------------------------------------------------------------------
        if self._units is None:
            ihead = next(self.blockdata('INTEHEAD'), None)
            if ihead:
                self._units = {1:'metric', 2:'field', 3:'lab', 4:'pvt-m'}[ihead[2]]
        return self._units

    #--------------------------------------------------------------------------------
    def days(self, **kwargs):                                            # UNRST_file
    #--------------------------------------------------------------------------------
        # Read units only once
        #self._units = self._units or self.units()
        convert = 1
        # if self._units == 'lab':
        if self.units() == 'lab':
            # DOUBHEAD[0] is given in hours in lab units
            convert = 1/24
        return (next(flatten(dh))*convert for dh in self.blockdata('DOUBHEAD', singleton=True, **kwargs))
        #return (dh[0]*convert for dh in self.blockdata('DOUBHEAD', singleton=True, **kwargs))

    # #--------------------------------------------------------------------------------
    # def days2(self, start:datetime, **kwargs):                           # UNRST_file
    # #--------------------------------------------------------------------------------
    #     for date in self.dates(**kwargs):
    #         delta = date-start
    #         yield delta.total_seconds()/86400 + delta.microseconds/86400e6
    #     #return ((date-start).total_seconds()/86400 for date in self.dates(**kwargs))

    #--------------------------------------------------------------------------------
    def section(self, days=None, date=None):                      # UNRST_file
    #--------------------------------------------------------------------------------
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

    # #--------------------------------------------------------------------------------
    # def find_section(self, year, month, day):                            # UNRST_file
    # #--------------------------------------------------------------------------------
    #     stop = 
    #     return next(i for i,date in enumerate(self.dates()) if date >= stop)

    #--------------------------------------------------------------------------------
    def end_key(self):                                                   # UNRST_file
    #--------------------------------------------------------------------------------
        block = next(self.tail_blocks(), None)
        if block:
            return block.key()

    #--------------------------------------------------------------------------------
    def from_Xfile(self, xfile, log=False, delete=False):                 # UNRST_file
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def as_Xfiles(self, log=False, stop=None):                           # UNRST_file
    #--------------------------------------------------------------------------------
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
            
    #--------------------------------------------------------------------------------
    def block_inflow(self, flow):                           # UNRST_file
    #--------------------------------------------------------------------------------
        flow_in = zeros_like(flow)
        for axis in range(3):
            # Shift positive values one step in positive direction along given axis 
            pos_roll = roll(flow[..., axis], shift=1, axis=axis)
            # Shift negative values one step in negative direction along given axis 
            neg_roll = roll(flow[..., axis], shift=-1, axis=axis)
            pos_mask = pos_roll > 0
            neg_mask = neg_roll < 0
            flow_in[..., axis][pos_mask] += pos_roll[pos_mask]
            flow_in[..., axis][neg_mask] += npabs(neg_roll[neg_mask])
        return flow_in


    #--------------------------------------------------------------------------------
    def block_flows(self, phase, resrate=True):                           # UNRST_file
    #--------------------------------------------------------------------------------
        Flow = namedtuple('Flow', 'days dt Qin Qout vol dVol')
        phase = phase.upper()
        labunits = self.units() == 'lab'
        if resrate:
            rates = self.resrate(phase)
        else:
            rates = self.resrate_from_surfrate(phase)
        sat_pvol = self.blockdata('S'+phase, 'RPORV')
        # Initial values
        sat, porvol = self.reshape_dim(*next(sat_pvol))
        old_vol = sat * porvol
        old_days, *_ = next(rates)
        for data, (days, rate, pvt) in zip(sat_pvol, rates):
            sat, porvol = self.reshape_dim(*data)
            vol = sat * porvol
            dt = days - old_days
            if labunits:
                # Convert from days to hours
                dt *= 24
            dvol = vol - old_vol
            rate_in = self.block_inflow(rate)
            yield Flow(days, dt, rate_in, rate, vol, dvol), pvt
            old_days, old_vol = days, vol


    #--------------------------------------------------------------------------------
    def resrate(self, phase):                                            # UNRST_file
    #--------------------------------------------------------------------------------
        phase = phase.upper()
        for dhead, *flow in self.blockdata('DOUBHEAD', *[f'FLR{phase}{ijk}+' for ijk in 'IJK']):
            time = dhead[0]
            yield time, stack(self.reshape_dim(*flow), axis=-1)

    #--------------------------------------------------------------------------------
    def resrate_from_surfrate(self, phase):                              # UNRST_file
    #--------------------------------------------------------------------------------
        phase = phase.lower()
        oil = ('FLOOILI+', 'FLOOILJ+', 'FLOOILK+')
        wat = ('FLOWATI+', 'FLOWATJ+', 'FLOWATK+')
        gas = ('FLOGASI+', 'FLOGASJ+', 'FLOGASK+')
        oil_or_gas = phase in ('oil', 'gas')
        if oil_or_gas:
            flow_keys = (*oil, *gas) 
            pvt_keys = ('RS', 'RV', 'BO', 'BG')
        else: # water
            flow_keys = wat
            pvt_keys = ('BW',)
        PVT = namedtuple('PVT', *pvt_keys)
        for dhead, *file_data in self.blockdata('DOUBHEAD', *flow_keys, *pvt_keys):
            if oil_or_gas:
                # Unpack data from file into numpy arrays
                OI, OJ, OK,  GI, GJ, GK,  RS, RV, BO, BG = [nparray(d) for d in file_data]
                # Create 3D arrays for oil- and gas-flows
                OIL = stack((OI, OJ, OK), axis=-1)
                GAS = stack((GI, GJ, GK), axis=-1)
                # Add new axis to 1D arrays for broadcast
                pvt = PVT(*[arr[:, newaxis] for arr in (RS, RV, BO, BG)])
                denom = 1 - pvt.RS * pvt.RV
                if npany(denom == 0):
                    raise ValueError("Denominator is zero, which will cause a division by zero error.")
                if phase == 'oil':
                    resrate = pvt.BO * (         OIL - pvt.RV*GAS ) / denom
                else: # gas
                    resrate = pvt.BG * ( -pvt.RS*OIL +        GAS ) / denom   # ( 1 - RS*RV )
            else: # water
                # Unpack data from file into numpy arrays
                WI, WJ, WK,  BW = [nparray(d) for d in file_data]
                # Create 3D array for water-flows
                WAT = stack((WI, WJ, WK), axis=-1)
                # Add new axis to 1D array for broadcast
                pvt = PVT(BW[:, newaxis])
                resrate = pvt.BW * WAT

            # From IORSim code
            # rate.oil   = Bo * (     STC_rate.oil - Rv*STC_rate.gas )/( 1.0 - Rs*Rv );
            # rate.gas   = Bg * ( -Rs*STC_rate.oil +    STC_rate.gas )/( 1.0 - Rs*Rv );
            # rate.water = Bw *       STC_rate.water;
            
            time = dhead[0]
            rate = resrate.reshape((*self.dim(), 3), order='F')
            yield (time, rate, pvt)


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


#====================================================================================
class RFT_file(unfmt_file):                                                # RFT_file
#====================================================================================
    start = 'TIME'
    end = 'CONNXT'
    var_pos =  {'time'     : ('TIME', 0),
                'wellname' : ('WELLETC', 1),
                'welltype' : ('WELLETC', 6),
                'waterrate': ('CONWTUB', None),
                'I'        : ('CONIPOS', None),
                'J'        : ('CONJPOS', None),
                'K'        : ('CONKPOS', None)}

    #--------------------------------------------------------------------------------
    def __init__(self, file, wait_func=None, **kwargs):                    # RFT_file
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.RFT')
        self.check = check_blocks(self, start=self.start, end=self.end, wait_func=wait_func, **kwargs)
        self.current_section = 0

    #--------------------------------------------------------------------------------
    def not_in_sync(self, time, prec=0.1):                                 # RFT_file
    #--------------------------------------------------------------------------------
        data = self.check.data()
        if data and any(abs(nparray(data)-time) > prec):
            return True
        return False
        
    #--------------------------------------------------------------------------------
    def end_time(self):                                                    # RFT_file
    #--------------------------------------------------------------------------------
        # Return data from last check if it exists
        if data := self.check.data():
            return data[-1]
        # Return time-value from tail of file
        return next(self.read('time', tail=True), None) or 0
        #return time
        #return (data := self.check.data()) and data[-1] or 0

    #--------------------------------------------------------------------------------
    def time_slice(self):                                                  # RFT_file
    #--------------------------------------------------------------------------------
        """
        Yield time and slice for equal time sections
        """
        endpos = self.endpos
        ends = self.section_ends(only_new=True)
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
                    self.endpos = a.header.startpos
                    yield (time, (first.header.startpos, self.endpos))
                    first = a
                    time = t
                last = b
        except ValueError:
            self.endpos = endpos
            yield (None, None)
            return

    #--------------------------------------------------------------------------------
    def sections_matching_time(self, days, acc=1e-5):                      # RFT_file
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def wellstart(self, *wellnames):                                       # RFT_file
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def wellbbox(self, *wellnames, zerobase=True):                          # RFT_file
    #--------------------------------------------------------------------------------
        wpos = self.wellpos(*wellnames, zerobase=zerobase)
        bbox = {well:None for well in wellnames}       
        for well, wp in zip(wellnames, wpos):
            bbox[well] = [(p[0], p[-1]) for p in map(sorted, zip(*wp))]
        return tuple(bbox.values())

    #--------------------------------------------------------------------------------
    def wellpos(self, *wellnames, zerobase=True):                          # RFT_file
    #--------------------------------------------------------------------------------
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

    #--------------------------------------------------------------------------------
    def grid2wellname(self, dim, *wellnames):                              # RFT_file
    #--------------------------------------------------------------------------------
        poswell = {pos:[] for pos in product(*(range(d) for d in dim))}
        for well, pos in zip(wellnames, self.wellpos(*wellnames)):
            for p in pos:
                poswell[p].append(well)
        return poswell
        
    #--------------------------------------------------------------------------------
    def active_wells(self):                                                # RFT_file
    #--------------------------------------------------------------------------------
        wells = []
        current_time = next(self.read('time'))
        for time, well in self.read('time', 'wellname'):
            if time > current_time:
                yield current_time, wells
                wells = []
            wells.append(well)
            current_time = time
            
    #--------------------------------------------------------------------------------
    def rate_from_tubs(self, tub, wellrate, connxt):                       # RFT_file
    #--------------------------------------------------------------------------------
        connxt = atleast_1d(connxt)
        tub = atleast_1d(tub)
        injector = wellrate < 0
        idx = list(argsort(connxt))
        miss = missing_elements(connxt) or [len(connxt)]
        is_neigh = ones(len(idx) + len(miss), dtype=np_bool)
        for m in miss:
            idx.insert(m, (m if m<len(connxt) else 0) if injector else m-1)
            is_neigh[m] = 0
        #is_neigh[list(miss)] = 0
        if injector:
            # Injector
            diff = zeros(len(tub))
            for i in range(len(tub)):
                if is_neigh[i+1]:
                    diff[idx[i+1]] = tub[i] - tub[idx[i+1]]
            diff[idx[0]] = wellrate - tub[idx[0]]
        else:
            # Producer
            diff = tub - is_neigh[1:]*tub[idx[1:]]
        return diff
        #return squeeze(diff)


    #--------------------------------------------------------------------------------
    def rate_from_surf(self, oil, wat, gas, pos, pvt):                       # RFT_file
    #--------------------------------------------------------------------------------
        phase = phase.lower()
        pvt = pvt._asdict()
        RS, RV, BO, BG, BW = [pvt[key][*pos, :] for key in ('RS', 'RV', 'BO', 'BG', 'BW')]
        denom = (1 - pvt.RS * pvt.RV)
        resoil = BO * (     oil - RV * gas) / denom
        resgas = BG * ( -RS*oil +      gas) / denom 
        reswat = BW * wat
        return resoil, reswat, resgas


    #--------------------------------------------------------------------------------
    def well_flows(self, zerobase=True, resrate=True):                       # RFT_file
    #--------------------------------------------------------------------------------
        Resrate = namedtuple('Resrate', 'days well pos oilrate watrate gasrate')
        keys = ('TIME', 'CONIPOS', 'CONJPOS', 'CONKPOS', 'WELLPLT', 'WELLETC', 'CONNXT')
        if resrate:
            rate_keys = ('CONOTUBL', 'CONWTUBL', 'CONGTUBL')
        else:
            rate_keys = ('CONORAT', 'CONWRAT', 'CONGRAT')
        rates = []
        current_time = next(self.read('time'))
        for time, i, j, k, wellplt, welletc, connxt, *conrates in self.blockdata(*keys, *rate_keys, singleton=True):
            if time[0] > current_time:
                yield rates
                current_time = time[0]
                rates = []
            if resrate:
                wellplt = nparray(wellplt)
                phase_rates = wellplt[5]*(wellplt[:3]/sum(wellplt[:3]))
                resrate = (self.rate_from_tubs(tubl, phase_rates[i], connxt) for i, tubl in enumerate(conrates))
            else:
                resrate = conrates
            pos = [i, j, k]
            if zerobase:
                pos = [[x-1 for x in p] for p in pos]
            rates.append(Resrate(time[0], welletc[1], pos, *resrate))

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
    def params(self, *keys):                                            # UNSMRY_file
    #--------------------------------------------------------------------------------
        keypos = {key:i for i, key in enumerate(next(self.spec.blockdata('KEYWORDS')))}
        ind = [keypos[key] for key in keys]
        for param in self.blockdata('PARAMS'):
            yield [param[i] for i in ind]

    #--------------------------------------------------------------------------------
    def steptype(self):                                          # UNSMRY_file
    #--------------------------------------------------------------------------------
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
            reader = self.read('days', 'welldata', only_new=only_new, singleton=True, **kwargs)
            try:
                #days, data = zip(*self.read('days', 'welldata', only_new=only_new, **kwargs))
                #days, data = zip(*self.read2('days', 'welldata', only_new=only_new, **kwargs))
                days, data = zip(*islice(reader, start, stop, step))
                days = tuple(flatten(days))
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
            Values = namedtuple('Values', 'key well data')
            values = (Values(k, w, d) for k,w,d in kwd)
            return namedtuple('Welldata', 'days dates values')(days, tuple(dates), tuple(values))
        return ()

    #--------------------------------------------------------------------------------
    def plot(self, keys=(), wells=(), ncols=1, date=True, fignr=1,
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
        # get truncated while mmap'ed which will cause a bus-error
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
                if not isinstance(self.wells, (list, tuple)):
                    self.wells = (self.wells,)
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
            if not self._ind:
                return ()
            items = itemgetter(*self._ind)(val)
            # Need this check because the return type of itemgetter is not consistent
            # For single indices it returns a value instead of a list 
            if len(self._ind) > 1:
                return items
            return (items,)
        return super().__getattr__(item)

    #--------------------------------------------------------------------------------
    def check_missing_keys(self):                                             # SMSPEC_file
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
        chunks = (txt for txt in self.reversed(size=10*1024) if 'TIME' in txt)
        if data:=next(chunks, None):
            days = findall(self._pattern['time'], data)
            return float(days[-1]) if days else 0
        return 0


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
            if block.header._key == self._keys[start]:
                if (data := block.data()):
                    self._keys[start_val].append(data[0])
                else:
                    return False
            if block.header._key == self._keys[end]:
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
    #include_regex = rb'^[ \t]*\bINCLUDE\b\s*"*([\w.-]+)"*'
    # Return 'filename' and 'key1=val1 key2=val2' as groups from the following format:
    # INCLUDE "filename" {key1=val1, key2=val2}
    include_regex = rb'\bINCLUDE\b\s*"*([^"]+)"*\s*\{([^}]+)\}'

    #--------------------------------------------------------------------------------
    def __init__(self, file, check=False, **kwargs):                       # AFI_file
    #--------------------------------------------------------------------------------
        super().__init__(file, suffix='.afi', role='Top level Intersect input-file',
                         ignore_suffix_case=True, **kwargs)
        self._data = None
        self.pattern = None
        if check:
            self.exists(raise_error=True)
            
    #--------------------------------------------------------------------------------
    def data(self):                                                        # AFI_file
    #--------------------------------------------------------------------------------
        self._data = self._data or self.binarydata()
        return self._data

    #--------------------------------------------------------------------------------
    def ixf_files(self):                                                   # AFI_file
    #--------------------------------------------------------------------------------
        #self.data = self.data or self.binarydata()
        # self.pattern = self.pattern or re_compile(self.include_regex, flags=MULTILINE)
        #matches_ = findall(self.include_regex, self.data, flags=MULTILINE)
        #files = (Path(m[0].decode()) for m in matches_)
        #return (self.with_name(file) for file in files if file.suffix.lower() == '.ixf')
        #return (self.with_name(file) for file in self.files(self.data) if file.suffix.lower() == '.ixf')
        return (file for file in self.include_files(self.data()) if file.suffix.lower() == '.ixf')

    #--------------------------------------------------------------------------------
    def matches(self, data:bytes=None):                                   # AFI_file
    #--------------------------------------------------------------------------------
        self.pattern = self.pattern or re_compile(self.include_regex, flags=MULTILINE)
        return self.pattern.finditer(data or self.data())

    #--------------------------------------------------------------------------------
    def include_files(self, data:bytes=None):                              # AFI_file
    #--------------------------------------------------------------------------------
        return (self.path.with_name(m[1].decode()) for m in self.matches(data or self.data()))
        #data =  #binarydata()
        #self.pattern = self.pattern or re_compile(self.include_regex, flags=MULTILINE)
        #return (self.path.with_name(m.group(1).decode()) for m in self.finditer(data or self.data()))
        # return (f[0] for f in self._included_file_data(self.data))

    #--------------------------------------------------------------------------------
    def included_file_data(self, data:bytes=None):                        # AFI_file
    #--------------------------------------------------------------------------------
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
    def __contains__(self, key):                                           # IXF_node
    #--------------------------------------------------------------------------------
        return key in self.content

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
        return split_in_lines(self.content)

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
    def get(self, *items):                                                   # IXF_node
    #--------------------------------------------------------------------------------
        #return self.as_dict().get(item)
        values = flatten(self.as_dict().get(item) or [None] for item in items)
        return [val.split('#')[0].strip().replace('"', '') for val in values if val]
        # return list(flatten(self.as_dict().get(item) or [None] for item in items))

    #--------------------------------------------------------------------------------
    def update(self, node=None): #, on_top=False):                         # IXF_node
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
    #def node(self, *nodes, convert=(), brace=(rb'{',rb'}')):               # IXF_file
    def node(self, *nodes, convert=(), table=False):               # IXF_file
    #--------------------------------------------------------------------------------
        if table:
            begin, end = b'\\[', b'\\]'
        else:
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


#====================================================================================
class IX_input:                                                            # IX_input
#====================================================================================
    STAT_FILE = '.ecl2ix' # Check if ECL-input has changed and new conversion is needed

    @classmethod
    #--------------------------------------------------------------------------------
    def datesteps(cls, start, stop, step=1):
    #--------------------------------------------------------------------------------
        """
        datesteps((1971, 7, 1), 10, 5) -> DATE "01-Jul-1971"
                                          DATE "06-Jul-1971"
        """
        fmt = '%d-%b-%Y %H:%M:%S'
        dates = (f'DATE "{date}"' for date in date_range(start, stop, step, fmt=fmt))
        print('\n'.join(dates))

    #--------------------------------------------------------------------------------
    def __init__(self, case, check=False, **kwargs):                       # IX_input
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

    #--------------------------------------------------------------------------------
    def ifind(self, astr:str):                                              # IX_input
    #--------------------------------------------------------------------------------
        enc_str = astr.encode()
        for file, data in self.afi.included_file_data():
            if enc_str in data:
                yield file

    #--------------------------------------------------------------------------------
    def find(self, *args):                                                 # IX_input
    #--------------------------------------------------------------------------------
        return next(self.ifind(*args), None)

    #--------------------------------------------------------------------------------
    def findall(self, *args):                                                 # IX_input
    #--------------------------------------------------------------------------------
        return tuple(self.ifind(*args))

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
        if include and (missing := [f for f in self.include_files() if not f.exists()]):
            raise SystemError(f'ERROR {list2text([f.name for f in missing])} '
                              f'included from {self} is missing in folder {missing[0].parent}')
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
    def _timestep_files(self):                                             # IX_input
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
        start = self.start()
        return [start + timedelta(days=days) for days in accumulate(self.timesteps())]
    
    #--------------------------------------------------------------------------------
    def wellnames(self, contains:str=''):                                  # IX_input
    #--------------------------------------------------------------------------------
        wells = self.wells()
        if contains:
            return sorted(well[0] for well in wells if contains in well[1])
        return sorted(well[0] for well in wells)
        #return tuple(set(node.name for node in self.nodes('Well')))
        #return tuple(set(node.name for node in self.nodes('WellDef')))

    #--------------------------------------------------------------------------------
    def wells(self):                                                       # IX_input
    #--------------------------------------------------------------------------------
        return (set((well.name, _type[0]) for well in self.nodes('Well') if (_type:=well.get('Type'))))

    #--------------------------------------------------------------------------------
    def wellpos(self, *wellnames):                                         # IX_input
    #--------------------------------------------------------------------------------
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
        # Check if this is a restart-run
        match = next((m for m in self.afi.matches() if b'restart' in m[2]), None)
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

    #--------------------------------------------------------------------------------
    def UNRST_settings(self):                                              # IX_input
    #--------------------------------------------------------------------------------
        nodename = 'Recurrent3DReport'
        nodes = list(self.nodes(nodename))
        if nodes:
            return nodes[-1]
        raise SystemError((f"ERROR Node {nodename} not found in {self}"))

    #--------------------------------------------------------------------------------
    def write_unified_UNRST(self):                                         # IX_input
    #--------------------------------------------------------------------------------
        unified = self.UNRST_settings().get('Unified')
        # Default is True if not set
        if not unified or unified[0] in ('TRUE', 'True', 'true'):
            return True
        return False


#====================================================================================
class Rates():                                                             # Rates
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, root):                                           # Rates
    #--------------------------------------------------------------------------------
        self.unrst = UNRST_file(root)
        self.rft = RFT_file(root)
        self.dt_accuracy = 1e-5

    #--------------------------------------------------------------------------------
    def volume_error(self, phase='wat', reservoir=False, tube=False):       # Rates
    #--------------------------------------------------------------------------------
        phase = phase.lower()
        Volumes = namedtuple('Volumes', 'time dt dQdt dV error')
        for flow, volume in zip(self.flows(phase, reservoir, tube), self.volumes(phase)):
            if abs(flow.time - volume.time) > self.dt_accuracy:
                raise ValueError(f'Difference in flow ({flow.time}) and volume ({volume.time}) timesteps!')
            dQdt = (flow.rate_in - flow.rate_out) * flow.dt
            dQdt = npsum(dQdt, axis=-1)
            error = zeros_like(dQdt)
            mask = npabs(volume.diff) > 1e-11
            #error = 100 * (1 - dQdt/volume.diff)
            error[mask] = dQdt[mask] - volume.diff[mask]
            yield Volumes(flow.time, flow.dt, dQdt, volume.diff, error)

    #--------------------------------------------------------------------------------
    def flows(self, phase, reservoir=True, tube=True):                        # Rates
    #--------------------------------------------------------------------------------
        #Flow = namedtuple('flow', 'time dt rate_in rate_out')
        Flow = namedtuple('flow', 'time dt rate_in rate_out rate_change')
        block_flow = self.block_flow(phase, reservoir)
        well_flow = self.well_flow(reservoir=reservoir, tube=tube)
        # Initial time
        time = next(block_flow).time
        # Loop over block flows and well injection flows
        for block, wells in zip(block_flow, well_flow):
            rates_in = {ph:self._block_inflow(rate) for ph,rate in block.rates.items()}
            rates_out = {ph:npabs(rate) for ph,rate in block.rates.items()}
            # rates_out, rates_in = self.block_inout_flow(block.rates)
            dQ = {ph:npsum(fin-fout, axis=-1) for (ph,fout),fin in zip(rates_out.items(), rates_in.values())}
            #print(wells)
            # Loop over active wells
            for well in wells:
                if abs(well.time - block.time) > self.dt_accuracy:
                    raise ValueError(f'Difference in block ({block.time}) and well ({well.time}) timesteps!')
                for ph, dq in dQ.items():
                    print(ph, dq[*well.pos], 'well:', well.rate[ph])
                    dq[*well.pos] -= well.rate[ph]
                # Loop over well phase rates
                # for wphase, wrate in well.rate.items():
                #     if wphase == phase:
                #         if wrate[0] < 0:
                #             pass
                #             # Injector
                #             #rates_in[wphase][*well.pos, 2] -= wrate # or np.abs()
                #             # rates_in[wphase][*well.pos, 2] = npabs(wrate)
                #         else:
                #             print(wphase, wrate, well.pos)
                #             # Producer
                #             #rates_out[wphase][*well.pos, 2] += wrate
            # ### TODO: Add flow from NNC
            # Take absolute value of rates, the sign indicates direction
            #rates_out = {ph:npabs(rate) for ph,rate in block.rates.items()}
            #rates = (rates_in[phase], rates_out[phase])
            rate_change = dQ[phase]
            if block.pvt:
                # Convert from surface rates to reservoir rates
                # rates = self._resrate_from_surfrate(block.pvt, phase, rates_in, rates_out)
                rate_change, = self._resrate_from_surfrate(block.pvt, phase, dQ)
            #yield Flow(block.time, block.time-time, *rates)
            yield Flow(block.time, block.time-time, rates_in[phase], rates_out[phase], rate_change)
            time = block.time


    #--------------------------------------------------------------------------------
    def volumes(self, phase):                                                 # Rates
    #--------------------------------------------------------------------------------
        Volume = namedtuple('volume', 'time phase pore dphase dpore')
        phase = phase.upper()
        blockdata = self.unrst.blockdata('DOUBHEAD', 0, 'S'+phase, 'RPORV')
        # Initial values
        time, *data = next(blockdata)
        old_sat, old_porvol = self.unrst.reshape_dim(*data)
        for time, *data in blockdata:
            sat, porvol = self.unrst.reshape_dim(*data)
            vol = sat * porvol
            old_vol = old_sat * old_porvol
            #            time phase  pore       dphase         dpore
            yield Volume(time, vol, porvol, vol - old_vol, porvol - old_porvol)
            old_sat, old_porvol = sat, porvol

    #--------------------------------------------------------------------------------
    def block_flow(self, phase, reservoir=True):                           # Rates
    #--------------------------------------------------------------------------------
        Blockflow = namedtuple('blockflow', 'time rates pvt')
        phases, flow_keys, pvt_keys = self.flow_pvt_keys(phase, reservoir)
        #print(flow_keys, pvt_keys)
        for time, *data in self.unrst.blockdata('DOUBHEAD', 0, *flow_keys, *pvt_keys):
            #print(time)
            # Split data into rates and PVT data
            data = batched(self.unrst.reshape_dim(*data), 3)
            # First part of data is rates
            rates = {ph:stack(next(data), axis=-1) for ph in phases}
            # Rest of data is PVT data (flatten grabs rest)
            # pvt = {k:d[..., newaxis] for k, d in zip(pvt_keys, flatten(data))}
            pvt = {k:d for k, d in zip(pvt_keys, flatten(data))}
            yield Blockflow(time, rates, pvt)

    # #--------------------------------------------------------------------------------
    # def block_inout_flow(self, rates:dict):
    # #--------------------------------------------------------------------------------
    #     rates_out, rates_in = {}, {}
    #     for ph,rate in rates.items():
    #         pos_mask, neg_mask = rate > 0, rate < 0
    #         # Positive rates are outflow
    #         ro = zeros_like(rate)
    #         ro[pos_mask] = rate[pos_mask]
    #         # Negative rates are inflow
    #         ri = zeros_like(rate)
    #         ri[neg_mask] = -rate[neg_mask]
    #         ro_roll = zeros_like(ro)
    #         ri_roll = zeros_like(ri)
    #         for axis in range(3):
    #             # Outflow is inflow for next block in positive direction
    #             rro = roll(ro[..., axis], 1, axis=axis)
    #             ind = 3 * [slice(None)]
    #             # Remove periodic boundary end elements
    #             ind[axis] = 0
    #             rro[*ind] = 0
    #             ro_roll[..., axis] = rro
    #             # Inflow is outflow for previous block in negative direction
    #             rri = roll(ri[..., axis], -1, axis=axis)
    #             ind[axis] = -1
    #             rri[*ind] = 0
    #             ri_roll[..., axis] = rri            
    #         rates_out[ph] = ro + ri_roll
    #         rates_in[ph] = ri + ro_roll
    #     return rates_out, rates_in

    #--------------------------------------------------------------------------------
    def flow_pvt_keys(self, phase, reservoir):                               # Rates
    #--------------------------------------------------------------------------------
        # Return flow- and PVT-keys for UNRST-file depending on phase and reservoir/surface-rate
        phases = [phase.lower()]
        RO = 'R'
        pvt_keys = ()
        if not reservoir:
            RO = 'O'
            # Read PVT data to convert surface- to reservoir-rates
            if phases[0] in ('oil', 'gas'):
                phases = ('oil', 'gas')
                pvt_keys = ('RS', 'RV', 'BO', 'BG')
            else:
                pvt_keys = ('BW',)
        PHS = [phs.upper() for phs in phases]
        flow_keys = [f'FL{RO}{a}{b}+' for a,b in product(PHS,'IJK')]
        self.unrst.check_missing_keys(*flow_keys, *pvt_keys)
        return phases, flow_keys, pvt_keys

    # #--------------------------------------------------------------------------------
    # def well_inflow(self, **kwargs):                                           # Rates
    # #--------------------------------------------------------------------------------
    #     Inflow = namedtuple('Inflow', 'time well phase pos rate')
    #     inflow = []
    #     for wells in self.well_flow(**kwargs):
    #         for well in wells:
    #             phase_inj = (ph for ph,rate in well.rate.items() if any(r<0 for r in rate))
    #             if phase := next(phase_inj, None):
    #                 inflow.append(Inflow(well.time, well.name, phase, well.pos, well.rate[phase]))
    #         yield inflow
    #         inflow = []

    #--------------------------------------------------------------------------------
    def well_flow(self, reservoir=True, tube=True, zerobase=True):       # Rates
    #--------------------------------------------------------------------------------
        Well = namedtuple('Well', 'time name pos rate')
        keys = ('TIME', 'CONIPOS', 'CONJPOS', 'CONKPOS', 'WELLPLT', 'WELLETC', 'CONNXT')
        L = 'L' if reservoir else ''
        TUBRAT = 'TUB' if tube else 'RAT'
        if reservoir and not tube:
            raise SystemError('ERROR Well reservoir rates without tubes (CONORATL) are not available')
        rate_keys = [f'CON{owg}{TUBRAT}{L}' for owg in 'OWG']
        #print(rate_keys)
        #rates = {'in':[], 'out':[]}
        rates = []
        current_time = next(self.rft.read('time'))
        for time, i, j, k, wellplt, welletc, connxt, *wellrat in self.rft.blockdata(*keys, *rate_keys, singleton=True):
            time = time[0]
            wellname = welletc[1]
            if time > current_time:
                yield rates
                current_time = time
                #rates = {'in':[], 'out':[]}
                rates = []
            if tube:
                wellrat = self._wellrate_from_tubs(wellrat, wellplt, connxt)
            wellrat = {ph:rat for ph,rat in zip(('oil', 'wat', 'gas'), wellrat)}
            # Fix zero-based position indices
            pos = [i, j, k]
            if zerobase:
                pos = [[x-1 for x in p] for p in pos]
            #well = Well(time, wellname, pos, wellrat)
            # if wellplt[5] < 0:
            #     rates['in'].append(well)
            # else:
            #     rates['out'].append(well)
            rates.append(Well(time, wellname, pos, wellrat))


    #--------------------------------------------------------------------------------
    def _block_inflow(self, flow):                                      # Rates
    #--------------------------------------------------------------------------------
        flow_in = zeros_like(flow)
        # Define masks ([0, :, :], [-1, :, :]) for edge values
        # def mask(axis, shift):
        #     edge = 0 if shift > 0 else -1
        #     mask_ = roll((edge, *repeat(slice(None), 2)), axis)
        #     print(axis, shift, mask_)
        #     return mask_
        
        mask = [(i, *repeat(slice(None), 2)) for i in (0,-1)]
        # Roll in the opposite direction for the z-axis
        #pos_shift = ( 1,  1,  1)
        #neg_shift = (-1, -1, -1)
        for axis in range(3):
            #flow_out = flow[..., axis]
            #if axis == 2:
            #    flow_out = -flow_out
            # Shift positive values one step in positive direction along given axis
            pos_roll = roll(flow[..., axis], shift=1, axis=axis)
            #pos_roll = roll(flow_out, shift=pos_shift[axis], axis=axis)
            # Remove periodic boundary start elements rolled from last element
            pos_roll[ *roll(mask[0], axis) ] = 0
            # pos_roll[ *mask(axis, pos_shift[axis]) ] = 0
            # Add only positive shifted values
            pos_mask = pos_roll > 0
            flow_in[..., axis][pos_mask] += pos_roll[pos_mask]

            # Shift negative values one step in negative direction along given axis
            neg_roll = roll(flow[..., axis], shift=-1, axis=axis)
            #neg_roll = roll(flow_out, shift=neg_shift[axis], axis=axis)
            # Remove periodic boundary end elements rolled from first element
            neg_roll[ *roll(mask[1], axis) ] = 0
            # neg_roll[ *mask(axis, neg_shift[axis]) ] = 0
            # Add only absolute value of negative shifted values
            neg_mask = neg_roll < 0
            flow_in[..., axis][neg_mask] += npabs(neg_roll[neg_mask])
        return flow_in


    #--------------------------------------------------------------------------------
    def _wellrate_from_tubs(self, tubflows, wellplt, connxt):                       # Rates
    #--------------------------------------------------------------------------------
        wellplt = nparray(wellplt)
        phase_rates = wellplt[5]*(wellplt[:3]/sum(wellplt[:3]))
        connxt = atleast_1d(connxt)
        rates = []
        for i, tub in enumerate(tubflows):
            wellrate = phase_rates[i]
            tub = atleast_1d(tub)
            injector = wellrate < 0
            idx = list(argsort(connxt))
            miss = missing_elements(connxt) or [len(connxt)]
            is_neigh = ones(len(idx) + len(miss), dtype=np_bool)
            for m in miss:
                idx.insert(m, (m if m<len(connxt) else 0) if injector else m-1)
                is_neigh[m] = 0
            #is_neigh[list(miss)] = 0
            if injector:
                # Injector
                diff = zeros(len(tub))
                for i in range(len(tub)):
                    if is_neigh[i+1]:
                        diff[idx[i+1]] = tub[i] - tub[idx[i+1]]
                diff[idx[0]] = wellrate - tub[idx[0]]
            else:
                # Producer
                diff = tub - is_neigh[1:]*tub[idx[1:]]
            rates.append(diff)
        return rates


    #--------------------------------------------------------------------------------
    def _resrate_from_surfrate(self, pvt, phase, *rates):             # Rates
    #--------------------------------------------------------------------------------
        # From IORSim code
        # rate.oil   = Bo * (     STC_rate.oil - Rv*STC_rate.gas )/( 1.0 - Rs*Rv );
        # rate.gas   = Bg * ( -Rs*STC_rate.oil +    STC_rate.gas )/( 1.0 - Rs*Rv );
        # rate.water = Bw *       STC_rate.water;

        #print('Converting surface rates to reservoir rates')
        resrates = []
        #print(phase, [(k,v[10,10,3]) for k,v in pvt.items()])
        for rate in rates:
            #print([(k,v[10,10,3]) for k,v in rate.items()])
            if phase in ('oil', 'gas'):
                denom = 1 - pvt['RS'] * pvt['RV']
                if npany(denom == 0):
                    raise ValueError("Denominator is zero, which will cause a division by zero error.")
                if phase == 'oil':
                    resrate = pvt['BO'] * (              rate['oil'] - pvt['RV'] * rate['gas']) / denom
                else:
                    resrate = pvt['BG'] * ( -pvt['RS'] * rate['oil'] +             rate['gas']) / denom
            else:
                    resrate = pvt['BW'] * rate['wat']
            resrates.append(resrate)
        return resrates



