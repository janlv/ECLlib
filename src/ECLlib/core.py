"""Shared core components for ECLlib modules."""
import inspect
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from mmap import mmap, ACCESS_READ, ACCESS_WRITE
from pathlib import Path
from shutil import copy
from ECLlib.constants import DEBUG
from ECLlib.utils import tail_file, head_file, last_line

#from .Files import DATA_file, ECL2IX_LOG, File, Restart

# __all__ = [
#     "File",
#     "DATA_file",
#     "Restart",
#     "ECL2IX_LOG",
# ]


#==================================================================================================
class RefreshIterator:
#==================================================================================================
    """
    RefreshIterator is an iterator wrapper that allows re-reading from a source iterator
    by recreating it using a provided factory function. The factory must accept an 'only_new'
    parameter, which is set to True to ensure only new items are produced.

    Args:
        iterable_factory (callable): A function that returns a new iterator each time it is called.
            Must accept an 'only_new' keyword argument.
        *args: Positional arguments to pass to the factory.
        **kwargs: Keyword arguments to pass to the factory.

    Raises:
        ValueError: If the factory does not support the 'only_new' parameter.

    Usage:
        - On exhaustion of the underlying iterator, RereadIterator will refresh the iterator
        by calling the factory again with the same arguments.
        - If the refreshed iterator is also exhausted, StopIteration is raised.

    Methods:
        __iter__(): Returns self as an iterator.
        __next__(): Returns the next item from the underlying iterator, refreshing if needed.
    """

    #----------------------------------------------------------------------------------------------
    def __init__(self, iterable_factory, *args, **kwargs):
    #----------------------------------------------------------------------------------------------
        """
        iterable_factory: callable returning a NEW iterator every time it is invoked.
        only_new: if True and supported by the factory, pass only_new=True to the factory.
        """
        self._factory = iterable_factory
        params = inspect.signature(iterable_factory).parameters
        if not 'only_new' in params:
            raise ValueError(f"Function {iterable_factory.__name__} does not support 'only_new' parameter.")
        kwargs['only_new'] = True
        self._iter = self._factory(*args, **kwargs)
        self._args = args
        self._kwargs = dict(kwargs)

    #----------------------------------------------------------------------------------------------
    def __iter__(self):
    #----------------------------------------------------------------------------------------------
        return self

    #----------------------------------------------------------------------------------------------
    def _refresh(self):
    #----------------------------------------------------------------------------------------------
        """Create a fresh underlying iterator from the factory."""
        self._iter = self._factory(*self._args, **self._kwargs)

    #----------------------------------------------------------------------------------------------
    def __next__(self):
    #----------------------------------------------------------------------------------------------
        """
        Get the next element.
        If underlying iterator is exhausted, refresh once and try again.
        If still exhausted, raise StopIteration.
        """
        try:
            return next(self._iter)
        except StopIteration:
            self._refresh()
            return next(self._iter)  # may raise StopIteration again (desired)


#==================================================================================================
@dataclass
class Dtyp(frozen=True, slots=True):
#==================================================================================================
    name     : str = ''     # ECL type name
    unpack   : str = ''     # Char used by struct.unpack/pack to read/write binary data
    size     : int = 0      # Bytesize 
    max      : int = 0      # Maximum number of data records in one block
    nptype   : type = None  # Type used in numpy arrays 
    max_bytes: int = field(init=False) # max * size

    #----------------------------------------------------------------------------------------------
    def __post_init__(self):
    #----------------------------------------------------------------------------------------------
        self.max_bytes = self.max * self.size


#                        name  unpack size max  nptype
# DTYPE = {b'INTE' : Dtyp('INTE', 'i',   4, 1000, int32),
#          b'REAL' : Dtyp('REAL', 'f',   4, 1000, float32),
#          b'DOUB' : Dtyp('DOUB', 'd',   8, 1000, float64),
#          b'LOGI' : Dtyp('LOGI', 'i',   4, 1000, np_bool),
#          b'CHAR' : Dtyp('CHAR', 's',   8, 105 , 'S8'),
#          b'C008' : Dtyp('C008', 's',   8, 105 , 'S8'),
#          b'C009' : Dtyp('C009', 's',   9, 105 , 'S9'),
#          b'MESS' : Dtyp('MESS', 's',   1, 1   , 'S1')}

DTYPE = {b'INTE' : Dtyp('INTE', 'i',   4, 1000, 'i4'),
         b'REAL' : Dtyp('REAL', 'f',   4, 1000, 'f4'),
         b'DOUB' : Dtyp('DOUB', 'd',   8, 1000, 'f8'),
         b'LOGI' : Dtyp('LOGI', 'i',   4, 1000, 'b1'),
         b'CHAR' : Dtyp('CHAR', 's',   8, 105 , 'S8'),
         b'C008' : Dtyp('C008', 's',   8, 105 , 'S8'),
         b'C009' : Dtyp('C009', 's',   9, 105 , 'S9'),
         b'MESS' : Dtyp('MESS', 's',   1, 1   , 'S1')}

DTYPE_LIST = [v.name for v in DTYPE.values()]
        

#==================================================================================================
@dataclass
class Restart(frozen=True, slots=True):                                     # Restart
#==================================================================================================
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



#==================================================================================================
class File:                                                                    # File
#==================================================================================================
    """
    A class representing a file, allowing various operations such as reading, writing,
    memory mapping, and manipulation of file metadata. This class also supports file
    backup, text replacement, and more.

    Attributes:
        path (Path): The resolved path of the file.
        role (str): Optional role identifier for the file.
        debug (bool): Debug flag, prints debug messages when True.
    """

    #----------------------------------------------------------------------------------------------
    def __init__(self, filename, suffix=None, role=None, ignore_suffix_case=False, exists=False):          # File
    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
    def __repr__(self):                                                        # File
    #----------------------------------------------------------------------------------------------
        """
        Returns a string representation of the File object.

        Returns:
            str: String representation of the File object including its path and role.
        """
        return f"<{self.__class__.__name__}, file={self.path}, role={self.role or None}>"

    #----------------------------------------------------------------------------------------------
    def __str__(self):                                                         # File
    #----------------------------------------------------------------------------------------------
        """
        Returns a human-readable string representation of the file.

        Returns:
            str: The role and name of the file.
        """
        return f'{self.role}{self.name}'

    #----------------------------------------------------------------------------------------------
    def __del__(self):                                                         # File
    #----------------------------------------------------------------------------------------------
        """
        Destructor for the File object, optionally printing debug information if enabled.
        """
        if self.__class__.__name__ == File.__name__ and self.debug:
            print(f'Deleting {repr(self)}')

    #----------------------------------------------------------------------------------------------
    def __getattr__(self, item):                                               # File
    #----------------------------------------------------------------------------------------------
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
    #----------------------------------------------------------------------------------------------
    def mmap(self, write=False):                                               # File
    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
    def resize(self, start=0, end=0):                                   # File
    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
    def binarydata(self, pos=None, raise_error=False):                                   # File
    #----------------------------------------------------------------------------------------------
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
 
    #----------------------------------------------------------------------------------------------
    def as_text(self, **kwargs):                                               # File
    #----------------------------------------------------------------------------------------------
        """
        Reads the file and returns its content as text.

        Returns:
            str: The textual content of the file.
        """
        return self.binarydata(**kwargs).decode()

    #----------------------------------------------------------------------------------------------
    def delete(self, raise_error=False, echo=False):                           # File
    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
    def rename(self, newname, raise_error=False, echo=False):                  # File
    #----------------------------------------------------------------------------------------------
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


    #----------------------------------------------------------------------------------------------
    def is_file(self):                                                         # File
    #----------------------------------------------------------------------------------------------
        """
        Checks if the path points to a file.

        Returns:
            bool: True if the path is a file, False otherwise.
        """
        if not self.path:
            return False
        return self.path.is_file()

    #----------------------------------------------------------------------------------------------
    def with_name(self, file):                                                 # File
    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
    def with_tag(self, head:str='', tail:str=''):                              # File
    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
    def with_suffix(self, suffix, ignore_case=False, exists=False):            # File
    #----------------------------------------------------------------------------------------------
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

        
    #------------------------------------------------------------------------------------------------
    def glob(self, pattern):                                                   # File
    #------------------------------------------------------------------------------------------------
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

    #------------------------------------------------------------------------------------------------
    def exists(self, raise_error=False):                                       # File
    #------------------------------------------------------------------------------------------------
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

    #------------------------------------------------------------------------------------------------
    def __stat(self, attr):                                                    # File
    #------------------------------------------------------------------------------------------------
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

    #------------------------------------------------------------------------------------------------
    def size(self):                                                            # File
    #------------------------------------------------------------------------------------------------
        """
        Returns the size of the file in bytes.

        Returns:
            int: The file size in bytes, or -1 if the file does not exist.
        """
        return self.__stat('st_size')

    #------------------------------------------------------------------------------------------------
    def creation_time(self):                                                   # File
    #------------------------------------------------------------------------------------------------
        """
        Returns the file creation time.

        Returns:
            float: The file's creation time as a timestamp, or -1 if the file does not exist.
        """
        return datetime.fromtimestamp(self.__stat('st_ctime'))

    #------------------------------------------------------------------------------------------------
    def tail(self, **kwargs):                                                  # File
    #------------------------------------------------------------------------------------------------
        """
        Returns the last few lines of the file.

        Returns:
            str: The last lines of the file.
        """
        return next(tail_file(self.path, **kwargs), '')

    #------------------------------------------------------------------------------------------------
    def reversed(self, **kwargs):                                             # File
    #------------------------------------------------------------------------------------------------
        """
        Reads the file's content in reverse order.

        Returns:
            generator: A generator yielding lines in reverse order.
        """
        return tail_file(self.path, **kwargs)

    #------------------------------------------------------------------------------------------------
    def head(self, **kwargs):                                                  # File
    #------------------------------------------------------------------------------------------------
        """
        Returns the first few lines of the file.

        Returns:
            str: The first lines of the file.
        """
        return next(head_file(self.path, **kwargs), '')

    #------------------------------------------------------------------------------------------------
    def lines(self):                                                           # File
    #------------------------------------------------------------------------------------------------
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

    #------------------------------------------------------------------------------------------------
    def line_matching(self, word):                                             # File
    #------------------------------------------------------------------------------------------------
        """
        Finds the first line in the file that contains the specified word.

        Args:
            word (str): The word to search for in the file.

        Returns:
            str: The first line that contains the word, or None if not found.
        """
        return next((line for line in self.lines() if word in line), None)

    #------------------------------------------------------------------------------------------------
    def last_line(self):                                                       # File
    #------------------------------------------------------------------------------------------------
        """
        Returns the last line of the file.

        Returns:
            str: The last line of the file.
        """
        return last_line(self.path)

    #------------------------------------------------------------------------------------------------
    def backup(self, tag, overwrite=False):                                    # File
    #------------------------------------------------------------------------------------------------
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


    #----------------------------------------------------------------------------------------------
    def replace_text(self, text=(), pos=()):                                   # File
    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
    def append_bytes(self, data):                                              # File
    #----------------------------------------------------------------------------------------------
        with open(self.path, 'ab') as file:
            file.write(data)

    #----------------------------------------------------------------------------------------------
    def write_bytes(self, data):                                              # File
    #----------------------------------------------------------------------------------------------
        with open(self.path, 'wb') as file:
            file.write(data)

