
# -*- coding: utf-8 -*-

from operator import itemgetter, sub
from pathlib import Path
from re import findall, compile as re_compile, DOTALL, sub as re_sub, IGNORECASE, MULTILINE
from subprocess import check_output
from threading import Thread
from time import sleep, time
from datetime import timedelta, datetime, time as dt_time
from mmap import mmap, ACCESS_READ
from signal import SIGTERM
from contextlib import contextmanager
from itertools import chain, groupby, islice, tee, zip_longest
from collections import deque, namedtuple
from collections.abc import Iterable
from shutil import copy2, rmtree
import stat
from fnmatch import fnmatch
from os import SEEK_END, SEEK_CUR
from numpy import (arange, array, meshgrid, stack, trapezoid, sum as npsum, concatenate, 
                   diff as npdiff, where, append as npappend, roll as nproll, ndarray)
from psutil import Process, NoSuchProcess, wait_procs
#from operator import attrgetter
from matplotlib.pyplot import figure as pl_figure, show as pl_show, close as pl_close
from molmass import Formula

# Short Python regexp guide:
#   \s : whitespace, [ \t\n\r\f\v]
#   \w : alphanumeric, [a-zA-Z0-9_]
#   \d : decimal digit, [0-9]
#   \b : word-delimiter
#    ? : 0 or 1 repetitions
#    + : 1 or more rep.
#    * : 0 or more rep.

#--------------------------------------------------------------------------------
def to_coords(flat, shape, as_list=False):
#--------------------------------------------------------------------------------
    """
    Converts a 1D array of flat indices `flat` to (x, y, z) coordinates.
    flat: array of shape (N,)
    shape: tuple of grid dimensions (X, Y, Z)
    Returns: list of tuples with coordinates of shape (N, 3)
    """

    if not isinstance(flat, ndarray):
        flat = array(flat)
    z = flat // (shape[0] * shape[1])
    y = (flat // shape[0]) % shape[1]
    x = flat % shape[0]
    if as_list:
        return list(zip(x.tolist(), y.tolist(), z.tolist()))
    return stack((x, y, z), axis=1)
    # Y, Z = shape[1], shape[2]
    # x, rem = divmod(flat, Y * Z)
    # y, z = divmod(rem, Z)
    # return list(zip(x, y, z))

#--------------------------------------------------------------------------------
def to_flat(coords, shape, as_list=False):
#--------------------------------------------------------------------------------
    """Convert 3D coordinates to flat index."""
    if not isinstance(coords, ndarray):
        coords = array(coords)
    x, y, z = coords.T #moveaxis(coords, 0, 1)
    flat = x + y * shape[0] + z * shape[0] * shape[1]
    if as_list:
        return flat.tolist()
    return flat

#--------------------------------------------------------------------------------
def neighbour_connections(dim):
#--------------------------------------------------------------------------------
    """
    Generate the connection indices for the six block-faces in a 3D grid.

    This function computes the indices of neighboring blocks in both positive 
    and negative directions along each axis (i, j, k) for a 3D grid of given 
    dimensions.

    Args:
        dim (tuple of int): The dimensions of the 3D grid as a tuple (nx, ny, nz), 
                            where nx, ny, and nz are the sizes along the x, y, 
                            and z axes, respectively.

    Returns:
        tuple:
            - pos_neigh (numpy.ndarray): Indices of neighboring blocks in the 
            positive direction along each axis. The shape of the array is 
            (nx, ny, nz, 3, 3), where the (3, 3) array are the (i, j, k) indices 
            of the positive neighbors along the x, y, and z axes.
            - neg_neigh (numpy.ndarray): Indices of neighboring blocks in the 
            negative direction along each axis. The shape of the array is 
            (nx, ny, nz, 3, 3), where the last dimension corresponds to the 
            negative neighbors along the x, y, and z axes.
    """
    ind = index_array(dim)
    kwargs = {'as_scalar': True, 'wrapped': -1}
    # Connection-indices of block (i,j,k) in 0) i+1, 1) j+1, 2) k+1 direction
    pos_conn = roll_xyz(ind, -1, **kwargs).swapaxes(-2, -1)
    # Connection-indices of block (i,j,k) in 0) i-1, 1) j-1, 2) k-1 direction
    neg_conn = roll_xyz(ind, 1, **kwargs).swapaxes(-2, -1)
    return concatenate((pos_conn, neg_conn), axis=-2)

#--------------------------------------------------------------------------------
def roll_xyz(src, shift=1, as_scalar=False, wrapped=0):
#--------------------------------------------------------------------------------
    """
    Rolls the values of a source array along all axes and removes periodic 
    boundary end elements.

    Parameters:
        src (numpy.ndarray): The source array to be rolled. Can be a scalar 
            field or a vector field.
        shift (int, optional): The number of positions by which elements are 
            shifted. Positive values shift to the right, and negative values 
            shift to the left. Default is 1.
        as_scalar (bool, optional): If True, treats the input as a scalar 
            field even if it has more than 3 dimensions. Default is False.

    Returns:
        numpy.ndarray: A new array with the rolled values along all axes. 
        For vector fields, the result is stacked along the last axis.

    Notes:
        - For vector fields (when `src.ndim > 3` and `as_scalar` is False), 
            each axis is rolled individually.
        - Periodic boundary end elements (at index 0 or -1, depending on the 
            shift direction) are set to 0 after rolling.
    """
    end = 0 if shift > 0 else -1
    roll_list = []
    arr = src
    for axis in range(3):
        ind = 3 * [slice(None)]
        ind[axis] = end
        if src.ndim > 3 and not as_scalar:
            # Vectorfield, roll axes individually
            arr = src[..., axis]
        rolled = nproll(arr, shift, axis=axis)
        # Remove periodic boundary end elements
        rolled[tuple(ind)] = wrapped
        roll_list.append(rolled)
    return stack(roll_list, axis=-1)


#--------------------------------------------------------------------------------
def index_array(shape):
#--------------------------------------------------------------------------------
    i, j, k = meshgrid(arange(shape[0]), arange(shape[1]), arange(shape[2]), indexing='ij')
    return stack((i, j, k), axis=-1)

#--------------------------------------------------------------------------------
def run_length_encode(data):
#--------------------------------------------------------------------------------
    # Create a mask detecting changes in consecutive elements
    mask = concatenate(([True], data[:-1] != data[1:]))
    # Compute run lengths
    counts = npdiff(where(npappend(mask, True))[0])
    # Zip counts and unique values
    return list(flatten(zip(counts, data[mask])))

#--------------------------------------------------------------------------------
def any_cell_in_box(cells, box):
#--------------------------------------------------------------------------------
    """
    Return True if any of the cells given by ((i1, j1, k1), (i2, j2, k2)) is inside 
    the box given by ((i_min, i_max), (j_min, j_max), (k_min, k_max))
    """
    for cell in cells:
        if all(box[n][0] <= cell[n] < box[n][1] for n in range(3)):
            return True
    return False

#--------------------------------------------------------------------------------
def bounding_box(pos):
#--------------------------------------------------------------------------------
    return [(p[0], p[-1]) for p in map(sorted, zip(*pos))]

#--------------------------------------------------------------------------------
def get_terminal_environment(var, file='~/.bashrc'):
#--------------------------------------------------------------------------------
    env = check_output(['bash', '-c', f'source {file} && env'], text=True)
    match = (v for v in env.split('\n') if v.startswith(var))
    return next(match, '=').split('=')[1]

#--------------------------------------------------------------------------------
def call_if_callable(func, *args, **kwargs):
#--------------------------------------------------------------------------------
    """
    Calls the given function if it is callable.

    Parameters:
    func (callable): The function to be called.
    *args: Variable length argument list to pass to the function.
    **kwargs: Arbitrary keyword arguments to pass to the function.

    Returns:
    The return value of the function if it is callable, otherwise False.
    """
    if callable(func):
        return func(*args, **kwargs)
    return False

#--------------------------------------------------------------------------------
def dates_after(date, datelist):
#--------------------------------------------------------------------------------
    """
    Filters a list of dates, returning only those that are on or after a specified date.

    Args:
        date (datetime.date): The reference date to compare against.
        datelist (list of datetime.date): A list of dates to be filtered.

    Returns:
        list of datetime.date: A list of dates from `datelist` that are on or after the specified `date`.
    """
    return [d for d in datelist if d >= date]

#--------------------------------------------------------------------------------
def empty_folder(folder):
#--------------------------------------------------------------------------------
    if not isinstance(folder, Path):
        folder = Path(folder)
    if folder.exists():
        rmtree(folder)
    folder.mkdir()
    return folder

#--------------------------------------------------------------------------------
def slice_range(start=0, stop=None, step=None):
#--------------------------------------------------------------------------------
    range_with_stop = chain(range(start, stop, step), [stop])
    return (pair for pair in pairwise(range_with_stop))

#--------------------------------------------------------------------------------
def split_number(input_number, base):
#--------------------------------------------------------------------------------
    # Finn antall hele base-enheter
    num_full_units = input_number // base
    # Finn rest
    remainder = input_number % base
    # Lag en liste med alle base-enhetene, pluss resten hvis den finnes
    result = [base] * num_full_units
    if remainder > 0:
        result.append(remainder)
    return result

#--------------------------------------------------------------------------------
def ensure_bytestring(astring:str):
#--------------------------------------------------------------------------------
    if isinstance(astring, bytes):
        return astring
    return astring.encode()

#--------------------------------------------------------------------------------
def ensure_list(values):
#--------------------------------------------------------------------------------
    try:
        return values[:]
    except TypeError:
        return [values]
    # if isinstance(var, list):
    #     return var
    # return [var]

#--------------------------------------------------------------------------------
def unique_key(key, keylist:list, symbol='#'):
#--------------------------------------------------------------------------------
    # Append number if key is not unique
    if (count := keylist.count(key)):
        key += f'{symbol}{count}'
    return key

#--------------------------------------------------------------------------------
def list_prec(iterable, fmt='.2e'):
#--------------------------------------------------------------------------------
    return '[' + ', '.join(f'{i:{fmt}}' for i in iterable) + ']'

#--------------------------------------------------------------------------------
def batched_as_list(iterable, n):
#--------------------------------------------------------------------------------
    for batch in batched(iterable, n):
        yield list(batch)

#--------------------------------------------------------------------------------
def missing_elements(L):
#--------------------------------------------------------------------------------
    """
    Return a set of consecutive elements missing in the given list:
        missing_elements([1,3,4]) -> {2}
    """
    L = sorted(L)
    return set(range(L[0], L[-1]+1)).difference(L)

#--------------------------------------------------------------------------------
def ppm2molL(specie:str):
#--------------------------------------------------------------------------------
    # PPM is g/Mg, where Mg is mega-gram
    # Divide by Mg = 1000 kg to convert to g/kg
    # For a dilute solution, 1 kg of water equals 1 L: g/kg = g/L 
    # Divide g/L by atomic weight (g/mol) to get mol/L
    return 1/1000/Formula(specie).mass

#--------------------------------------------------------------------------------
def molL2ppm(specie:str):
#--------------------------------------------------------------------------------
    return 1000*Formula(specie).mass

#-----------------------------------------------------------------------
def day2time(days):
#-----------------------------------------------------------------------
    rest = (days%1)*24*3600 # Convert day to seconds
    time = [int(days)]
    for base in (3600, 60, 1):
        time.append(rest//base)
        rest -= time[-1]*base
    return namedtuple('Time', 'day hour min sec msec')(*time, rest)

#-----------------------------------------------------------------------
def float_range(start, stop, step):
#-----------------------------------------------------------------------
    while start < stop:
        yield round(start, 10)  # rounding to avoid floating-point arithmetic issues
        start += step

#-----------------------------------------------------------------------
def date_range(start, stop, step=1, fmt=None):
#-----------------------------------------------------------------------
    """
    daterange((1971, 7, 1), 10, format="%d-%b-%Y") -> ['01-Jul-1971', '02-Jul-1971', '03-Jul-1971']
    """
    if not isinstance(start, datetime):
        start = datetime(*start)
    dates = (start + timedelta(days=d) for d in float_range(0, stop, step))
    if fmt:
        return [date.strftime(fmt) for date in dates]
    return list(dates)

#-----------------------------------------------------------------------
def first_index(cond, alist, fail=None):
#-----------------------------------------------------------------------
    """
    Return index of first element that satisfy condition cond. If cond is 
    never true, return last index
    """
    return next((i for i,v in enumerate(alist) if cond(v)), fail) 


#-----------------------------------------------------------------------
def batched_when(A, cond):
#-----------------------------------------------------------------------    
    pos = chain((i for i,a in enumerate(A) if cond(a)), [len(A)])
    return (A[a:b] for a,b in pairwise(pos))


#-----------------------------------------------------------------------
def pad(A, length, fill=None):
#-----------------------------------------------------------------------    
    if isinstance(A, tuple):
        fill = (fill,)
    else:
        fill = [fill]
    return A + fill*(length-len(A))

#-----------------------------------------------------------------------
def to_letter(num, base=26, case='lower'):
#-----------------------------------------------------------------------
    if num < 1:
        return ''
    num -= 1
    q = num//base
    shift = {'upper':65, 'lower':97}
    a = chr(shift[case] + num - q*base)
    return to_letter(q, base, case) + a

#-----------------------------------------------------------------------
def letter_range(length, base=26, case='lower'):
#-----------------------------------------------------------------------
    for i in range(length):
        yield to_letter(i+1, base, case)

#-----------------------------------------------------------------------
def has_write_access(path, error=False):
#-----------------------------------------------------------------------
    path = Path(path)
    name = 'write.access'
    if path.is_file():
        test = path.with_name(name)
    else:
        test = path/name
    try:
        test.touch()
    except PermissionError as perm_err:
        if error:
            raise SystemError(error) from perm_err
        return False
    test.unlink()
    return True


#-----------------------------------------------------------------------
def cumtrapz(y, x, *args, **kwargs):
#-----------------------------------------------------------------------
    """
    Alternative to scipy cumptrapz using numpy trapz
    
    Example:
        x = np.linspace(0, 2*np.pi)
        pl.figure()
        pl.plot(x, cumtrapz(np.sin(x), x), 'ro')
        pl.plot(x, -np.cos(x) + 1, 'b-')
    """
    return array([trapezoid(y[:i], x[:i], *args, **kwargs) for i in range(1, len(x)+1)])

#-----------------------------------------------------------------------
def match_in_wildlist(string, wildlist):
#-----------------------------------------------------------------------
    """ Return matched string in a list of strings that can include wildcards """
    return next((pattern for pattern in wildlist if fnmatch(string, pattern)), None)
    #return any(fnmatch(string, pattern) for pattern in wildlist)

#-----------------------------------------------------------------------
def expand_pattern(patterns, strings, invert=False):
#-----------------------------------------------------------------------
    """ 
    Return expanded patterns that match strings. For invert=True, return 
    strings that do not match any pattern. Pattern order is preserved for 
    non-inverted patterns.
    """
    if invert:
        return [s for s in strings if not any(fnmatch(s, pat) for pat in patterns)]
    return [s for pat in patterns for s in strings if fnmatch(s, pat)]

#-----------------------------------------------------------------------
def make_user_executable(path):
#-----------------------------------------------------------------------
    path = Path(path)
    print('before:', path, stat.filemode(path.stat().st_mode))
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    print('after:', path, stat.filemode(path.stat().st_mode))

#-----------------------------------------------------------------------
def split_in_lines(text):
#-----------------------------------------------------------------------
    return (line for t in text.split('\n') if (line:=t.strip()))
    # if text:
    #     return [line for t in text.split('\n') if (line:=t.strip())]
    # return []

#-----------------------------------------------------------------------
def running_jupyter():
#-----------------------------------------------------------------------
    from IPython import get_ipython
    if get_ipython():
        return True
    return False

#-----------------------------------------------------------------------
def ordered_intersect(A, B):
#-----------------------------------------------------------------------
    B = frozenset(B)
    return [a for a in A if a in B]

#-----------------------------------------------------------------------
def ordered_intersect_index(A, B):
#-----------------------------------------------------------------------
    B = frozenset(B)
    return [i for i,a in enumerate(A) if a in B]

#-----------------------------------------------------------------------
def removeprefix(prefix, string):
#-----------------------------------------------------------------------
    if string.startswith(prefix):
        return string[len(prefix):]
    return string

#-----------------------------------------------------------------------
def group_indices(A):
#-----------------------------------------------------------------------
    """ 
    Group consecutive indices into (first, last) limits. 
    Single values are also handled: (2,) -> (2,3) 
    Example:
        group_indices((1,2,3,4,8,9,10,20)) --> ((1,5),(8,11),(20,21))
    """
    jumps = (p for p in pairwise(A) if -sub(*p)>1)
    return ((a, b+1) for a,b in batched(chain([A[0]], *jumps, [A[-1]]), 2))
    #return batched(chain([A[0]], jumps, [A[-1]]), 2) 


#-----------------------------------------------------------------------
def index_limits(index):
#-----------------------------------------------------------------------
    """ 
    Group consecutive indices into (first, last) limits 
    Return () if first < 0
    Return (0,1) if given (0,)
    """
    # index_lim((1,2,3,4,8,9,10)) --> (1,5),(8,11)
    jumps = (index[0],) + flat_list((a,b) for a,b in pairwise(index) if b-a>1) + (index[-1],)
    #limits = [(a,b+1) if a>=0 else () for a,b in grouper(jumps, 2)]
    limits = [() if a<0 else (a,b+1) for a,b in grouper(jumps, 2)]
    return limits


#-----------------------------------------------------------------------
def string_split(str, l, strip=False):
#-----------------------------------------------------------------------
    strings = (str[i:i+l] for i in range(0, len(str), l))
    if strip:
        return (s.strip() for s in strings)
    return strings

#-----------------------------------------------------------------------
def copy_recursive(src, dst, log=None) -> None:
#-----------------------------------------------------------------------
    """ Copy a src file or a src folder recursively to the
        dst file or dst folder 
    """
    src = Path(src)
    dst = Path(dst)
    if src.is_file():
        src_items = (src,)
    else:
        # src is folder
        src_items = tuple(src.iterdir())
    if dst.is_file():
        dst_items = (dst,)
        dst = dst.parent
    else:
        # dst is folder
        dst_items = (dst/item.name for item in src_items)
    dst.mkdir(exist_ok=True, parents=True)
    #print(dst)
    #print(src_items)
    #print(dst_items)
    for _src, _dst in zip(src_items, dst_items):
        #new_file = dst/item.name
        if _src.is_file():
            copy2(_src, _dst)
            if log:
                log(f'Copied {_src} -> {_dst}')
        else:
            copy_recursive(_src, _dst, log=log)


### pairwise is new in python 3.10, define it for older versions
# try:
#     from itertools import pairwise
# except ImportError:
#     from itertools import tee

# Taken from: https://stackoverflow.com/questions/14822184/is-there-a-ceiling-equivalent-of-operator-in-python
#-----------------------------------------------------------------------
def ceildiv(a, b):
#-----------------------------------------------------------------------
    return -(-a//b)

#-----------------------------------------------------------------------
def sliding_window(iterable, n): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    "Collect data into overlapping fixed-length chunks or blocks."
    # sliding_window('ABCDEFG', 4) --> ABCD BCDE CDEF DEFG
    it = iter(iterable)
    window = deque(islice(it, n-1), maxlen=n)
    for x in it:
        window.append(x)
        yield tuple(window)

#-----------------------------------------------------------------------
def pairwise(iterable): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

#-----------------------------------------------------------------------
def triplewise(iterable): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    "Return overlapping triplets from an iterable"
    # triplewise('ABCDEFG') --> ABC BCD CDE DEF EFG
    for (a, _), (b, c) in pairwise(pairwise(iterable)):
        yield a, b, c

#-----------------------------------------------------------------------
def nth(iterable, n, default=None): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)
    
#-----------------------------------------------------------------------
def take(n, iterable): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    "Return first n items of the iterable as a list"
    return tuple(islice(iterable, n))
    #return list(islice(iterable, n))

#-----------------------------------------------------------------------
def tail(n, iterable): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    "Return an iterator over the last n items"
    # tail(3, 'ABCDEFG') --> E F G
    return iter(deque(iterable, maxlen=n))

# #------------------------------------------------
# def tail_file(fname, n=0):
# #------------------------------------------------
#     'Return the last n lines of a file'
#     if fname and Path(fname).is_file():   
#         with open(fname) as f:
#             return tail(n,f)
#     return iter(())

#------------------------------------------------
def last_line(path):
#------------------------------------------------
#
# Taken from https://stackoverflow.com/questions/46258499/how-to-read-the-last-line-of-a-file-in-python
#
    with open(path, 'rb') as file:
        try:  # catch OSError in case of a one line file
            file.seek(-2, SEEK_END)
            while file.read(1) != b'\n':
                file.seek(-2, SEEK_CUR)
        except OSError:
            file.seek(0)
        return file.readline().decode()


#------------------------------------------------
def tail_file(path, size=10*1024, size_limit=False):
#------------------------------------------------
    """ 
    A generator that yields chunks the file starting from the end.
    
    Arguments:
        size : default, 10 kilobytes
            byte-size of the file-chunks
            
        size_limit : default, False
            return None if the size of the file is less than size   
    """
    path = Path(path)
    if not path.is_file():
        return
    filesize = path.stat().st_size
    if size_limit and filesize < size:
        return
    size = pos = min(size, filesize)
    with open(path, 'rb') as file:
        while pos <= filesize:
            file.seek(-pos, 2)
            yield decode(file.read(size))
            if pos == filesize:
                return
            pos = min(pos + size, filesize)


#------------------------------------------------
def head_file(path, size=10*1024):
#------------------------------------------------
    path = Path(path)
    if not path.is_file():
        return
    with open(path, 'rb') as file:
        while data := file.read(size):
            yield decode(data)


#-----------------------------------------------------------------------
def prepend(value, iterator): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    "Prepend a single value in front of an iterator"
    # prepend(1, [2, 3, 4]) -> 1 2 3 4
    return chain([value], iterator)

#-----------------------------------------------------------------------
def flatten(list_of_lists): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    return chain.from_iterable(list_of_lists)

#-----------------------------------------------------------------------
def flat_list(list_or_tuple): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    "Flatten one level of nesting"
    try:
        flat = chain.from_iterable(list_or_tuple)
        if isinstance(list_or_tuple, list):
            return list(flat)
        return tuple(flat)
    except TypeError:
        return list_or_tuple

#-----------------------------------------------------------------------
def flatten_all(list_of_lists):  #https://stackoverflow.com/questions/2158395/flatten-an-irregular-arbitrarily-nested-list-of-lists        
#-----------------------------------------------------------------------
    "Flatten arbitrarily nested lists"
    for list_ in list_of_lists:
        if isinstance(list_, Iterable) and not isinstance(list_, (str, bytes)):
            yield from flatten_all(list_)
        else:
            yield list_


#-----------------------------------------------------------------------
def grouper(iterable, n, *, incomplete='fill', fillvalue=None): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, fillvalue='x') --> ABC DEF Gxx
    # grouper('ABCDEFG', 3, incomplete='strict') --> ABC DEF ValueError
    # grouper('ABCDEFG', 3, incomplete='ignore') --> ABC DEF
    args = [iter(iterable)] * n
    if incomplete == 'fill':
        return zip_longest(*args, fillvalue=fillvalue)
    if incomplete == 'strict':
        return zip(*args, strict=True)
    if incomplete == 'ignore':
        return zip(*args)
    else:
        raise ValueError('Expected fill, strict, or ignore')

#-----------------------------------------------------------------------
def batched(iterable, n): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch
        
#-----------------------------------------------------------------------
def consume(iterator, n=None): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    "Advance the iterator n-steps ahead. If n is None, consume entirely."
    # Use functions that consume iterators at C speed.
    if n is None:
        # feed the entire iterator into a zero-length deque
        deque(iterator, maxlen=0)
    else:
        # advance to the empty slice starting at position n
        next(islice(iterator, n, n), None)


# #--------------------------------------------------------------------------------
# def flat_list(alist):
# #--------------------------------------------------------------------------------
#     return [item for sublist in alist for item in sublist]


#-----------------------------------------------------------------------
def iter_index(iterable, value, start=0): # From https://docs.python.org/3/library/itertools.html
#-----------------------------------------------------------------------
    "Return indices where a value occurs in a sequence or iterable."
    # iter_index('AABCADEAF', 'A') --> 0 1 4 7
    seq_index = iterable.index
    i = start - 1
    try:
        while True:
            yield (i := seq_index(value, i+1))
    except ValueError:
        pass

#-----------------------------------------------------------------------
def groupby_sorted(iterable, key=None, reverse=False):
#-----------------------------------------------------------------------
    """ Sort before applying groupby and remove key from result """
    key = key or itemgetter(0)
    ordered = sorted(iterable, key=key, reverse=reverse)
    for tag, groups in groupby(ordered, key):
        yield tag, [[g for g in group if g != tag] for group in groups]

#-----------------------------------------------------------------------
def get_tuple(tuple_list_or_val):
#-----------------------------------------------------------------------
    if isinstance(tuple_list_or_val, (tuple, list)):
        return tuple_list_or_val
    # Is a value, create tuple
    return (tuple_list_or_val,)

#-----------------------------------------------------------------------
def unique_names(names, sep='-'):
#-----------------------------------------------------------------------
    """ Append number to identical names """
    new_names = []
    for i, name in enumerate(names):
        tot = names.count(name)
        count = names[:i+1].count(name)
        new_names.append(f'{name}{sep}{count-1}' if tot > 1 and count > 1 else name)
    return new_names

#-----------------------------------------------------------------------
def split_by_words(string, words): #, wb=r'\b'):
#-----------------------------------------------------------------------
    """
    Split a string (possibly bytes-like), with comments, into sections based on a list of unique words.
    Returns a dict with words as keys and a tuple of begin and end positins
    """
    #regex =  r'^\s*(\b' + r'\b|\b'.join(words) + r'\b)'
    regex =  r'^\s*\b(' + '|'.join(words) + r')\b'
    if isinstance(string, bytes):
        regex = regex.encode()
    matches_ = re_compile(regex, flags=IGNORECASE|MULTILINE).finditer(string)
    # Append string end pos as tuple of tuple
    tag_pos = chain( ((m.group(1), m.start()) for m in matches_), [('', len(string))] )
    return ((tag, a, b) for (tag, a), (_, b) in pairwise(tag_pos))
    #return [(a[0],a[1],b[1]) for a,b in pairwise(tag_pos)]


# #-----------------------------------------------------------------------
# def get_keyword(file, keyword, end='', comment='#', ignore_case=True, raise_error=True):
# #-----------------------------------------------------------------------
#     if not Path(file).is_file():
#         return []
#     flags = 0
#     if ignore_case:
#         flags = RegexFlag.IGNORECASE
#     data = remove_comments(file, comment=comment, raise_error=raise_error)
#     if data == []:
#         return []
#     space = r'\s' # regex space
#     slash = '/'
#     if end in (' ',r'\s','\n','\t'):
#         end = space
#         space = ''
#     if end == slash:
#         slash = ''
#     # Lookahead used at the end to mark end without consuming
#     regex = re_compile(fr"{keyword}\s+([0-9A-Za-z._+:{space}{slash}\\-]+)(?={end})", flags=flags)
#     return [list(convert_float_or_str(v.split())) for v in regex.findall(data)]


#--------------------------------------------------------------------------------
def convert_float_or_str(words):
#--------------------------------------------------------------------------------
    for w in words:
        try:
            v = float(w)
        except ValueError:
            v = str(w).strip()
        yield v

#-----------------------------------------------------------------------
def string_in_file(string, file):
#-----------------------------------------------------------------------
    with open(file, 'rb') as f:
        output = f.read()
    return string.encode() in output


@contextmanager
#-----------------------------------------------------------------------
def safezip(*gen):
#-----------------------------------------------------------------------
    """
    Zip generators and close them if the zip exits. Zip exits when the first generator 
    is exhausted, causing the __exit__() function not to be called for the non-exhausted 
    generators. This routine fixes that problem by closing all the generators explicitly.
    """
    try:
        yield zip(*gen)
    finally:
        for g in gen:
            g.close()


#-----------------------------------------------------------------------
def remove_chars(chars, text):
#-----------------------------------------------------------------------
    for c in chars:
        if c in text:
            text = text.replace(c,'')
    return text

#-----------------------------------------------------------------------
def remove_leading_nondigits(txt):
#-----------------------------------------------------------------------
    return re_sub(r'^[a-zA-Z-+._]*', '', txt)

#-----------------------------------------------------------------------
def try_except_loop(*args, limit=1, pause=0.05, error=None, raise_error=True, func=None, log=None, **kwargs):
#-----------------------------------------------------------------------
    for i in range(limit):
        #print(f'{func.__qualname__}({args},{kwargs}): {i}')
        try:
            result = func(*args, **kwargs)
            break
        except error as err:
            if log:
                log(i, err)
            sleep(pause)
    if i==limit-1 and raise_error:
        raise SystemError(f'Unable to complete {func.__qualname__} within {limit} tries during {limit*pause} seconds: {error}')
    return result

#-----------------------------------------------------------------------
def kill_process(pid, signal=SIGTERM, children=False, timeout=5, on_terminate=None):
#-----------------------------------------------------------------------
    procs = []
    parent = try_except_loop(pid, func=Process, limit=10, pause=0.05, error=NoSuchProcess)
    if children:
        procs.extend(parent.children(recursive=True))
    procs.append(parent)
    for p in procs:
        try:
            p.send_signal(signal)
        except NoSuchProcess:
            pass
    gone, alive = wait_procs(procs, timeout=timeout, callback=on_terminate)
    for p in alive:
        p.kill()
    return gone + alive

#-----------------------------------------------------------------------
def pad_zero(lists):
#-----------------------------------------------------------------------
    N = max([len(a) for a in lists])
    return [a+(N-len(a))*['0'] for a in lists]


#-----------------------------------------------------------------------
def strip_zero(numbers):
#-----------------------------------------------------------------------
    return [f'{num:.3f}'.rstrip('0').rstrip('.') for num in numbers]

#-----------------------------------------------------------------------
def file_exists(file, raise_error=False):
#-----------------------------------------------------------------------
    if Path(file).is_file():
        return True
    else:
        if raise_error:
            raise SystemError(f'ERROR {Path(file).name} does not exist')
        else:
            return False

#-----------------------------------------------------------------------
def file_not_empty(file):
#-----------------------------------------------------------------------
    file = Path(file)
    if file.is_file() and file.stat().st_size > 0:
        return True
    return False


#--------------------------------------------------------------------------------
def get_python_version():
#--------------------------------------------------------------------------------
    from sys import version_info
    return version_info

#--------------------------------------------------------------------------------
def same_length(*lists):
#--------------------------------------------------------------------------------
    it = iter(lists)
    len1 = len(next(it))
    return all(len(l) == len1 for l in it)


#--------------------------------------------------------------------------------
def clear_dict(adict):
#--------------------------------------------------------------------------------
    for val in adict.values():
        if isinstance(val, list):
            val.clear()
        elif isinstance(val, dict):
            clear_dict(val)

#--------------------------------------------------------------------------------
def print_dict(adict):
#--------------------------------------------------------------------------------
    return ', '.join([f'{k}={v}' for k,v in adict.items()])

#--------------------------------------------------------------------------------
def print_file(file):
#--------------------------------------------------------------------------------
    with open(file) as f:
        lines = f.readlines()
    print(''.join(lines))

#--------------------------------------------------------------------------------
def print_error(func):
#--------------------------------------------------------------------------------
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except SystemError as e:
            print('\n   ' + str(e) + '\n')
    return wrapper

#--------------------------------------------------------------------------------
def read_line(n, file, raise_error=True):
#--------------------------------------------------------------------------------
    file = Path(file)
    if not file.is_file():
        if raise_error:
            raise SystemError(f'ERROR {file} not found in read_line()')
        return ''
    with open(file, 'r') as fileobj:
        for _ in range(n-1):
            fileobj.readline()
        return fileobj.readline()

#--------------------------------------------------------------------------------
def read_file(file, raise_error=True, skip=None):
#--------------------------------------------------------------------------------
    file = Path(file)
    if not file.is_file():
        if raise_error:
            raise SystemError(f'ERROR {file} not found in read_file()')
        return ''
    with open(file, 'rb') as fileobj:
        if skip:
            fileobj.seek(skip)
        return decode(fileobj.read())
    
#--------------------------------------------------------------------------------
def decode(data):
#--------------------------------------------------------------------------------
    encoding = ('utf-8', 'latin1')
    for enc in encoding:
        try:
            return data.decode(encoding=enc)
        except UnicodeError:
            continue
    raise SystemError(f'ERROR decode with {encoding} encoding failed!')

#--------------------------------------------------------------------------------
def write_file(path, text):
#--------------------------------------------------------------------------------
    for encoding in ('utf-8', 'latin-1'):
        try:
            with open(path, 'w', encoding=encoding) as file:
                return file.write(text)
        except UnicodeError:
            continue
    raise SystemError(f'ERROR Unable to write to file {Path(path).name}')


#--------------------------------------------------------------------------------
def remove_comments(path, comment='--', join=True, raise_error=True):
#--------------------------------------------------------------------------------
    path = Path(path)
    try:
        if not path.is_file():
            if raise_error:
                raise SystemError(f'ERROR {path} not found in remove_comments()')
            return []
    except PermissionError:
        return []
    comment = comment.encode()
    with open(path, 'rb') as file:
        data = file.read()
        lines = (line.split(comment)[0].strip() for l in data.split(b'\n') if (line:=l.strip()) and not line.startswith(comment))
        if join:
            return decode(b'\n'.join(lines)) + '\n'
        return [decode(line) for line in lines]

# #--------------------------------------------------------------------------------
# def remove_comments(path, comment='--', join=True, raise_error=True, encoding=None, end=None):
# #--------------------------------------------------------------------------------
#     try:
#         path = Path(path)
#         if not path.is_file:
#             if raise_error:
#                 raise SystemError(f'ERROR {path} not found in remove_comments()')
#             return []
#         with open(path, encoding=encoding) as file:
#             lines = (line.split(comment)[0].strip() for l in file if (line:=l.strip()) and not line.startswith(comment))
#             if end:
#                 # If end should be included
#                 #lines = chain(takewhile(lambda x: x != end, lines), (end,))
#                 lines = takewhile(lambda x: x != end, lines)
#             if join:
#                 return '\n'.join(lines)+'\n'
#             return list(lines)
#     except (FileNotFoundError, PermissionError):
#         return []
#     except UnicodeDecodeError as e:
#         if encoding:
#             raise SystemError('ERROR {path} raised UnicodeDecodeError for both UTF-8 and latin-1 encodings: {e}')
#         return remove_comments(path, encoding='latin-1', comment=comment, join=join, raise_error=raise_error, end=end)


#--------------------------------------------------------------------------------
def safeindex(alist, value):
#--------------------------------------------------------------------------------
    return alist.index(value) if value in alist else None

#--------------------------------------------------------------------------------
def date_to_datetime(dates):
#--------------------------------------------------------------------------------
    return [datetime.combine(d, dt_time.min) for d in dates]

#--------------------------------------------------------------------------------
def upper_and_lower(alist):
#--------------------------------------------------------------------------------
    return flat_list([[item.upper(),item.lower()] for item in alist])

#--------------------------------------------------------------------------------
def is_file_ignore_suffix_case(file):
#--------------------------------------------------------------------------------
    file = Path(file)
    files = [ file.with_suffix(e) for e in upper_and_lower([file.suffix]) ]
    for f in files:
        if f.is_file():
            return f
    return False

#--------------------------------------------------------------------------------
def replace_line(fname, find=None, replace=None):
#--------------------------------------------------------------------------------
    if not Path(fname).is_file():
        return False
    with open(fname, 'r') as file:
        lines = file.readlines()
    pos = [i for i,line in enumerate(lines) if find in line]
    if pos:
        lines[pos[0]] = replace
    else:
        lines.append(replace)
    with open(fname, 'w') as file:
        file.write(''.join(lines))
    return True


# #--------------------------------------------------------------------------------
# def file_contains(fname, text='', regex='', comment='#', end=None, raise_error=True):
# #--------------------------------------------------------------------------------
#     #print(f'file_contains({fname}, {text})')
#     if not Path(fname).is_file():
#         if raise_error:
#             raise SystemError('ERROR ' + fname + ' not found in file_contains()')    
#         else:
#             return False
#     if isinstance(text, str):
#         text = [text]
#     regex = [rf'\b{t}\b' for t in text]
#     lines = remove_comments(fname, comment=comment, end=end)
#     if any(search(r, lines) for r in regex):
#         return True
#     # for reg in regex:
#     #     #regex = rf'\b{text}\b'
#     #     if search(reg, lines): 
#     #         return True
#     return False

#--------------------------------------------------------------------------------
def delete_all(folder, keep_folder=False, ignore_error=()):
#--------------------------------------------------------------------------------
    if not Path(folder).is_dir():
        return
    for child in Path(folder).iterdir():
        try:
            if child.is_file():
                child.unlink()
            else:
                delete_all(child)
        except ignore_error:
            pass
    if not keep_folder:
        Path(folder).rmdir()
    
#--------------------------------------------------------------------------------
def return_matching_string(str_list, string):
#--------------------------------------------------------------------------------
    for s in str_list:
        if s in string:
            return s
    
#--------------------------------------------------------------------------------
def get_substrings(string, length):
#--------------------------------------------------------------------------------
    length = int(length)
    return [string[i:i+length].strip() for i in range(0,len(string),length)]
    
#--------------------------------------------------------------------------------
def delta_timestring(string1, string2):
#--------------------------------------------------------------------------------
    FMT = '%H:%M:%S'
    sec = datetime.strptime(string1, FMT) - datetime.strptime(string2, FMT)
    return str(timedelta(seconds=int(sec)))
    
# #--------------------------------------------------------------------------------
# def exit_without_atexit():
# #--------------------------------------------------------------------------------
#     #from os import _exit as os_exit
#     os._exit(0)

#--------------------------------------------------------------------------------
def assert_python_version(major=None, minor=None):
#--------------------------------------------------------------------------------    
    from sys import version_info

    ##############################################################################################
    ##                                                                                          ##
    ##  Note: This assert_python_version function will never work.                              ##
    ##                                                                                          ##
    ##  One has to write a translation unit in code                                             ##
    ##  accepted by python 1, 2 and 3 where the version number is checked...                    ##
    ##  (Code digested by python 2 and 3 is on file IORSim_runner.py)                           ##
    ##                                                                                          ##
    ##  As it is now, pre-3.6 interpreters are bailing out with 'SyntaxError: invalid syntax'.  ##
    ##  Python 2.7.18: File "ior2ecl.py", line 36,                                              ##
    ##  Python 3.4.2:  File "ior2ecl.py", line 76                                               ##
    ##                                                                                          ##
    ##############################################################################################
    
    # check python version
    sysmajor = version_info.major
    sysminor = version_info.minor
    if sysmajor<major or (sysmajor==major and sysminor<minor):
        raise SystemExit('\nThis script requires python{}.{} or higher, you are using python{}.{}\n'.
                         format(major, minor, sysmajor, sysminor))
    return True

#------------------------------------------------
def silentdelete(*fname, echo=False):
#------------------------------------------------
    # if isinstance(fname, (str,Path)):
    #     fname = [str(fname)]
    # if isinstance(fname, (list, tuple)): # or isinstance(fname, tuple):
    for f in fname:
        file = Path(f)
        try:
            file.is_file() and file.unlink()
        except (PermissionError, FileNotFoundError) as e:
            if echo:
                print(f'Unable to delete {f}: {e}')
            # else:
            #     pass
        # else:
        if echo:
            print(f'Deleted {f}')
    # else:
    #     raise SystemError(f'silentdelete: Unknown format {type(fname)} passed')


#--------------------------------------------------------------------------------
def float_or_str(words): 
#--------------------------------------------------------------------------------
    if not isinstance(words, (list, tuple)):
        words = (words,)
    # try:
    #     iterator = iter(words)
    # except TypeError: 
    #     iterator = (words,)
    values = []
    for w in words: #iterator:
        try:
            v = float(w)
        except ValueError:
            v = str(w)
        values.append(v)
    return values

    
#------------------------------------------------
def delete_files_matching(*pattern, echo=False, raise_error=False):
#------------------------------------------------
    msg = ''
    # if not isinstance(pattern, (list, tuple)):
    #     pattern = (pattern,)
    for pat in pattern:
        pat = Path(pat)
        for file in pat.parent.glob(pat.name):
            if echo:
                print('Removing ' + str(file))
            try:
                file.unlink()
            except PermissionError as err:
                msg = 'WARNING Unable to delete file '+str(file)+', maybe it belongs to another process'
                if raise_error:
                    raise SystemError(msg) from err
    return msg


#------------------------------------------------
def loop_until(func, *args, limit=None, pause=None, loop_func=None, **kwargs):
#------------------------------------------------
    n = 0
    if not loop_func:
        loop_func = lambda:None
    while True:
        if func(*args, **kwargs):
            return n
        if pause:
            sleep(pause)
        n += 1
        if limit and n > limit:
            return -1
        loop_func()



#------------------------------------------------
def list2str(alist, start='', end='', sep='', count=False):
#------------------------------------------------
    #return start + '%s'%', '.join(f'{sep}{i}{sep}' for i in alist) + end
    if count:
        return ', '.join([f'{v} (n={npsum(array(alist)==v)})' for v in set(alist)])
    return f"{start}{', '.join(f'{sep}{i}{sep}' for i in alist)}{end}"

#------------------------------------------------
def list2text(alist):
#------------------------------------------------
    text = ', '.join([str(a) for a in alist])
    return ' and'.join(text.rsplit(',',1))

# #------------------------------------------------
# def tail_file(fname, nchars=0, nlines=0):
# #------------------------------------------------
#     if Path(fname).is_file():
#         if nlines:
#             with open(fname) as f:
#                 lines = f.readlines()                  
#                 return ''.join(lines[-nlines:])
#         else:
#             with open(fname, 'rb') as f:
#                 f.seek(-nchars, 2)
#                 return f.read(nchars).decode()
#     return ''


#------------------------------------------------
def safeopen(filename, mode):
#------------------------------------------------
    try:
        filehandle = open(filename, mode)
        return filehandle
    except OSError as error:
        raise SystemError(f'Unable to open file {filename}: {error}') from error
        
#--------------------------------------------------------------------------------
def warn_empty_file(file, comment=''):
#--------------------------------------------------------------------------------
    with open(file, 'r') as f:
        for line in f:
            if not line.startswith(comment) and not line.isspace():
                return
    print(f'WARNING! {file} is empty')

#--------------------------------------------------------------------------------
def matches(file=None, pattern=None, length=0, multiline=False, pos=None, check=None):
#--------------------------------------------------------------------------------
    if not Path(file).is_file() or Path(file).stat().st_size < 1:
        return []
    flags = 0
    if multiline:
        flags = DOTALL
    regexp = re_compile(pattern.encode(), flags=flags)
    with open(file) as f:
        with mmap(f.fileno(), length=length, access=ACCESS_READ) as data:
            if pos:
                data = data[pos:]
            if check and not check in data:
                data = b''
            yield from regexp.finditer(data)
            # for match in regexp.finditer(data):
            #     yield match

#--------------------------------------------------------------------------------
def bytes_string(size, d=0):
#--------------------------------------------------------------------------------
    i = 0
    unit = {0:'', 1:'K', 2:'M', 3:'G', 4:'T'}
    while size > 1024:
        i += 1
        size /= 1024
    return f'{size:.{d}f} {unit[i]}'

#--------------------------------------------------------------------------------
def number_of_blocks(file=None, blockstart=None):
#--------------------------------------------------------------------------------
    prev = None
    for match in matches(file=file, pattern=blockstart):
        if prev:
            blocksize = match.start()-prev
            break
        prev = match.start()
    return round(Path(file).stat().st_size/blocksize)

#--------------------------------------------------------------------------------
def count_match(file=None, pattern=None):
#--------------------------------------------------------------------------------
    with open(file) as f:
        # Use mmap to read the whole file into memory
        wholefile = mmap(f.fileno(), 0, access=ACCESS_READ)
        return len(findall(pattern.encode(), wholefile))



#====================================================================================
class Progress:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, N=1, format='%', indent=3, min=0):
    #--------------------------------------------------------------------------------
        """
        format: '%'   - show percent
                '10#' - show max 10 hashes 
        """
        self.start_time = datetime.now()
        #self.update = update
        self.N = N
        self.n = 0
        self.n0 = 0       # n0 != 0 if time-counter is reset
        self.min = min    # sets a minimum for the progress bar
        if '%' in format:
            self.format = self.format_percent
        if '#' in format:
            self.format = self.format_bar
            try:
                n = int(format.split('#')[0])
            except ValueError:
                n = 1
            self.bar_length = n
        self.indent = indent*' '
        #self.eta = 0
        self.eta = None
        self.time_last_eta = None
        self.time_str = '--:--:--'
        self.length = 0
        self.prev_n = -1
        #print('Progress:', N, update, format, indent, min)

    #--------------------------------------------------------------------------------
    def __str__(self):
    #--------------------------------------------------------------------------------
        return ', '.join(f'{k}:{v}' for k,v in self.__dict__.items() if k[0] != '_' and not callable(v))
        #return f'<Progress start_time:{self.start_time}, N:{self.N}, n0:{self.n0}, min:{self.min}, eta:{self.eta}, last_eta:{self.time_last_eta}, {self.time_str}>'

    #--------------------------------------------------------------------------------
    def set_min(self, min):
    #--------------------------------------------------------------------------------
        self.reset_time()
        self.min = self.n0 = min
        #print('set_min:', min)

    #--------------------------------------------------------------------------------
    def reset(self, N=1, **kwargs):
    #--------------------------------------------------------------------------------
        self.N = N
        self.n0 = 0
        # self.eta = None
        # self.time_last_eta = None
        self.reset_time(**kwargs)

    #--------------------------------------------------------------------------------
    def reset_time(self, n=0, min=None):
    #--------------------------------------------------------------------------------
        #print('reset_time', n)
        self.start_time = datetime.now() #time()
        #self.start_time = None
        #self.min = min if min else 0
        self.min = min or 0
        # self.n0 = n
        self.time_str = '--:--:--'
        self.n0 = max(n, self.min)
        self.prev_n = -1
        self.eta = None
        self.time_last_eta = None

    # #--------------------------------------------------------------------------------
    # def calc_estimated_arrival(self, n):
    # #--------------------------------------------------------------------------------
    #     nn = max(n-self.min, 0)
    #     Dt = time()-self.start_time
    #     self.ela = timedelta(seconds=int(Dt))
    #     self.eta = timedelta(seconds=int((self.N-n)*Dt/nn))

    #--------------------------------------------------------------------------------
    def format_percent(self, n):
    #--------------------------------------------------------------------------------
        nn = max(n-self.min, 0)
        percent = 100*nn/(self.N-self.min)
        # return 'Progress {: 4d} / {:4d} = {:.0f} %   ETA: {}'.format(int(n), int(self.N), percent, self.eta) 
        return f'Progress {n: 4d} / {self.N:4d} = {percent:.0f} %   ETA: {self.eta}' 

    #--------------------------------------------------------------------------------
    def format_bar(self, n):
    #--------------------------------------------------------------------------------
        return f'{self.fraction(n)}  [{self.bar(n)}]  {self.time_str}'

    #--------------------------------------------------------------------------------
    def bar(self, n):
    #--------------------------------------------------------------------------------
        hash = 0
        nn = max(n-self.min, 0)
        if (diff := self.N-self.min) > 0:
            hash = int(self.bar_length*nn/diff)
        rest = self.bar_length - hash
        if hash <= self.bar_length:        
            return f'{hash*"#"}{rest*"-"}'
        return f'-- E R R O R, n:{n}, N:{self.N}, min:{self.min} --'

    #--------------------------------------------------------------------------------
    def fraction(self, n):
    #--------------------------------------------------------------------------------
        n = n or 0  # if n is None
        nn = max(n-self.min, 0)
        t, T = strip_zero((n, self.N))
        if self.min > 0 and n >= self.min:
            a, b = strip_zero((self.min, nn))
            t = f'({a} + {b})'
        #print(f'fraction: {t} / {T}')
        return f'{t} / {T}'

    #--------------------------------------------------------------------------------
    def set_N(self, N):
    #--------------------------------------------------------------------------------
        self.N = N

    #--------------------------------------------------------------------------------
    def print(self, n=-1, head=None, text=None):
    #--------------------------------------------------------------------------------
        if n<0:
            self.n += 1
            n = self.n
        self.remaining_time(n)
        line = self.format(n)
        trail_space = max(1, self.length - len(line))
        self.length = len(line)
        print('\r' + (head+' ' if head else '') + self.indent + line 
               + (' '+text if text else '') + trail_space*' ', end='', flush=True)

    #--------------------------------------------------------------------------------
    def remaining_time(self, n):
    #--------------------------------------------------------------------------------
        time_ = timedelta(0)
        if self.prev_n < self.min:
            self.start_time = datetime.now()
        if n>self.prev_n and n>self.min and n>self.n0:
            ### Calculate estimated time of arrival, eta
            nn = n-self.n0
            eta = max( int( (self.N-n) * (datetime.now()-self.start_time).total_seconds()/nn ) , 0)
            self.eta = timedelta(seconds=eta)
            ### Time of this estimate (used if progress printed more often than estimated)
            self.time_last_eta = datetime.now()
            time_ = self.eta
        elif self.eta:
            time_ = self.eta-(datetime.now()-self.time_last_eta)
        self.prev_n = n        
        time_ = max(timedelta(0), time_)
        self.time_str = str(time_).split('.')[0]
        return self.time_str        


#====================================================================================
class Timer:
#====================================================================================

    #--------------------------------------------------------------------------------    
    def __init__(self, filename=None):
    #--------------------------------------------------------------------------------    
        self.counter = 0
        self.timefile = Path(f'{filename}_timer.dat')
        self.timefile.write_text('# step \t seconds\n')
        self.starttime = time()
        self.info = f'Execution time saved in {self.timefile.name}'


    #--------------------------------------------------------------------------------    
    def start(self):
    #--------------------------------------------------------------------------------    
        self.counter += 1
        self.starttime = time()


    #--------------------------------------------------------------------------------    
    def stop(self):
    #--------------------------------------------------------------------------------    
        with self.timefile.open('a') as f:
            f.write(f'{self.counter:d}\t{time()-self.starttime:.3e}\n')

#====================================================================================
class TimerThread:
#====================================================================================
    DEBUG = False
    
    #--------------------------------------------------------------------------------    
    def __init__(self, limit=0, prec=0.5, func=None):
    #--------------------------------------------------------------------------------    
        self._func = func
        self._call_func = func
        self._limit = limit
        self._idle = prec   # Idle time between checks given by precision
        self._running = False
        self._starttime = None
        self._endtime = None
        self._thread = Thread(target=self._timer, daemon=True)
        self.DEBUG and print(f'Creating {self}')

    #--------------------------------------------------------------------------------    
    def __str__(self):
    #--------------------------------------------------------------------------------    
        return f'<TimerThread (limit={self._limit}, prec={self._idle}, func={self._func.__qualname__}, thread={self._thread})>'

    #--------------------------------------------------------------------------------    
    def __del__(self):
    #--------------------------------------------------------------------------------    
        self.DEBUG and print(f'Deleting {self}')

    #--------------------------------------------------------------------------------    
    def __enter__(self):
    #--------------------------------------------------------------------------------    
        return self

    #--------------------------------------------------------------------------------    
    def __exit__(self, exc_type, exc_value, traceback):
    #--------------------------------------------------------------------------------    
        self.close()

    #--------------------------------------------------------------------------------    
    def endtime(self):
    #--------------------------------------------------------------------------------    
        return self._endtime

    #--------------------------------------------------------------------------------    
    def uptime(self):
    #--------------------------------------------------------------------------------    
        return self._limit - self.time()


    #--------------------------------------------------------------------------------    
    def start(self):
    #--------------------------------------------------------------------------------    
        self._endtime = None
        self._call_func = self._func
        self._starttime = datetime.now()
        if not self._running:
            self._running = True
            self._thread.start()

    #--------------------------------------------------------------------------------    
    def close(self):
    #--------------------------------------------------------------------------------    
        self._running = False
        if self._thread.is_alive():
            self._thread.join()

    #--------------------------------------------------------------------------------    
    def cancel_if_alive(self):
    #--------------------------------------------------------------------------------    
        #print((datetime.now()-self._starttime).total_seconds(), self._is_alive)
        if not self._endtime:
            self._call_func = lambda : None
            self._endtime = self.time()
            return True
        return False

    #--------------------------------------------------------------------------------    
    def is_alive(self):
    #--------------------------------------------------------------------------------    
        return not self._endtime

    #--------------------------------------------------------------------------------    
    def time(self):
    #--------------------------------------------------------------------------------    
        return (datetime.now()-self._starttime).total_seconds()

    #--------------------------------------------------------------------------------    
    def _timer(self):
    #--------------------------------------------------------------------------------    
        while self._running:
            sleep(self._idle)
            if not self._endtime:
                time = self.time()
                if time >= self._limit:
                    self._call_func()
                    self._endtime = time
                    #print('Called '+self._caller.__qualname__+f' at {sec}')




#====================================================================================
class LivePlot:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, figure=1, func=None, loop=None, **kwargs):              # Plot
    #--------------------------------------------------------------------------------
        from IPython import get_ipython
        if ipython := get_ipython():
            ipython.run_line_magic('matplotlib', 'widget')
        else:
            msg = 'ERROR! LivePlot can only be used inside a Jupyter Notebook/IPython session'
            raise SystemError(msg)
        pl_close('all')
        self.fig = pl_figure(figure) #, clear=True)
        canvas = self.fig.canvas
        canvas.header_visible = False
        #print(self.fig)
        #canvas.layout.width = '200px'
        #canvas.layout.height = '200px'
        pl_show()
        #self.fig.canvas.draw()
        self.func = func
        self.kwargs = kwargs
        self.running = False
        self.loop = loop

    #--------------------------------------------------------------------------------
    #def loop(self, wait=1.0, thread=None):                 # Plot
    def start(self, wait=1.0):                 # Plot
    #--------------------------------------------------------------------------------
        import asyncio
        async def update():
            self.running = True
            #while thread and thread.is_alive():
            while self.running:
                #print('calling!')
                self.func(**self.kwargs)
                self.fig.canvas.draw_idle()
                await asyncio.sleep(wait)
            #print('LivePlot has stopped!')
        #loop = asyncio.get_running_loop()
        #loop.create_task(update())
        #print(loop)
        #print(asyncio.get_event_loop())
        if self.loop:
            self.loop.run_until_complete(update())
        else:
            loop = asyncio.get_running_loop()
            loop.create_task(update())

    #--------------------------------------------------------------------------------
    def stop(self):                 # Plot
    #--------------------------------------------------------------------------------
        if self.running:
            print('Stopping LivePlot!')
            self.running = False
        else:
            print('LivePlot is not running!')

