from __future__ import annotations

from itertools import chain, groupby, islice, repeat
from mmap import mmap, ACCESS_READ
from operator import sub as subtract
from struct import pack, unpack, error as struct_error

from numpy import (array as nparray, asarray, char as npchar, cumsum, dtype as npdtype,
    frombuffer, ndarray, split as npsplit)

from ...core import File, DTYPE
from ...config import DEBUG, ENDIAN
from ...utils import (batched, batched_when, ensure_bytestring, expand_pattern, flatten, flatten_all,
    index_limits, match_in_wildlist, nth, pad, pairwise, slice_range, string_split, take)

__all__ = ["unfmt_header", "unfmt_block", "unfmt_file", "ENDSOL"]

#==================================================================================================
class unfmt_header:                                                                  # unfmt_header
#==================================================================================================
    """Metadata describing an unformatted Eclipse block header."""

    #         | h e a d e r  |     d a t a     |     d a t a     |    d a t a      |
    #         |4i|8s|4i|4s|4i|4i| 1000 data |4i|4i| 1000 data |4i|4i| 1000 data |4i|
    #  bytes  |      24      |4 | 1000*size |  8  | 1000*size |  8  | 1000*size |4 |

    #pack_format = 'i8si4si'

    #----------------------------------------------------------------------------------------------
    def __init__(self, key:str=b'', length:int=0, type:str=b'',
                 startpos:int=0, endpos:int=0):                                      # unfmt_header
    #----------------------------------------------------------------------------------------------
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
    #----------------------------------------------------------------------------------------------
    def from_bytes(cls, _bytes, startpos=0):                                         # unfmt_header
    #----------------------------------------------------------------------------------------------
        """Create an instance from raw bytes."""
        try:
            # Header is 24 bytes, but we skip size int of length 4 before and after
            # Length of data must be 24 - 8 = 16
            key, length, typ = unpack(ENDIAN+'8si4s', _bytes)
            return cls(key, length, typ, startpos)
        except (ValueError, struct_error):
            return False

    #----------------------------------------------------------------------------------------------
    def as_bytes(self):                                                              # unfmt_header
    #----------------------------------------------------------------------------------------------
        """Return the data serialized as bytes."""
        #  | h e a d e r  |
        #  |4i|8s|4i|4s|4i|
        #key = self.key if isinstance(self.key, bytes) else self.key.encode()
        #data = (16, ensure_bytes(self.key), self.length, ensure_bytes(self.type), 16)
        return pack(ENDIAN+'i8si4si', 16, self._key, self.length, ensure_bytestring(self.type), 16)

    #----------------------------------------------------------------------------------------------
    def __str__(self):                                                               # unfmt_header
    #----------------------------------------------------------------------------------------------
        """
        Return a string representation of the unfmt_header object.
        """
        return (f'key={self._key.decode():8s}, type={self.type.decode():4s}, bytes={self.bytes:8d},' 
                f'length={self.length:8d}, start={self.startpos:8d}, end={self.endpos:8d}')

    #----------------------------------------------------------------------------------------------
    def _data_pos(self, pos):                                                        # unfmt_header
    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
    def _data_slices(self, limits=((None,),)):                                       # unfmt_header
    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
    def is_char(self):                                                               # unfmt_header
    #----------------------------------------------------------------------------------------------
        """
        Check if the data type is a character (string) type.

        Returns:
        bool: True if the data type is character-based, False otherwise.
        """
        return self.type[0:1] == b'C'




#==================================================================================================
class unfmt_block:                                                                    # unfmt_block
#==================================================================================================
    """Container for an unformatted Eclipse data block."""

    #
    # Block of unformatted Eclipse data
    #
    #  | h e a d e r  |   d a t a     |
    #  |4i|8s|4i|4s|4i|4i|1000 data|4i| 
    #  |    24 bytes  |
    #  |              |4i|8d| 

    #----------------------------------------------------------------------------------------------
    def __init__(self, header:unfmt_header=None, data=None,
                 file=None, file_obj=None):                                           # unfmt_block
    #----------------------------------------------------------------------------------------------
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
    #----------------------------------------------------------------------------------------------
    def from_data(cls, key:str, data, _dtype):                                        # unfmt_block
    #----------------------------------------------------------------------------------------------
        """Create an instance from existing data."""
        dtype = {'int':b'INTE', 'float':b'REAL', 'double':b'DOUB',
                 'bool':b'LOGI', 'char':b'CHAR', 'mess':b'MESS'}[_dtype]
        if isinstance(data, ndarray) and data.ndim > 1:
            # Flatten multi-dimensional arrays
            data = data.flatten(order='F')
        header = unfmt_header(ensure_bytestring(key.ljust(8)[:8]), len(data), dtype)
        return cls(header, data)

    #----------------------------------------------------------------------------------------------
    def binarydata(self):                                                             # unfmt_block
    #----------------------------------------------------------------------------------------------
        """Return the file contents as bytes."""
        sl = slice(self.header.startpos, self.header.endpos)
        if self._data:
            return self._data[sl]
        return self.read_file(sl)


    #----------------------------------------------------------------------------------------------
    def __str__(self):                                                                # unfmt_block
    #----------------------------------------------------------------------------------------------
        """
        Return a string representation of the unfmt_block object.
        """
        return str(self.header)

    #----------------------------------------------------------------------------------------------
    def __repr__(self):                                                               # unfmt_block
    #----------------------------------------------------------------------------------------------
        """
        Return a formal string representation of the unfmt_block object.
        """
        return f'<{self}>'

    #----------------------------------------------------------------------------------------------
    def __contains__(self, key):                                                      # unfmt_block
    #----------------------------------------------------------------------------------------------
        """
        Check if a specific key is present in the block.

        Parameters:
        key (str): The key to check.

        Returns:
        bool: True if the key is present, False otherwise.
        """
        return self.key() == key

    #----------------------------------------------------------------------------------------------
    def __del__(self):                                                                # unfmt_block
    #----------------------------------------------------------------------------------------------
        """
        Destructor for unfmt_block object. Prints debug information if DEBUG is enabled.
        """
        if DEBUG:
            print(f'Deleting {self}')

    #----------------------------------------------------------------------------------------------
    def __getattr__(self, item):                                                      # unfmt_block
    #----------------------------------------------------------------------------------------------
        """
        Delegate attribute access to the associated unfmt_header if the attribute is not found.

        Parameters:
        item (str): The attribute name.

        Returns:
        object: The value of the attribute from the umfmt_header.
        """
        return getattr(self.header, item)

    #----------------------------------------------------------------------------------------------
    def key(self):                                                                    # unfmt_block
    #----------------------------------------------------------------------------------------------
        """
        Return the decoded key of the block.

        Returns:
        str: The key as a string.
        """
        return self.header._key.decode().strip()

    #----------------------------------------------------------------------------------------------
    def type(self):                                                                   # unfmt_block
    #----------------------------------------------------------------------------------------------
        """
        Return the decoded type of the block.

        Returns:
        str: The type of the block.
        """
        return self.header.type.decode()
        
    #----------------------------------------------------------------------------------------------
    def read_file(self, sl:slice):                                                    # unfmt_block
    #----------------------------------------------------------------------------------------------
        """
        Read data from a file within the given slice range.

        Parameters:
        sl (slice): A slice object specifying the range of bytes to read.

        Returns:
        bytes: The read data.
        """
        self._file_obj.seek(sl.start)
        return self._file_obj.read(sl.stop - sl.start)

    #----------------------------------------------------------------------------------------------
    def fix_payload_errors(self):                                                     # unfmt_block
    #----------------------------------------------------------------------------------------------
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

    #----------------------------------------------------------------------------------------------
    def _read_data(self, limit):                                                      # unfmt_block
    #----------------------------------------------------------------------------------------------
        """Read raw data for the current block."""
        slices = tuple(self.header._data_slices(limit))
        if self._data:
            # File is mmap'ed
            data = (self._data[sl] for sl in slices)
        else:
            # File object
            data = (self.read_file(sl) for sl in slices)
        dtype = npdtype(self.dtype.nptype).newbyteorder(ENDIAN)
        return frombuffer(b''.join(data), dtype=dtype)

    #----------------------------------------------------------------------------------------------
    def data_old(self, *index, limit=((None,),), strip=False, unpack=True):           # unfmt_block
    #----------------------------------------------------------------------------------------------
        """Return block data using the legacy reader."""
        if self.header.length == 0:
            return () if unpack else ((),)
        if index:
            limit = [pad(index, 2, fill=index[0]+1)]
        values = iter(self._read_data(limit))
        if self.header.is_char():
            values = (string_split(next(values).decode(), self.header.dtype.size))
            if strip:
                values = (v.strip() for v in values)
        if index:
            pos = slice(*index) if len(index)>1 else index[0]
            ret = tuple(values)[pos]
            return ret
        if None in limit[0]:
            ret = tuple(values) if unpack else (tuple(values),)
            return ret
        ndata = (-subtract(*l) for l in limit)
        return tuple(take(n, values) for n in ndata)

    #----------------------------------------------------------------------------------------------
    def data(self, *index, limit=((None,),), strip=False, unpack=True):               # unfmt_block
    #----------------------------------------------------------------------------------------------
        """
        Returnerer NumPy-arrays. 
        - unpack=True: gir direkte array / tuple-of-arrays / list-of-arrays 
        - unpack=False: pakker det samme resultatet inn i én tuple
        """
        #print(f'{index=}, {limit=}, {strip=}, {unpack=}')
        # 1) Tom fil
        if self.header.length == 0:
            out = nparray([], dtype=self.header.dtype.nptype)
            return out if unpack else (out,)

        # 2) Hvis man bruker index direkte, overstyr limit
        if index:
            limit = [pad(index, 2, fill=index[0] + 1)]

        # 3) Les alt som én flat NumPy-array
        flat = self._read_data(limit)

        # 4) Hvis tegn-felt: dekode, splitte og strippe
        if self.header.is_char():
            # a) Dekode byte-strenger til Python-streng
            decoded = npchar.decode(flat, 'utf-8')
            # b) Split i biter av fast bredde
            pieces = [
                part
                for text in decoded
                for part in string_split(text, self.header.dtype.size)
            ]
            values = nparray(pieces)
            if strip:
                values = npchar.strip(values)
        else:
            values = flat  # vanlig tall-array

        # 5) Direkte indeks-tilfelle
        if index:
            #print('5')
            pos = slice(*index) if len(index) > 1 else index[0]
            sel = values[pos]
            return sel if unpack else (sel,)

        # 6) “Hent alt” dersom None i limit[0]
        if None in limit[0]:
            #print('6')
            out = values
            return out if unpack else (out,)

        # 7) Flere segmenter: split basert på limit
        if len(limit) > 1:
            #print('7')
            # beregn hvor vi skal splitte
            sizes = [stop - start for (start, stop) in limit]
            split_pts = cumsum(sizes)[:-1]
            segments = tuple(npsplit(values, split_pts))
            #print(limit, segments)
            return segments #if unpack else (segments,)

        # 8) Én enkelt del
        #print('8')
        out = values
        return out if unpack else (out,)


    #----------------------------------------------------------------------------------------------
    def _pack_data(self):                                                             # unfmt_block
    #----------------------------------------------------------------------------------------------
        """Pack block data into bytes."""
        # 4i| 1000 data |4i|4i| 1000 data |4i|4i| 1000 data |4i|...|4i| 1000 data |4i|
        dtype = self.header.dtype
        for a,b in slice_range(0, len(self._data), dtype.max):
            length = b - a
            size = length * dtype.size
            #print(dtype.unpack, size, self._data.dtype)
            yield pack(ENDIAN + f'i{length}{dtype.unpack}i', size, *self._data[a:b], size)
        

    #----------------------------------------------------------------------------------------------
    def _pack_data_fast(self):                                                        # unfmt_block
    #----------------------------------------------------------------------------------------------
        """Pack block data using NumPy acceleration."""
        # 4i| 1000 data |4i|4i| 1000 data |4i|4i| 1000 data |4i|...|4i| 1000 data |4i|
        dtype = self.header.dtype
        #np_dt  = npdtype(f'{ENDIAN}{dtype.unpack}{dtype.size}')
        np_dt  = npdtype(f'{ENDIAN}{dtype.nptype}')
        
        for a, b in slice_range(0, len(self._data), dtype.max):
            length = b - a
            size   = length * dtype.size

            # Bulk-konverter dataene
            arr = asarray(self._data[a:b], dtype=np_dt)

            # Pakk size én gang
            payload = pack(ENDIAN + 'i', size)

            # Sett sammen header + data + footer
            yield payload + arr.tobytes() + payload


    #----------------------------------------------------------------------------------------------
    def as_bytes(self):                                                               # unfmt_block
    #----------------------------------------------------------------------------------------------
        """Return the data serialized as bytes."""
        return self.header.as_bytes() + b''.join(self._pack_data_fast())



#==================================================================================================
class unfmt_file(File):                                                                # unfmt_file
#==================================================================================================
    """Reader for unformatted Eclipse binary files."""

    start = None
    end = None
    var_pos = {}

    #----------------------------------------------------------------------------------------------
    def __init__(self, filename, **kwargs):                                            # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Initialize the unfmt_file."""
        super().__init__(filename, **kwargs)
        self._endpos = 0
        if DEBUG:
            print(f'Creating {unfmt_file.__repr__(self)}')

    #----------------------------------------------------------------------------------------------
    def __repr__(self):                                                                # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Return a developer-friendly representation."""
        return f'<{super().__repr__()}, endpos={self._endpos}>'

    #----------------------------------------------------------------------------------------------
    def at_end(self):                                                                  # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Return whether the file cursor is at the end."""
        return self._endpos == self.size()

    #----------------------------------------------------------------------------------------------
    def is_flushed(self, endkey):                                                      # unfmt_file
    #----------------------------------------------------------------------------------------------
        """
        Check if the file is flushed up to the specified end key.

        Args:
            endkey: The key to check against the last block in the file.

        Returns:
            bool: True if the last block's key matches endkey, False otherwise.
        """
        if self.is_file():
            last_block = next(self.tail_blocks(), None)
            if last_block and last_block.key() == endkey:
                return True

    #----------------------------------------------------------------------------------------------
    def reset(self):                                                                   # unfmt_file
    #----------------------------------------------------------------------------------------------
        """
        Resets the internal end position to zero, effectively clearing the file's read position.
        """
        self._endpos = 0

    #----------------------------------------------------------------------------------------------
    def offset(self):                                                                  # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Return the byte offset within the file."""
        return self.size() - self._endpos

    #----------------------------------------------------------------------------------------------
    def __prepare_limits(self, *keylim):                                               # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Return prepare limits."""
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

    #----------------------------------------------------------------------------------------------
    def blocks_to_file(self, filename, keys=None, invert=False, append=False):         # unfmt_file
    #----------------------------------------------------------------------------------------------
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
        if isinstance(filename, File):
            filename = filename.path
        with open(filename, mode) as file:
            for block in self.blocks():
                if block.key() in keylist:
                    file.write(block.binarydata())


    #----------------------------------------------------------------------------------------------
    def blockdata_old(self, *keylim, limits=None, strip=True,
                  tail=False, singleton=False, **kwargs):                              # unfmt_file
    #----------------------------------------------------------------------------------------------
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
                dkeys = list(f'{key}_{l[0]}' for l in limit)
                for i,dk in enumerate(dkeys):
                    data[dk] = values[i]
            # Only data:None is omitted, data:() is allowed
            if not any(val is None for val in data.values()):
            #if all(data.values()):
                if singleton:
                    yield tuple(data.values())
                else:
                    # Unpack single values
                    values = tuple(v if len(v)>1 else v[0] for v in data.values())
                    yield values if len(values)>1 else values[0]
                data = {key:None for key in dictkeys}

    #----------------------------------------------------------------------------------------------
    def blockdata(self, *keylim, limits=None, strip=True,
                  tail=False, singleton=False, only_new=False, **kwargs):              # unfmt_file
    #----------------------------------------------------------------------------------------------
        """ 
        Return data in the order of the given keys, not the reading order.
        The keys-list may contain wildcards (*, ?, [seq], [!seq]) 
        A key may be followed by zero, one, or two index values. 
        Two indices are interpreted as a slice, zero indices are 
        interpreted as the whole array.

        Example: ('KEY1', 10, 20, 'KEY2', 'KEY3', 5)
        """
        #starttime = datetime.now()
        keys, limits, dictkeys = self.__prepare_limits(*keylim)
        limits = dict(zip(keys, limits))
        data = {key:None for key in dictkeys}
        for blocks in self.section_blocks(tail=tail, only_new=only_new, **kwargs):
            for block in blocks:
                if key:=match_in_wildlist(block.key(), keys):
                    limit = limits[key]
                    values = block.data(strip=strip, limit=limit, unpack=False)
                    #print('blockdata', values)
                    dkeys = list(f'{key}_{l[0]}' for l in limit)
                    for i,dk in enumerate(dkeys):
                        data[dk] = values[i]
            # Only data:None is omitted, data:() is allowed
            if not any(val is None for val in data.values()):
                if singleton:
                    result = tuple(data.values())
                else:
                    # Unpack single values
                    values = tuple(v if v.size > 1 else v[0] for v in data.values())
                    result = values if len(values) > 1 else values[0]
                yield result
                data = {key:None for key in dictkeys}


    #----------------------------------------------------------------------------------------------
    def read(self, *varnames, **kwargs):                                               # unfmt_file
    #----------------------------------------------------------------------------------------------
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
        keylim = list(keylim)
        nvar = [len(pos) for _,*pos in var_pos]
        #print(keylim, nvar)
        for values in self.blockdata(*keylim, **kwargs):
            if any(n>1 for n in nvar):
                # Split values to match number of input variables
                values = iter(values)
                yield [take(n, values)[0] if n==1 else take(n, flatten(values)) for n in nvar]
            else:
                yield values


    #----------------------------------------------------------------------------------------------
    def last_value(self, var:str):                                                     # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Return the last value in the block."""
        return next(self.read(var, tail=True), None) or 0

    #----------------------------------------------------------------------------------------------
    def read_header(self, data, startpos):                                             # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Read the next block header."""
        try:
            # Header is 24 bytes, but we skip size int of length 4 before and after
            # Length of data must be 24 - 8 = 16 
            key, length, typ = unpack(ENDIAN+'8si4s', data)
            #key, length, typ = unpack(ENDIAN+'4x8si4s', data) # x is pad byte
            return unfmt_header(key, length, typ, startpos)
        except (ValueError, struct_error):
            return False


    #----------------------------------------------------------------------------------------------
    def blocks(self, only_new=False, start=None, use_mmap=True, **kwargs):             # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Iterate over blocks in the file."""
        if not self.is_file():
            return ()
        startpos = 0
        if only_new:
            startpos = self._endpos
        if start:
            startpos = start
        if self.size() - startpos < 24: # Header is 24 bytes
            return ()
        if use_mmap:
            return self.blocks_from_mmap(startpos, only_new=only_new, **kwargs)
        return self.blocks_from_file(startpos, only_new=only_new)
        #     yield from self.blocks_from_mmap(startpos, **kwargs)
        # yield from self.blocks_from_file(startpos)

    #----------------------------------------------------------------------------------------------
    def blocks_from_mmap(self, startpos, only_new=False, write=False):                 # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Yield blocks streamed from a memory map."""
        try:
            with self.mmap(write=write) as data:
                size = data.size()
                pos = startpos
                while pos < size:
                    header = self.read_header(data[pos+4:pos+20], pos)
                    if not header:
                        return
                    pos = header.endpos
                    if only_new:
                        self._endpos = pos
                    yield unfmt_block(header=header, data=data, file=self.path)
        except ValueError: # Catch 'cannot mmap an empty file'
            return #() #False

    #----------------------------------------------------------------------------------------------
    def blocks_from_file(self, startpos, only_new=False):                              # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Yield blocks streamed directly from the file."""
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
                    self._endpos = pos
                yield unfmt_block(header=header, file_obj=file, file=self.path)


    #----------------------------------------------------------------------------------------------
    def tail_blocks(self, **kwargs):                                                   # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Return the last blocks in the file."""
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

    #----------------------------------------------------------------------------------------------
    def fix_errors(self):                                                              # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Attempt to repair corrupt block boundaries."""
        # If reading from tail does not work we need to fix block payload errors
        if not next(self.tail_blocks(), False):
            # Fix errors in-place
            return sum(b.fix_payload_errors() for b in self.blocks(write=True))
        return 0

    #----------------------------------------------------------------------------------------------
    def count_sections(self):                                                          # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Return the number of sections in the file."""
        return sum(1 for block in self.blocks() if self.start in block)

    #----------------------------------------------------------------------------------------------
    def count_blocks(self):                                                            # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Return the number of blocks per keyword."""
        return sum(1 for _ in self.blocks())

    #----------------------------------------------------------------------------------------------
    def section_keys(self, n=0):                                                       # unfmt_file
    #----------------------------------------------------------------------------------------------
        """ 
        Return keywords from section n, n=0 is default
        """
        return [bl.key() for bl in nth(self.section_blocks(), n)]

    #----------------------------------------------------------------------------------------------
    def section_filepos(self):                                                         # unfmt_file
    #----------------------------------------------------------------------------------------------
        """ 
        Return file-positions at the start of sections
        These positions can be used in blocks_matching(*keys, start=pos) to
        get fast access to the blocks in the section.
        """
        return [bl.startpos for _,bl in self.blocks_matching(self.start)]

    #----------------------------------------------------------------------------------------------
    def check_missing_keys(self, *keys, raise_error=True):                             # unfmt_file
    #----------------------------------------------------------------------------------------------
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
        missing = list( set(keys) - set(self.section_keys()) )
        if missing and raise_error:
            raise ValueError(f'Missing keywords in {self}: {missing}')
        return missing

    #----------------------------------------------------------------------------------------------
    def find_keys(self, *keys, sec=0):                                                 # unfmt_file
    #----------------------------------------------------------------------------------------------
        """
        Return matching keywords from section sec, sec=0 is default
        """
        return expand_pattern(keys, self.section_keys(sec))

    #----------------------------------------------------------------------------------------------
    def blocks_matching(self, *keys, **kwargs):                                        # unfmt_file
    #----------------------------------------------------------------------------------------------
        """
        Yields blocks from the file whose keys match any of the specified keys, along with their associated step.

        Args:
            *keys: Variable length argument list of keys to match against block keys.
            **kwargs: Arbitrary keyword arguments passed to the `blocks` method.

        Yields:
            tuple: A tuple containing the current step (int or relevant type) and the matching block object.

        Notes:
            - The step is updated whenever a block contains `self.start`.
            - Only blocks whose key matches one of the provided keys are yielded.
        """
        step = -1
        keyset = set(keys)
        for b in self.blocks(**kwargs):
            if self.start in b:
                step = b.data()[0]
            #if any(key in b for key in keys):
            if b.key() in keyset:
                yield (step, b)

    #----------------------------------------------------------------------------------------------
    def section_start_indices(self, tail=False, start=None):                           # unfmt_file
    #----------------------------------------------------------------------------------------------
        """
        Generator that yields the block index of the start-block of each section in a file,
        and finally the total number of blocks.
        This is useful for determining the start and end blocks of sections, for example,
        when using 'pairwise' to process sections.
        Args:
            tail (bool, optional): If True, use tail_blocks and end token; otherwise, use blocks and start token.
            start (optional): Starting position or block for iteration (passed to blocks_func).
        Yields:
            int: Block index of the start-block of each section.
            int: After all sections, yields the total number of blocks.
        """
        blocks_func = self.tail_blocks if tail else self.blocks
        start_token = self.end if tail else self.start

        n = 0
        for i, block in enumerate(blocks_func(start=start)):
            n = i + 1
            if start_token in block:
                yield i
        yield n


    #----------------------------------------------------------------------------------------------
    def section_blocks(self, tail=False, only_new=False, **kwargs):                    # unfmt_file
    #----------------------------------------------------------------------------------------------
        """
        Yields batches of blocks corresponding to sections in the file. The sections are 
        determined by the start blocks defined in the file.

        Args:
            tail (bool, optional): If True, use tail_blocks instead of blocks. Defaults to False.
            only_new (bool, optional): If True, process only new blocks since the last read. Defaults to False.
            **kwargs: Additional keyword arguments passed to the blocks function.

        Yields:
            tuple: A batch of blocks for each section.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        self.exists(raise_error=True)
        blocks_func = self.tail_blocks if tail else self.blocks
        start_indx = self.section_start_indices(tail=tail, start=self._endpos if only_new else None)
        # Use pairwise to get start and end positions of each section
        # TODO: use start argument to skip to correct section
        section_pos = islice(pairwise(start_indx), None)
        blocks_iter = blocks_func(only_new=only_new, **kwargs)
        prev = 0
        a, b = next(section_pos, (0, 0))
        while True:
            # Use islice to get blocks from a to b, where a and b are the start and end
            # Need to subtract prev to get the correct slice since the iterator is consumed
            batch = tuple(islice(blocks_iter, a - prev, b - prev))
            if not batch:
                break
            yield batch
            prev = b
            a, b = next(section_pos, (b, b))
            if a == b:  # No more sections
                break
    

    #----------------------------------------------------------------------------------------------
    def section_data(self, start=(), end=(), rename=(), begin=0):                      # unfmt_file
    #----------------------------------------------------------------------------------------------
        """
        Extracts sections of data from a memory-mapped file based on specified start 
        and end block markers.

        Args:
            start (tuple): A tuple specifying the start block marker and its attribute 
                           (e.g., ('SEQNUM', 'startpos')).
            end (tuple): A tuple specifying the end block marker and its attribute 
                         (e.g., ('ENDSOL', 'endpos')).
            rename (tuple, optional): A tuple of (old_name, new_name) pairs to rename 
                                      occurrences in the extracted data.
            begin (int, optional): The minimum step value to begin extraction from. 
                                   Defaults to 0.

        Yields:
            tuple: A tuple containing the step value and the corresponding data slice 
                   from the file, with optional renaming applied.

        Example:
            start = ('SEQNUM', 'startpos')
            end = ('ENDSOL', 'endpos')
            for step, data in section_data(start, end, rename=[('OLDNAME', 'NEWNAME')]):
                # process data
        """
        keys, attrs = zip(start, end)
        pairs = batched(self.blocks_matching(*keys), 2)
        with self.mmap() as filemap:
            for step_pair in pairs:
                step, pair = zip(*step_pair)
                if step[0] < begin:
                    continue
                # Get 'endpos' or 'startpos' attrs for the start/end blocks
                _slice = slice(*(getattr(p,a) for p,a in zip(pair, attrs)))
                data = filemap[_slice]
                if rename and (names:=[rn for rn in rename if rn[0] in data]):
                    for old, new in names:
                        data = data.replace(old.ljust(8), new.ljust(8))
                yield (step[0], data)


    #----------------------------------------------------------------------------------------------
    def section_slices(self, start, end, only_new=False, **kwargs):                    # unfmt_file
    #----------------------------------------------------------------------------------------------
        """
        Yields slices representing file positions between specified start and end blocks.

        Args:
            start (tuple): A tuple specifying the start block key and its attribute (e.g., ('SEQNUM', 'startpos')).
            end (tuple): A tuple specifying the end block key and its attribute (e.g., ('ENDSOL', 'endpos')).
            **kwargs: Additional keyword arguments passed to `section_blocks`.

        Yields:
            tuple: A tuple containing:
                - step (int): The step value, typically extracted from the block containing `self.start`.
                - slice (slice): A slice object representing the file position between the start and end blocks.

        Example:
            start = ('SEQNUM', 'startpos')
            end = ('ENDSOL', 'endpos')
            for step, file_slice in obj.section_slices(start, end):
                # Use file_slice to access the desired section in the file.
        """
        # Start by splitting args in keys=('SEQNUM', 'ENDSOL') and attrs=('startpos', 'endpos')
        keys, attrs = zip(start, end)
        step = -1
        _matches = {k:None for k in keys}
        for section in self.section_blocks(only_new=only_new, **kwargs):
            for block in section:
                if self.start in block:
                    step = block.data()[0]
                if any(key in block for key in keys):
                    _matches[block.key()] = block
            # Get 'endpos' or 'startpos' for the start/end blocks
            _slice = slice(*(getattr(p,a) for p,a in zip(_matches.values(), attrs)))
            yield (step, _slice)
            _matches = {k:None for k in keys}

    #----------------------------------------------------------------------------------------------
    def section_data2(self, start=(), end=(), rename=(), begin=0):                     # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Return data grouped by section."""
        # Extract data from sections defined by start and end keywords. 
        # Used to merge sections of unfmt-files
        with self.mmap() as filemap:
            for step, _slice in self.section_slices(start, end):
                if step < begin:
                    continue
                data = filemap[_slice]
                if rename and (names:=[rn for rn in rename if rn[0] in data]):
                    for old, new in names:
                        data = data.replace(old.ljust(8), new.ljust(8))
                yield (step, data)

    #----------------------------------------------------------------------------------------------
    def section_start_end_blocks(self, **kwargs):                                      # unfmt_file
    #----------------------------------------------------------------------------------------------
        """
        Yields pairs of blocks that mark the start and end of sections within the file.

        Iterates over blocks whose keys match either the section start or end keywords.
        Groups these blocks into pairs (start, end) using the `batched` function.
        If a pair contains identical keys, raises a ValueError indicating an incomplete section.

        Keyword Arguments:
            **kwargs: Additional arguments passed to the `blocks` method.

        Yields:
            tuple: A tuple containing the start and end block for each section.

        Raises:
            ValueError: If a section is incomplete (i.e., missing a start or end keyword).
        """
        endwords = [self.start, self.end]
        ends = (bl for bl in self.blocks(**kwargs) if bl.key() in endwords)
        for first, last in batched(ends, 2):
            if first.key() == last.key():
                endwords.remove(first.key())
                raise ValueError(f"Incomplete section: '{endwords[0]}' keyword is missing")
            yield (first, last)

    #----------------------------------------------------------------------------------------------
    def merge(self, *section_data, progress=lambda x:None, cancel=lambda:None):        # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Merge consecutive blocks with identical keys."""
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

    #----------------------------------------------------------------------------------------------
    def remove_sections(self, nsec):                                                   # unfmt_file
    #----------------------------------------------------------------------------------------------
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


    #----------------------------------------------------------------------------------------------
    def assert_no_duplicates(self, raise_error=True):                                  # unfmt_file
    #----------------------------------------------------------------------------------------------
        """Ensure that block keys are unique."""
        allowed = (self.start, 'ZTRACER')
        seen = set()
        duplicate = (key for b in self.blocks() if (key:=b.key()) in seen or seen.add(key))
        if (dup:=next(duplicate, None)) and dup not in allowed:
            msg = f'Duplicate keyword {dup} in {self}'
            if raise_error:
                raise SystemError('ERROR ' + msg)
            print('WARNING ' + msg)

    #----------------------------------------------------------------------------------------------
    def reshape_dim(self, *data, dtype=None):                                          # unfmt_file
    #----------------------------------------------------------------------------------------------
        """ 
            Reshape data arrays to the dimensions defined by self.dim()
        """
        if not hasattr(self, 'dim'):
            raise AttributeError(f'No dim attribute in {type(self).__name__}')
        return [
            None if d is None else asarray(d, dtype=dtype).reshape(self.dim(), order='F')
            for d in data
        ]

#--------------------------------------------------------------------------------------------------
ENDSOL = unfmt_block.from_data('ENDSOL', [], 'mess')
#--------------------------------------------------------------------------------------------------
# Empty block that terminates a SEQNUM - ENDSOL section in UNRST-files

