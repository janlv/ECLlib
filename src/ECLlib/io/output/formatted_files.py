"""
Eclipse/INTERSECT formatted output files

Formatted Eclipse files are written in ASCII text rather than Fortran unformatted
(binary) format. Each data block is preceded by an 8-character keyword, followed by
a count of items and a 4-character data type descriptor (e.g. INTE, REAL, DOUB, CHAR).
They can be inspected and parsed using standard text tools and are useful for
debugging, validation, or creating reduced datasets for sharing.

File types:
- FUNRST: Formatted unified restart file containing human-readable cell-based 
  solution data (pressures, saturations, etc.) for all report steps.
- RSM: Run summary file containing tabular time-series output (e.g. production 
  rates, totals, pressures) in a simple column-based text format, readable by 
  spreadsheets and plotting tools.
"""

from collections import namedtuple
from itertools import chain, islice, repeat
from mmap import mmap
from pathlib import Path
from struct import pack

from ...config import ENDIAN
from ...core import DTYPE, File
from ...utils import batched, float_or_str, flatten
from .unformatted_files import UNRST_file

__all__ = ["fmt_block", "fmt_file", "FUNRST_file", "RSM_block", "RSM_file"]

#==================================================================================================
class fmt_block:                                                                        # fmt_block
#==================================================================================================
    """Representation of a formatted Eclipse block."""

    #----------------------------------------------------------------------------------------------
    def __init__(self, key=None, length=None, datatype=None, data=(),
                 filemap:mmap=None, start=0, size=0):                                   # fmt_block
    #----------------------------------------------------------------------------------------------
        """Initialize the fmt_block."""
        self._key = key
        self._length = length
        self._dtype = DTYPE[datatype]
        self.data = data
        self.filemap = filemap
        self.startpos = start
        self.size = size
        self.endpos = start + size

    #----------------------------------------------------------------------------------------------
    def __str__(self):                                                                  # fmt_block
    #----------------------------------------------------------------------------------------------
        """Return a human-readable representation."""
        return (f'key={self.key():8s}, type={self._dtype.name:4s},' 
                f'length={self._length:8d}, start={self.startpos:8d}, end={self.endpos:8d}')

    #----------------------------------------------------------------------------------------------
    def __repr__(self):                                                                 # fmt_block
    #----------------------------------------------------------------------------------------------
        """Return a developer-friendly representation."""
        return f'<{type(self)}, key={self.key():8s}, type={self._dtype.name}, length={self._length:8d}>'

    #----------------------------------------------------------------------------------------------
    def __contains__(self, key:str):                                                    # fmt_block
    #----------------------------------------------------------------------------------------------
        """Return whether the value exists."""
        return self.key() == key

    #----------------------------------------------------------------------------------------------
    def is_last(self):                                                                  # fmt_block
    #----------------------------------------------------------------------------------------------
        """Return whether the block is the final one."""
        return self.endpos == self.filemap.size()

    #----------------------------------------------------------------------------------------------
    def formatted(self):                                                                # fmt_block
    #----------------------------------------------------------------------------------------------
        """Return the block rendered as formatted text."""
        return self.filemap[self.startpos:self.endpos]

    #----------------------------------------------------------------------------------------------
    def key(self):                                                                      # fmt_block
    #----------------------------------------------------------------------------------------------
        """Return the block keyword."""
        return self._key.decode().strip()
        #return self.keyword.strip()
    
    #----------------------------------------------------------------------------------------------
    def as_binary(self):                                                                # fmt_block
    #----------------------------------------------------------------------------------------------
        """Return the block encoded as binary data."""
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
                
    #----------------------------------------------------------------------------------------------
    def print(self):                                                                    # fmt_block
    #----------------------------------------------------------------------------------------------
        """Print the current progress line."""
        print(self._key, self._length, self._dtype.name)



#==================================================================================================
class fmt_file(File):                                                                    # fmt_file
#==================================================================================================
    """Base reader for formatted Eclipse output."""

    #----------------------------------------------------------------------------------------------
    def __init__(self, filename, **kwargs):                                              # fmt_file
    #----------------------------------------------------------------------------------------------
        """Initialize the fmt_file."""
        super().__init__(filename, **kwargs)
        self.start = None 

    #----------------------------------------------------------------------------------------------
    def blocks(self):                                                                    # fmt_file
    #----------------------------------------------------------------------------------------------
        """Iterate over blocks in the file."""
        def double(string):
            """Return the data interpreted as double precision."""
            return float(string.replace(b'D',b'E'))
        def logi(string):
            """Return logical values decoded from the block."""
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

    #----------------------------------------------------------------------------------------------
    def first_section(self):                                                             # fmt_file
    #----------------------------------------------------------------------------------------------
        """Return the first data section."""
        # Get number of blocks and size of first section
        secs = ((i,b) for i, b in enumerate(self.blocks()) if 'SEQNUM' in b)
        count, first_block_next_section = tuple(islice(secs, 2))[-1]
        return namedtuple('section','count size')(count, first_block_next_section.startpos)

    #----------------------------------------------------------------------------------------------
    def section_blocks(self, count=None, with_attr:str=None):                            # fmt_file
    #----------------------------------------------------------------------------------------------
        """Return blocks grouped per section."""
        count = count or self.section_count()
        if with_attr:
            return batched((getattr(b, with_attr)() for b in self.blocks()), count)    
        return batched(self.blocks(), count)

    #----------------------------------------------------------------------------------------------
    def as_binary(self, outfile, stop:int=None, buffer=100, rename=(),
                  progress=lambda x:None, cancel=lambda:None):                           # fmt_file
    #----------------------------------------------------------------------------------------------
        """Return the block encoded as binary data."""
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



#==================================================================================================
class FUNRST_file(fmt_file):                                                          # FUNRST_file
#==================================================================================================
    """
    FUNRST (Formatted Unified Restart File)
    ASCII text equivalent of the UNRST binary file. Contains solution data arrays for all active 
    cells at each report step, including pressures, saturations, and other simulation variables. 
    Each record begins with a keyword line followed by numeric data in human-readable 
    scientific notation. Useful for inspection, testing, or data exchange when binary files are 
    not desired.
    """

    #----------------------------------------------------------------------------------------------
    def __init__(self, filename):                                                     # FUNRST_file
    #----------------------------------------------------------------------------------------------
        """Initialize the FUNRST_file."""
        super().__init__(filename, suffix='.FUNRST')
        self.start = 'SEQNUM'

    #----------------------------------------------------------------------------------------------
    def data(self, *keys):                                                            # FUNRST_file
    #----------------------------------------------------------------------------------------------
        """Return decoded block data."""
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

    #----------------------------------------------------------------------------------------------
    def as_unrst(self, outfile=None, **kwargs):                                       # FUNRST_file
    #----------------------------------------------------------------------------------------------
        """Return the block mapped to unified UNRST data."""
        outfile = Path(outfile) if outfile else self.path
        outfile = outfile.with_suffix('.UNRST')
        return UNRST_file( super().as_binary(outfile, **kwargs) )



#==================================================================================================
class RSM_block:                                                                        # RSM_block
#==================================================================================================
    """Representation of a single RSM block."""

    #----------------------------------------------------------------------------------------------
    def __init__(self, var, unit, well, data):                                          # RSM_block
    #----------------------------------------------------------------------------------------------
        """Initialize the RSM_block."""
        self.var = var
        self.unit = unit
        self.well = well
        self.data = data
        self.nrow = len(self.data)
        
    #----------------------------------------------------------------------------------------------
    def get_data(self):                                                                 # RSM_block
    #----------------------------------------------------------------------------------------------
        """Return raw data for the key."""
        for col,(v,u,w) in enumerate(zip(self.var, self.unit, self.well)):
            yield (v, u, w, [self.data[row][col] for row in range(self.nrow)])


#==================================================================================================
class RSM_file(File):                                                                    # RSM_file
#==================================================================================================
    """
    RSM (Run Summary File)
    Formatted ASCII summary file containing time-series output vectors for the run.
    Each row corresponds to a timestep or report step, and each column represents 
    a simulation variable such as field oil production rate, cumulative production, 
    or well pressure. Easily readable in spreadsheet or plotting software; often 
    used for quick visualization and post-processing of performance data.
    """

    #----------------------------------------------------------------------------------------------
    def __init__(self, filename, **kwargs):                                              # RSM_file
    #----------------------------------------------------------------------------------------------
        """Initialize the RSM_file."""
        #self.file = Path(filename)
        super().__init__(filename, **kwargs)
        self.fh = None
        self.tag = '1'
        self.nrow = self.block_length()-10
        self.colpos = None
        
    #----------------------------------------------------------------------------------------------
    def get_data(self):                                                                  # RSM_file
    #----------------------------------------------------------------------------------------------
        """Return raw data for the key."""
        if not self.path.is_file():
            return ()
        with open(self.path, 'r', encoding='utf-8') as self.fh:
            for line in self.fh:
                # line is now at the tag-line
                for block in self.read_block():
                    for data in block.get_data():
                        yield data
                            
    #----------------------------------------------------------------------------------------------
    def read_block(self):                                                                # RSM_file
    #----------------------------------------------------------------------------------------------
        """Read the payload for the next block."""
        self.skip_lines(3)
        var, unit, well = self.read_var_unit_well()
        self.skip_lines(2)
        data = self.read_data(ncol=len(var))
        yield RSM_block(var, unit, well, data)
                
    #----------------------------------------------------------------------------------------------
    def skip_lines(self, n):                                                             # RSM_file
    #----------------------------------------------------------------------------------------------
        """Skip the requested number of lines."""
        next(islice(self.fh, n, n), None)
        # for i in range(n):
        #     next(self.fh)
        
    #----------------------------------------------------------------------------------------------
    def read_data(self, ncol=None):                                                      # RSM_file
    #----------------------------------------------------------------------------------------------
        """Read raw block data."""
        data = [None]*self.nrow        
        for l in range(self.nrow):
            line = next(self.fh)
            cols = line.rstrip().split()
            if len(cols)<ncol:
                # missing column entries
                cols = self.get_columns_by_position(line=line)
            data[l] = [float_or_str(c) for c in cols]
        return data

    
    #----------------------------------------------------------------------------------------------
    def get_columns_by_position(self, line=None):                                        # RSM_file
    #----------------------------------------------------------------------------------------------
        """
        Return None if column is empty
        """
        n = len(self.colpos)
        words = [None]*n
        for i in range(n-1):
            a, b = self.colpos[i], self.colpos[i+1]
            words[i] = line[a:b].strip() or None
        return words

    
    #----------------------------------------------------------------------------------------------
    def read_var_unit_well(self):                                                        # RSM_file
    #----------------------------------------------------------------------------------------------
        """Read well unit assignments."""
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

    #----------------------------------------------------------------------------------------------
    def block_length(self):                                                              # RSM_file
    #----------------------------------------------------------------------------------------------
        """Return the length of the current block."""
        with open(self.path, 'r', encoding='utf-8') as fh:
            nb, n = 0, 0
            for line in fh:
                n += 1
                if line[0]==self.tag:
                    nb += 1
                    if nb==2:
                        return int(n)
