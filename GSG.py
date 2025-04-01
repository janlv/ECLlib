
# -*- coding: utf-8 -*-

from collections import defaultdict, namedtuple
from itertools import repeat, zip_longest
from pathlib import Path
from functools import partial, wraps
from math import prod
from re import compile as re_compile
from struct import pack, unpack, calcsize
from os import SEEK_CUR, SEEK_END
from numpy import array, concatenate, full

from IORlib.utils import batched_as_list, flatten, list_prec, take, run_length_encode

Dtype = namedtuple('dtype', 'format size value')
DTYPE = {0:Dtype('i', 4, 0), 1:Dtype('f', 4, 1), 2:Dtype('d', 8, 2)}
format2type = {v.format:k for k,v in DTYPE.items()}

# PROP-data
#    type: A single character that identify the data
#           Possible values seen in GSG-files: p, g, r, ?
#    name: The full variablename of the data
#    alias: Short version of the full name
PROP_data = namedtuple('PROP_data', 'type name alias array kind',
                        defaults=('', '', '', array([]), 0))

#--------------------------------------------------------------------------------
def read_prop_file(file, dim=None, raise_error=False):
#--------------------------------------------------------------------------------
    # Read data from GSG-file with PROP data.
    with File(file) as gsg:
        # Two ints preceeds the start of the data.
        # The first int is 0 for a full array, and 1 if the values are 
        # grouped (run-lenght encoded). The second int is a counter of the 
        # data-values in the GSG-file
        for var, name_typ, start, end in gsg.data_positions():
            if var != 'PROP':
                if raise_error:
                    raise ValueError(f'Expected PROP, got {var}')
                return PROP_data()
            name, typ = zip(*name_typ)
            gsg.goto(start)
            # The first two ints are the kind and the number of the data
            kind, _ = gsg.read_int(2)
            # Read alias and (b'ca  ', datatype, arr.size)
            alias, keydata = gsg.read_keyword('4sqi')
            dtype = DTYPE[keydata[0][1]]
            nbytes = end - gsg.current_pos()
            size = nbytes//dtype.size
            #print(size, nbytes/dtype.size, (size-1)//2)
            fmt = str(size) + dtype.format
            if kind == 1:
                if dtype.format == 'f':
                    fmt = 'i' + ''.join(repeat('if', (size-1)//2))
                data = gsg.read_unpack(fmt)
                data = concatenate([full(count, value) for count, value in zip(data[1::2], data[2::2])])
            else:
                data = array(gsg.read_unpack(fmt))
            if dim:
                data = data.reshape(dim, order='F')
            yield PROP_data(type=typ, name=name, alias=alias, array=data, kind=kind )


#--------------------------------------------------------------------------------
def write_prop_file(file, *prop_data:PROP_data):
#--------------------------------------------------------------------------------
    with File(file, write=True) as gsg:
        gsg.write_header('PetrelForIx', '2022.9.0')
        kind_pos = []
        for i, data in enumerate(prop_data):
            kind_pos.append((data.kind, gsg.current_pos() + 12))
            gsg.write_keyword('PROP', '2i', (data.kind, i+1))
            arr = data.array
            dataformat = format2type[arr.dtype.kind]
            gsg.write_keyword(data.alias, '4sqi', (b'ca  ', dataformat, arr.size))
            if arr.ndim > 1:
               arr = arr.flatten(order='F')
            if data.kind == 1:
                # Write grouped data
                rle_arr = [1] + run_length_encode(arr)
                fmt = str(len(rle_arr)) + 'i'
                if dataformat == 1:
                    # float
                    fmt = 'i' + ''.join(repeat('if', (len(rle_arr)-1)//2))
                gsg.write_pack(fmt, rle_arr)
            else:
                # Write full array
                gsg.write(arr.tobytes())
        # Meta data
        # CASE_PROPS
        case_prop_pos = gsg.current_pos() + 18
        gsg.write_keyword('CASE_PROPS', 'i4s2i', (0, b's   ', 0, len(prop_data)))
        # Variable names
        for i, data in enumerate(prop_data):
            pre = ('8si', (b'ca  s   ', 1))
            type_ = data.type.ljust(4, ' ').encode('utf-8')
            gsg.write_keyword(data.name, '4s2i', (type_, 1, i+1), pre=pre)
        # INDEX
        index_pos = gsg.current_pos()
        gsg.write_keyword('INDEX', '2i', (0, len(prop_data) + 1))
        # PROP
        for kind, pos in kind_pos:
            gsg.write_keyword('PROP', 'iq', (kind, pos))
        # CASE_PROPS
        gsg.write_keyword('CASE_PROPS', 'iqq', (0, case_prop_pos, index_pos))




#--------------------------------------------------------------------------------
def change_resolution(dim, rundir):
#--------------------------------------------------------------------------------
    rundir = Path(rundir)
    backup = rundir/'GSG_backup'
    print(f'Changing resolution to {dim} for GSG-files in {rundir}')
    backup.mkdir(exist_ok=True)
    for gsg in rundir.glob('*.GSG'):
        #print()
        #print(gsg)
        data = Data().from_file(gsg)
        #print(data)
        backup_path = backup/gsg.name
        print(f'Moving {gsg} -> {backup_path}')
        gsg.rename(backup_path)
        if data.is_grid():
            data.new_grid(dim)
        else:
            data.set_values(dim)
        data.to_file(gsg)
        # Test if new file is readable
        Data().from_file(gsg)
        #print(testfile)
        #print(test)


#------------------------------------------------------------------------------------
class File:
#------------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------
    def __init__(self, filepath, write=False):
    #--------------------------------------------------------------------------------
        self.filepath = Path(filepath) #.with_suffix('.GSG')
        self.file_obj = None
        self.mode = 'rb'
        if write:
            self.mode = 'wb+'

        data = (
            ('PROP',       '2i',        8),
            ('data',       '4sqi',     16 , self.read_data    , self.write_data),
            ('CASE_PROPS', 'i4s2i8si', 28),
            ('varname',    '4s2i',     12 , self.read_varnames, self.write_varnames))

        grid = (
            ('AXES',          'i12s2i6f', 48),
            ('GRID',          '2i',        8),
            ('case',          '4i',       16, self.read_case   , self.write_case),
            ('AREAL',         '5i',       20, self.read_areal  , self.write_areal),
            ('PILLARS',       '17i',      68, self.read_pillars, self.write_pillars),
            ('PROP',          '2i',        8),
            ('DEFINED_CELLS', '4s6i',     28),
            ('CASE_PROPS',    'i4si8si',  24),
            ('DEFINED_CELLS', '4s3i',     16))

        def get_readers(x):
            return tuple((b, c, d[0] if d else self.read_keyword) for _,b,c,*d in x)
        def get_writers(x):
            return tuple((b, d[1] if d else self.write_keyword) for _,b,_,*d in x)

        self.readers = {'PROP':get_readers(data), 'AXES':get_readers(grid)}
        self.writers = {'PROP':get_writers(data), 'AXES':get_writers(grid)}

    #--------------------------------------------------------------------------------
    def __str__(self):
    #--------------------------------------------------------------------------------
        return f'{self.filepath}'

    #--------------------------------------------------------------------------------
    def __enter__(self):
    #--------------------------------------------------------------------------------
        self.file_obj = open(self.filepath, self.mode)
        return self

    #--------------------------------------------------------------------------------
    def __exit__(self, exc_type, exc_val, exc_tb):
    #--------------------------------------------------------------------------------
        if self.file_obj:
            self.file_obj.close()

    #--------------------------------------------------------------------------------
    def data_positions(self):
    #--------------------------------------------------------------------------------
        meta = self.read_meta()
        nvar = meta['INDEX'][1]-1
        self.goto(meta['CASE_PROPS'][1] - 18)
        key, case_props = self.read_keyword('i4s2i') #, 28)
        try:
            num = case_props[0][3]
            kwdata = list(self.read_keyword('4s2i', pre='8si') for _ in range(num))
            name_type = defaultdict(list)
            for name, data in kwdata:
                num, typ = data[1][2], data[1][0].strip().decode('utf-8')
                name_type[num].append((name, typ))
            name_type = list(name_type.values())
        except ValueError:
            # For AXES files, var-names is not formatted as for PROP files
            name_type = nvar * [('', '')]
        var = [m[:4] if m[:4] == 'PROP' else m for m in meta][1:]
        # Subtract 4 bytes (int) to align startpos just after the 'var' keyword (e.g 'PROP')
        startpos = [pos[1] - 4 for pos in meta.values()][1:]
        endpos = [s - len(v) - 4 for v,s in zip(var, startpos)]
        return list(zip(var, name_type, startpos[:-1], endpos[1:]))

    #--------------------------------------------------------------------------------
    def blocks(self, file_format=None):
    #--------------------------------------------------------------------------------
        file_format = file_format or self.read_format()
        for fmt, size, reader in self.readers[file_format]:
            result = reader(fmt, size)
            if isinstance(result[0], str):
                yield result
            else:
                for key, value in result:
                    yield (key, value)
 

    #--------------------------------------------------------------------------------
    def block_writers(self, datadict:dict, _format:str):
    #--------------------------------------------------------------------------------
        nvar = 0
        writers = self.writers[_format]
        for (fmt, writer), (key, data) in zip_longest(writers, datadict.items(), fillvalue=writers[-1]):
            if nvar > 1:
                fmt += '8si'
                nvar -= 1
            yield (writer, key, fmt, data)
            if _format == 'PROP' and key == 'CASE_PROPS':
                nvar = data[0][3]
                # CASE_PROPS : [[0, b's   ', 0, 1, b'ca  s   ', 1]]
                #                        nvar --^

    #--------------------------------------------------------------------------------
    def write_pack(self, fmt:str, data, **kwargs):
    #--------------------------------------------------------------------------------
        self.write(pack('<'+fmt, *data, **kwargs))

    #--------------------------------------------------------------------------------
    def read_unpack(self, fmt:str, *args, **kwargs):
    #--------------------------------------------------------------------------------
        return unpack('<'+fmt, self.read(calcsize('='+fmt), *args, **kwargs))

    #--------------------------------------------------------------------------------
    def read(self, *args, **kwargs):
    #--------------------------------------------------------------------------------
        return self.file_obj.read(*args, **kwargs)

    #--------------------------------------------------------------------------------
    def read_keyword(self, fmt:str, pre=None):
    #--------------------------------------------------------------------------------
        data = []
        if pre:
            data.append(self.read_unpack(pre))
        key_size = self.read_unpack('i')[0]
        if key_size > 1000:
            self.skip(-4)
            raise ValueError(f'Wrong size of keyword ({key_size}) at {self.current_pos()}: {self.read(14)}')
        try:
            key = self.read(key_size).decode('utf-8')
        except UnicodeDecodeError as err:
            self.skip(-key_size)
            raise UnicodeDecodeError(f'Unable to read keyword of size {key_size}: {self.read(key_size)}') from err
        data.append(self.read_unpack(fmt))
        #data = unpack('<'+fmt, self.read(calcsize(fmt)))
        return (key, data)

    #--------------------------------------------------------------------------------
    def read_format(self):
    #--------------------------------------------------------------------------------
        """
        Read and return first keyword (PROP or AXES) to identify type of GSG-file 
        """
        try:
            keyword, _ = self.read_keyword('')
        except (UnicodeDecodeError, ValueError):
            # Try to correct the file position and read again
            self.read_header()
            keyword, _ = self.read_keyword('')
        self.skip(-(4+len(keyword)))
        return keyword


    #--------------------------------------------------------------------------------
    def read_data(self, *args):
    #--------------------------------------------------------------------------------
        # If PROP is [1,1] only 3 data-values are given
        # Rewind 8 bytes and read the PROP-values
        self.skip(-8)
        a, _ = self.read_int(2)
        key, data = self.read_keyword(*args)
        head = data[0]
        dtype = DTYPE[head[1]]
        len_data = head[2]
        fmt = f'{len_data}{dtype.format}'
        if a == 1:
            len_data = 3
            fmt = f'2i{dtype.format}'
        #print(fmt, dtype.size, len_data, dtype.size*len_data)
        data = list(unpack(fmt, self.read(dtype.size*len_data)))
        return (key, [head, data])

    #--------------------------------------------------------------------------------
    def write_data(self, key:str, fmt:str, data2d:list):
    #--------------------------------------------------------------------------------
        # If PROP is [1,1] only 3 data-values are given
        # Rewind 8 bytes and read the PROP-values
        self.skip(-8)
        a, _ = self.read_int(2)
        head, data = data2d
        dtype = DTYPE[head[1]]
        #head[2] = len(data)
        self.write_keyword(key, fmt, head)
        fmt = f'{len(data)}{dtype.format}'
        if a > 0:
            fmt = f'2i{dtype.format}'
        self.write(pack(fmt, *data))

    #--------------------------------------------------------------------------------
    def read_varnames(self, *args):
    #--------------------------------------------------------------------------------
        # It seems that the header-data after the keyword is different if more
        # than one variable is read. For 3 keywords, only the last keyword follow
        # the usual formatting. Hence, we need to change formatting for the two first
        # keywords. This seem to only apply to the _REGIONS.GSG files
        self.skip(-16)
        nvar = self.read_int()[0]
        self.skip(12)
        fmt, size = args
        # Make nvar copies of fmt and size as lists
        fmt, size = [list(par) for par in zip(*(nvar*[args]))]
        if nvar > 1:
            for i in range(2):
                fmt[i] += '8si'
                size[i] += 12
        return [self.read_keyword(fmt[i]) for i in range(nvar)]
        #return [self.read_keyword(fmt[i], size[i]) for i in range(nvar)]

    #--------------------------------------------------------------------------------
    def write_varnames(self, *args):
    #--------------------------------------------------------------------------------
        self.write_keyword(*args)

    #--------------------------------------------------------------------------------
    def read_case(self, *args):
    #--------------------------------------------------------------------------------
        key, data = self.read_keyword(*args)
        head = data[0]
        chars = []
        while (n := self.read_int()[0]) > 0:
            chars.extend((n, self.read(4*n)))
        return (key, [head + chars + [n]])

    #--------------------------------------------------------------------------------
    def write_case(self, key:str, fmt:str, data:list):
    #--------------------------------------------------------------------------------
        # Example data: [-1, 20, 1, 1, 1, b'scor', 1, b'sp  ', 1, b'vp  ', 0]
        data = data[0]
        fmt += f'i{4*data[4]}si{4*data[6]}si{4*data[8]}si'
        self.write_keyword(key, fmt, data)

    #--------------------------------------------------------------------------------
    def read_areal(self, *args, len_areal=6):
    #--------------------------------------------------------------------------------
        key, data = self.read_keyword(*args)
        head = data[0]
        num_areal = head[3]
        data = list(batched_as_list(self.read_int(len_areal*num_areal), len_areal))
        return (key, [head, data])

    #--------------------------------------------------------------------------------
    def write_areal(self, key:str, fmt:str, data2d:list):
    #--------------------------------------------------------------------------------
        head, data = data2d
        # nxny = len(data)
        # nx = data[0][2]-1
        # ny = nxny//nx
        # head[3:3+2] = (nxny, nx+ny+1)
        self.write_keyword(key, fmt, head)
        self.write_int(list(flatten(data)))

    #--------------------------------------------------------------------------------
    def read_pillars(self, *args):
    #--------------------------------------------------------------------------------
        key, data = self.read_keyword(*args)
        head = data[0]
        num_pillars, len_pillar, grid_type = head[2], head[3]+5, head[14]
        # Check if grid is not complex
        if grid_type != 0:
            print('Unable to read PILLARS for this type of grid, aborting...')
            return None
            # raise NotImplementedError('Unable to read PILLARS for this type of grid')
        bin_data = (self.read(4*len_pillar) for _ in range(num_pillars))
        data =  [list(unpack(f'<i{len_pillar-1}f', data)) for data in bin_data]
        return (key, [head, data])

    #--------------------------------------------------------------------------------
    def write_pillars(self, key:str, fmt:str, data2d:list):
    #--------------------------------------------------------------------------------
        head, data = data2d
        len_pillar = len(data[0])
        # num_pillars = len(data)
        # head[2], head[3] = num_pillars, len_pillar - 5
        self.write_keyword(key, fmt, head)
        fmt = f'<i{len_pillar-1}f'
        for val in data:
            self.write(pack(fmt, *val))

    #--------------------------------------------------------------------------------
    def write_header(self, creator, version):
    #--------------------------------------------------------------------------------
        header = (1, 1, len(creator), creator.encode(), len(version), version.encode(), 0, 0, 0, 1)
        packed_header = pack(f'<3i{len(creator)}si{len(version)}s4i', *header)
        self.write(b'GSG000_b\r\n1\n2\r34\x01\x02\x03\x04' + packed_header)

    #--------------------------------------------------------------------------------
    def read_header(self):
    #--------------------------------------------------------------------------------
        self.goto(28)
        creator, _ = self.read_keyword('')
        version, _ = self.read_keyword('4i')
        return (creator, version)

    #--------------------------------------------------------------------------------
    def read_meta(self, search=False):
    #-------------------------------------------------------------------------------
        """
        Read meta-keys at the end of the file. These keys are data-key positions taken 
        one int (4 bytes) away from the end of the keyword string
        """
        meta = {}
        if search:
            _, pos = next(self.keyword_positions('INDEX'))
            self.goto(pos-9)
        else:
            # Last 8 bytes of the file is the position of the INDEX-key
            self.end(-8)
            index_pos, = self.read_uint()
            self.goto(index_pos)
        #print(self.read(10))
        #self.skip(-10)
        index, data = self.read_keyword('2i')
        #print(index, data)
        meta[index] = data[0]
        for i in range(meta['INDEX'][1]):
            key, data = self.read_keyword('iq')
            #print(key, data)
            if key == 'PROP':
                key = f'PROP_{i}'
            meta[key] = data[0]
        meta[key] += self.read_uint()
        return meta

    #--------------------------------------------------------------------------------
    def write_meta(self, meta:dict):
    #-------------------------------------------------------------------------------
        """
        Write meta-keys at the end of the GSG-file with file-positions of the data-keys.
        The first (int) value in the meta-key refer to the first value of the data-key.
        """
        end_data = self.current_pos()
        # INDEX refers to the number of meta-keys
        self.write_keyword('INDEX', '2i', meta.pop('INDEX'))
        # Iterate over meta-key positions (after keyword)  
        keys = list(meta.keys())
        for key, data_pos in take(len(keys), self.keyword_positions(*keys)):
            # Save current position for writing
            write_pos = self.current_pos()
            # Read first value of the data-key
            self.goto(data_pos)
            ival = self.read_int()[0]
            # Jump to write position
            self.goto(write_pos)
            self.write_keyword(key, 'iq', (ival, data_pos+4))
        self.write_uint([end_data])

    #--------------------------------------------------------------------------------
    def read_int(self, length=1):
    #--------------------------------------------------------------------------------
        return unpack(f'<{length}i', self.read(4*length))

    #--------------------------------------------------------------------------------
    def read_uint(self, length=1):
    #--------------------------------------------------------------------------------
        return unpack(f'<{length}q', self.read(8*length))

    #--------------------------------------------------------------------------------
    def write_fmt(self, fmt, data, length=None):
    #--------------------------------------------------------------------------------
        length = length or len(data)
        return self.write(pack(f'<{length}{fmt}', *data))

    #--------------------------------------------------------------------------------
    def write_int(self, *args, **kwargs):
    #--------------------------------------------------------------------------------
        return self.write_fmt('i', *args, **kwargs)

    #--------------------------------------------------------------------------------
    def write_uint(self, *args, **kwargs):
    #--------------------------------------------------------------------------------
        return self.write_fmt('q', *args, **kwargs)

    #--------------------------------------------------------------------------------
    def write_keyword(self, keyword:str, fmt:str, data:tuple, pre=None):
    #--------------------------------------------------------------------------------
        """
        Writes a keyword and its associated data to a binary format.
        Args:
            keyword (str): The keyword to be written.
            fmt (str): The format string specifying the binary structure of the data.
            data (tuple): The data to be written, which can be a tuple or a 2D list.
            pre (tuple, optional): Additional data to be written before the main data. Defaults to None.
        Returns:
            int: The number of bytes written to the binary stream.
        """
        
        # Reduce to single list if a 2D list is passed
        if isinstance(data[0], list):
            data = data[0]
        if pre:
            self.write_pack(*pre)
        return self.write_pack(f'i{len(keyword)}s'+fmt, (len(keyword), keyword.encode(), *data))


    #--------------------------------------------------------------------------------
    def keyword_positions(self, *keywords):
    #--------------------------------------------------------------------------------
        chunk_size = 1024
        pattern = re_compile(rb'(' + rb'|'.join(k.encode() for k in keywords) + rb')')
        current_pos = self.current_pos()
        self.goto(0)
        try:
            for i,chunk in enumerate(iter(partial(self.read, chunk_size), b'')):
                for match in pattern.finditer(chunk):
                    pos = match.end() + i*chunk_size
                    yield (match.group().decode('utf-8'), pos)
        finally:
            self.goto(current_pos)

    #--------------------------------------------------------------------------------
    def skip(self, pos):
    #--------------------------------------------------------------------------------
        self.file_obj.seek(pos, SEEK_CUR)

    #--------------------------------------------------------------------------------
    def end(self, pos):
    #--------------------------------------------------------------------------------
        self.file_obj.seek(pos, SEEK_END)

    #--------------------------------------------------------------------------------
    def goto(self, pos):
    #--------------------------------------------------------------------------------
        self.file_obj.seek(pos)

    #--------------------------------------------------------------------------------
    def current_pos(self):
    #--------------------------------------------------------------------------------
        return self.file_obj.tell()

    #--------------------------------------------------------------------------------
    def size(self):
    #--------------------------------------------------------------------------------
        return self.filepath.stat().st_size

    #--------------------------------------------------------------------------------
    def write(self, *args, **kwargs):
    #--------------------------------------------------------------------------------
        return self.file_obj.write(*args, **kwargs)

    # #--------------------------------------------------------------------------------
    # def copy(self, pos1, pos2):
    # #--------------------------------------------------------------------------------
    #     self.goto(pos1)
    #     return self.read(pos2-pos1)


#------------------------------------------------------------------------------------
def grid_function(method):
#------------------------------------------------------------------------------------
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not self.is_grid():
            raise ValueError("Operation only allowed for grid-objects")
        return method(self, *args, **kwargs)
    return wrapper

#------------------------------------------------------------------------------------
def data_function(method):
#------------------------------------------------------------------------------------
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.is_grid():
            raise ValueError("Operation only allowed for data-objects")
        return method(self, *args, **kwargs)
    return wrapper

#------------------------------------------------------------------------------------
class Data():
#------------------------------------------------------------------------------------
    Grid = namedtuple('Grid', 'name dim size res num_areal num_pillars')

    #--------------------------------------------------------------------------------
    def __init__(self):
    #--------------------------------------------------------------------------------
        self.data = {}
        self.meta = {}
        self.creator = ''
        self.version = ''
        self.format = ''
        self.grid = None

    #--------------------------------------------------------------------------------
    def __str__(self):
    #--------------------------------------------------------------------------------
        if not self.data:
            return 'No data to show'
        marg = 1 + max(map(len, self.data.keys()))
        lines = []
        for key, val in self.data.items():
            valstr = str(val[0])
            if len(val) > 1:
                if len(val[1]) > 10:
                    valstr += f'{val[1][:3]} ... {val[1][-3:]}'
                else:
                    valstr += str(val[1])
            lines.append(f'{key:{marg}s}: ' + valstr)
        return '\n'.join(lines)

    #--------------------------------------------------------------------------------
    def __getitem__(self, key):
    #--------------------------------------------------------------------------------
        # Return value of the given key in self.data
        return self.data.get(key, None)

    #--------------------------------------------------------------------------------
    def __setitem__(self, key, value):
    #--------------------------------------------------------------------------------
        # Make sure keyword is unique
        #key = unique_key(key, list(self.data.keys()))
        self.data[key][1] = value

    #--------------------------------------------------------------------------------
    def is_grid(self):
    #--------------------------------------------------------------------------------
        if self.format == 'AXES':
            return True
        return False

    @data_function
    #--------------------------------------------------------------------------------
    def is_indexed(self):
    #--------------------------------------------------------------------------------
        # Data is given by a range of two ints and a common float/int value
        if self.data['PROP'][0][0] == 1:
            return True
        return False

    @data_function
    #--------------------------------------------------------------------------------
    def set_values(self, dim):
    #--------------------------------------------------------------------------------
        head, data = self.data[self.keys()[1]]
        head[2] = dim[0]*dim[1]
        if self.is_indexed():
            data[1] = head[2]
        else:
            data[:] = head[2]*[data[0]]


    @data_function
    #--------------------------------------------------------------------------------
    def num_variables(self):
    #--------------------------------------------------------------------------------
        return self.data['CASE_PROPS'][0][3]
        # CASE_PROPS : [[0, b's   ', 0, 1, b'ca  s   ', 1]]
        #                        nvar --^


    #--------------------------------------------------------------------------------
    def keys(self):
    #--------------------------------------------------------------------------------
        return tuple(self.data.keys())

    #--------------------------------------------------------------------------------
    def unique_key(self, key):
    #--------------------------------------------------------------------------------
        if (count := self.keys().count(key)):
            key += f'{'#'}{count}'
        return key
        #return unique_key(key, self.keys())

    #--------------------------------------------------------------------------------
    def from_file(self, filepath):
    #--------------------------------------------------------------------------------
        with File(filepath) as file:
            self.creator, self.version = file.read_header()
            self.format = file.read_format()
            # Read data blocks
            try:
                for keyword, data in file.blocks(self.format):
                    if not data:
                        return self
                    self.data[self.unique_key(keyword)] = data
            except ValueError as err:
                print(self)
                raise err
            # Read meta blocks with file positions
            self.meta = file.read_meta()
            # If this is a grid, set relevant variables
            if self.is_grid():
                self.set_grid_variables()
        return self

    #--------------------------------------------------------------------------------
    def to_file(self, filepath):
    #--------------------------------------------------------------------------------
        with File(filepath, write=True) as file:
            file.write_header(self.creator, self.version)
            for writer, key, fmt, data in file.block_writers(self.data, self.format):
                writer(key.split('#')[0], fmt, data)
            file.write_meta(self.meta)


    #--------------------------------------------------------------------------------
    def info(self):
    #--------------------------------------------------------------------------------
        if self.is_grid():
            self._grid_info()
        else:
            self._data_info()

    #--------------------------------------------------------------------------------
    def _data_info(self):
    #--------------------------------------------------------------------------------
        varname = self.keys()[1]
        print()
        print(f'GSG data: {varname}')
        print('-------------------------------------')
        print(f'Length : {len(self.data[varname])}' )

    @grid_function
    #--------------------------------------------------------------------------------
    def _grid_info(self):
    #--------------------------------------------------------------------------------
        fmt = '.4f'
        print()
        print(f'GSG grid: {self.grid.name}')
        print('-------------------------------------')
        print( 'Size       : ' + list_prec(self.grid.size, fmt))
        print(f'Dimension  : {self.grid.dim}')
        print( 'Resolution : '  + list_prec(self.grid.res, fmt))
        print(f'Areal      : {self.grid.num_areal}')
        print(f'Pillars    : {self.grid.num_pillars}')

    @grid_function
    #--------------------------------------------------------------------------------
    def set_grid_variables(self):
    #--------------------------------------------------------------------------------
        name = self.keys()[2]
        dim = self.data[name][0][1:4]
        num_areal = self.data['AREAL'][0][3]
        num_pillars = self.data['PILLARS'][0][2]
        size = self.grid_size(dim=dim)
        res = self.grid_resolution(size=size, dim=dim)
        self.grid = self.Grid(name, dim, size, res, num_areal, num_pillars)

    @grid_function
    #--------------------------------------------------------------------------------
    def grid_size(self, dim=None):
    #--------------------------------------------------------------------------------
        if not self.is_grid():
            print('This is not a grid')
            return
        dim = dim or self.grid.dim
        pill = self.data['PILLARS'][1]
        origo = pill[0]
        lx = pill[ dim[0] ][1] - origo[1]
        ly = pill[ (dim[0]+1)*dim[1] ][2] - origo[2]
        lz = origo[-1] - origo[5]
        return (lx, ly, lz)

    @grid_function
    #--------------------------------------------------------------------------------
    def grid_resolution(self, size=None, dim=None):
    #--------------------------------------------------------------------------------
        size = size or self.grid_size()
        dim = dim or self.grid.dim
        return [l/n for l,n in zip(size, dim)]

    @grid_function
    #--------------------------------------------------------------------------------
    def set_areal(self, dim):
    #--------------------------------------------------------------------------------
        head, data = self.data['AREAL']
        #self.data['AREAL'][0][3:3+2] = (dim[0]*dim[1], dim[0]+dim[1]+1)
        head[3:3+2] = (dim[0]*dim[1], dim[0]+dim[1]+1)
        #areal = []
        data[:] = []
        n = 0
        for j in range(dim[1]):
            a = j*(dim[0]+1)
            b = a + (dim[0]+1)
            for i in range(dim[0]):
                #areal.append((n, 4, b+i,  a+i , a+i+1, b+i+1))
                data.append((n, 4, b+i,  a+i , a+i+1, b+i+1))
                n += 1
        #self.data['AREAL'][1] = areal

    @grid_function
    #--------------------------------------------------------------------------------
    def set_pillars_xy(self, dim):
    #--------------------------------------------------------------------------------
        # NBNB! Currently only the xy-pos of the pillars are updated
        # Get new grid resolution. NB! Need to do this before modifying the original pillars
        dl = self.grid_resolution(dim=dim)
        nx, ny = dim[0]+1, dim[1]+1
        npillars = nx*ny
        head, data = self.data['PILLARS']
        # Update header
        head[2] = npillars
        head[3] = dim[2] + 1
        #self.data['PILLARS'][0][2] = npillars 
        #self.data['PILLARS'][0][3] = dim[2] + 1
        # Update data
        #pill0 = self.data['PILLARS'][1][0]
        pill0 = data[0]
        #pillars = list(batched_as_list(npillars*pill0, len(pill0)))
        data[:] = list(batched_as_list(npillars*pill0, len(pill0)))
        for y in range(ny):
            for x in range(nx):
                pos = x + y*nx
                data[pos][1] = x * dl[0]
                data[pos][2] = y * dl[1]
        #         pillars[pos][1] = x * dl[0]
        #         pillars[pos][2] = y * dl[1]
        # self.data['PILLARS'][1] = pillars

    @grid_function
    #--------------------------------------------------------------------------------
    def new_grid(self, dim):
    #--------------------------------------------------------------------------------
        if not self.grid:
            print('This is not a grid')
            return
        # Update grid dimension
        self.data[self.grid.name][0][1:4] = dim
        self.data['DEFINED_CELLS'][0][3:3+4:2] = 2*[prod(dim)]
        # Areal
        self.set_areal(dim)
        # Pillars
        self.set_pillars_xy(dim)
        return self
    
