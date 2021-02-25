#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#from struct import unpack
import struct
import os
from pathlib import Path
from .utils import list2str, float_or_str
from numpy import zeros, int32, float32, float64, ceil
#from collections import namedtuple

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

max_block_length = 1000

endian = '>' # big-endian

unpack_char = {b'INTE' : 'i',
               b'REAL' : 'f',
               b'LOGI' : 'i',
               b'DOUB' : 'd',
               b'CHAR' : 's',
               b'MESS' : ' '}

#pack_char = {'INTE' : 'i',
#             'REAL' : 'f',
#             'LOGI' : 'i',
#             'DOUB' : 'd',
#             'CHAR' : 's',
#             'MESS' : ' '}

datasize = {b'INTE' : 4,
            b'REAL' : 4,
            b'LOGI' : 4,
            b'DOUB' : 8,
            b'CHAR' : 8,
            b'MESS' : 1}

datatype = {'INTE' : int32,
            'REAL' : float32,
            'LOGI' : bool,
            'DOUB' : float64,
            'CHAR' : str,
            'MESS' : str}

var2key = {'nwell':b'INTEHEAD'}
var2pos = {'nwell':16}

#....................................................................................
def encode(string):
#....................................................................................
    return ('%-8s'%string).encode()



#====================================================================================
class _datablock:                                                          # datablock
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self):                                                   # datablock
    #--------------------------------------------------------------------------------
        self.reset()
        
    #--------------------------------------------------------------------------------
    def set_header(self, chunk, filepos):                                 # datablock
    #--------------------------------------------------------------------------------
        self._key, self.length, self.datatype = struct.unpack(endian+'8si4s', chunk)
        # Set filepos to start of header.
        # The header is 4+16+4 = 24 bytes long
        self.startpos = filepos - 24 # for forward parsing!
        
    #--------------------------------------------------------------------------------
    def add_chunk(self, chunk):                                           # datablock
    #--------------------------------------------------------------------------------
        self.chunk += chunk
        #self.empty = False

    #--------------------------------------------------------------------------------
    #def get_data_array(self):                                             # datablock
    def data(self):                                             # datablock
    #--------------------------------------------------------------------------------
        #if self.chunk in (b'0',):
        if len(self.chunk)==0:
            return None
        if len(self.chunk)==1 and self.chunk[0]==0:
            raise SystemError('No data to unpack in data().\nDid you pass data=True to parse_file()?')
        if self.datatype in (b'CHAR',):
            self.length *= datasize[self.datatype]
        try:
            return struct.unpack(endian + str(self.length) + unpack_char[self.datatype], self.chunk)
        except struct.error as e:
            raise SystemError(e)
        
    #--------------------------------------------------------------------------------
    def reset(self):                                                      # datablock
    #--------------------------------------------------------------------------------
        #self.chunk = b''
        self.chunk = bytearray()
        self.length = None
        #self.endpos = self.startpos = None
        
    #--------------------------------------------------------------------------------
    def ready(self):                                                      # datablock
    #--------------------------------------------------------------------------------
        #if self.chunk != b'' or self.length == 0:
        if len(self.chunk)!=0 or self.length==0:
            return True
        else:
            return False

    #--------------------------------------------------------------------------------
    def info(self):                                                       # datablock
    #--------------------------------------------------------------------------------
        string = '{:s} block of {:5d} bytes holding {:5d} {} at [{}, {}]'
        return string.format(self._key.decode(), len(self.chunk), self.length,
                             self.datatype.decode(), self.startpos, self.end())

    #--------------------------------------------------------------------------------
    def print(self):                                                      # datablock
    #--------------------------------------------------------------------------------
        print(self.info())
        
    #--------------------------------------------------------------------------------
    def get_nwell(self):                                                  # datablock
    #--------------------------------------------------------------------------------
        return self.get_value('nwell')
        #try:
        #    return self.get_data_array()[16]
        #except:
        #    raise ValueError('Only INTEHEAD blocks contain NWELL, this is a {:s} block'.format(self._key.decode())) 

    #--------------------------------------------------------------------------------
    def get_value(self, var):                                             # datablock
    #--------------------------------------------------------------------------------
        #if self._key == b'INTEHEAD':
        #if self._key != self.var2key[var]:
        #    raise ValueError('ERROR! {} blocks does not hold {}'.format(self._key.decode(), var))
        #return self.get_data_array()[var2pos[var]]
        return self.data()[var2pos[var]]
        
    #--------------------------------------------------------------------------------
    def key(self):                                             # datablock
    #--------------------------------------------------------------------------------
        return self._key.decode().strip()
        
    #--------------------------------------------------------------------------------
    def start(self):                                             # datablock
    #-------------------------------------------------------------------------------
        return self.startpos
        
    #--------------------------------------------------------------------------------
    def end(self):                                             # datablock
    #-------------------------------------------------------------------------------
        # A data-block is split in chunks if the length exceeds 1000 elements
        # Each data-block is sandwiched by a 4 byte int giving the size of the block
        return self.startpos + 24 + len(self.chunk) + 8*int(ceil(self.length/max_block_length))
    
    #--------------------------------------------------------------------------------
    def datatype(self):                                             # datablock
    #-------------------------------------------------------------------------------
        return self.datatype.decode(),
        

        
#====================================================================================
#class reader:                                                                # reader
class unfmt_file:
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, filename):                                            # reader
    #--------------------------------------------------------------------------------
        self.fileobj = None
        self._filename = Path(filename)
        #self.data_chunk = b''
        self.endpos = self.startpos = 0

    #--------------------------------------------------------------------------------
    def set_startpos(self, pos):                                             # reader
    #--------------------------------------------------------------------------------
        self.startpos = pos
        
    #--------------------------------------------------------------------------------
    def set_endpos(self, pos):                                             # reader
    #--------------------------------------------------------------------------------
        self.endpos = pos
        
    #--------------------------------------------------------------------------------
    def filename(self):                                             # reader
    #--------------------------------------------------------------------------------
        return str(self._filename)
        
    #--------------------------------------------------------------------------------
    def name(self):                                             # reader
    #--------------------------------------------------------------------------------
        return str(self._filename.name)
        
    #--------------------------------------------------------------------------------
    #def parse_forward(self, data=False, datalist=[], startpos=None):   # reader
    def blocks(self, data=True, datalist=[], only_new=False):   # reader
    #--------------------------------------------------------------------------------
        if not Path(self._filename).is_file():
            return
        if datalist:
            data = False
        with open(self._filename, 'rb') as self.fileobj:
            if only_new:
                startpos = self.endpos
            else:
                startpos = self.startpos
            self.fileobj.seek(startpos, 1)
            db = _datablock()
            while True:
                try:
                    chunk = self._read_chunk(self._read_bytes_forward, self._safe_seek_forward)
                except EOFError:
                    self.endpos = self.fileobj.tell()
                    if db.ready():
                        yield db
                    return
                except MemoryError:
                    return False
                if self._chunk_is_header(chunk):
                    if db.ready():
                        yield db
                        db.reset()
                    db.set_header(chunk, self.get_filepos())
                else:
                    if not data and not db._key in datalist:
                        #chunk = b'0'
                        chunk = bytearray(1)
                    #db.add_chunk(chunk)
                    db.chunk += chunk

    #--------------------------------------------------------------------------------
    #def parse_backward(self, data=False):                               # reader
    def tail_blocks(self, data=False):                               # reader
    #--------------------------------------------------------------------------------
        #self.set_direction('backward')
        with open(self._filename, 'rb') as self.fileobj:
            #if startpos:
            #    self.fileobj.seek(startpos, 1)
            self.fileobj.seek(0, 2) # go to end of file
            db = _datablock()
            while True:
                try:
                    chunk = self._read_chunk(self._read_bytes_backward, self._safe_seek_backward)
                except EOFError:
                    self.endpos = self.fileobj.tell()
                    return
                if self._chunk_is_header(chunk):
                    db.set_header(chunk, self.get_filepos()+24)
                    if db.ready():
                        yield db
                        db.reset()
                else:
                    if not data:
                        #chunk = b'0'
                        chunk = bytearray(1)
                    #db.add_chunk(chunk)
                    db.chunk += chunk

                    
    #--------------------------------------------------------------------------------
    def get_filepos(self):                                                   # reader
    #--------------------------------------------------------------------------------
        return self.fileobj.tell() #/self.filesize
                    
    #--------------------------------------------------------------------------------
    def _read_bytes_forward(self, nbytes):                                   # reader
    #--------------------------------------------------------------------------------
        bytesread = self.fileobj.read(nbytes)
        if not bytesread:
            raise EOFError('End of {:s} reached!'.format(self.fileobj.name))
        return(bytesread)

    #--------------------------------------------------------------------------------
    def _read_bytes_backward(self, nbytes):                                  # reader
    #--------------------------------------------------------------------------------
        self._safe_seek_backward(nbytes)
        bytesread = self.fileobj.read(nbytes)
        if not bytesread:
            raise EOFError('End of {:s} reached!'.format(self.fileobj.name))
        self._safe_seek_backward(nbytes)
        return(bytesread)
    
    #--------------------------------------------------------------------------------
    def _safe_seek_backward(self, nbytes):                                   # reader
    #--------------------------------------------------------------------------------
        if self.fileobj.tell() >= nbytes:
            self.fileobj.seek(-nbytes, 1)
        else:
            raise EOFError('Cannot seek beyond start of file {:s}'.format(self.fileobj.name))

    #--------------------------------------------------------------------------------
    def _safe_seek_forward(self, nbytes):                                    # reader
    #--------------------------------------------------------------------------------
        #try:
        self.fileobj.seek(nbytes,1)
        #except EOFError:
        #    pass

    #--------------------------------------------------------------------------------
    def _chunk_is_header(self, chunk):                                       # reader
    #--------------------------------------------------------------------------------
        if len(chunk)==16:
            try:
                datasize[chunk[-4:]]
                return True
            except KeyError as e:
                return False
        else:
            return False
        
    #--------------------------------------------------------------------------------
    def _read_chunk(self, read_func, seek_func):                             # reader
    #--------------------------------------------------------------------------------
        size = struct.unpack(endian + 'i', read_func(4))[0] # read one int
        chunk = read_func(size)
        seek_func(4) # skip trailing/leading int
        return chunk

    # #--------------------------------------------------------------------------------
    # def get_blocks(self, direction='fwd', data=False):                  # reader
    # #--------------------------------------------------------------------------------
    #     if direction == 'fwd':
    #         parser = self.parse_forward
    #     else:
    #         parser = self.parse_backward
    #     for block in parser(data=data):
    #         yield block

    # #--------------------------------------------------------------------------------
    # def get_new_blocks(self, direction='fwd', data=False):              # reader
    # #--------------------------------------------------------------------------------
    #     if direction == 'fwd':
    #         parser = self.parse_forward
    #     else:
    #         parser = self.parse_backward
    #     for block in parser(data=data):
    #         yield block

    # #--------------------------------------------------------------------------------
    # def get_tail_block(self, data=False):                               # reader
    # #--------------------------------------------------------------------------------
    #     for block in self.parse_backward(data=data):
    #         return block

    #--------------------------------------------------------------------------------
    def tail_block_is(self, key):                                            # reader
    #--------------------------------------------------------------------------------
        try:
            key = key.upper().strip().ljust(8).encode()
            for block in self.tail_blocks(data=False):
                if block._key == key:
                    #block.print()
                    return True
                else:
                    return False
        except (AttributeError, ValueError):
            pass
        except:
            raise
        
    #--------------------------------------------------------------------------------
    def has_new_blocks(self):                                           # reader
    #--------------------------------------------------------------------------------
        if self._filename.stat().st_size > self.startpos:
            return True            

    #--------------------------------------------------------------------------------
    def exists(self):                                                   # reader
    #--------------------------------------------------------------------------------
        #print('Checking for ' + self._filename)
        if self._filename.is_file():
            return True

    #--------------------------------------------------------------------------------
    def create(self, *args):
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
        out_file = open(self._filename, 'wb')
        for sec in sections:
            sec.open_file()
        # Write sections to out_file
        OK = True
        while OK: 
            for sec in sections:
                OK = sec.write_next_unit(out_file)
        # Close files
        for sec in sections:
            sec.close_file()
        out_file.close()
        return self._filename
    

    #--------------------------------------------------------------------------------
    def remove(self, blocks=None):
    #--------------------------------------------------------------------------------
        if not isinstance(blocks, tuple):
            blocks = (blocks,)
        startpos, size= [0,], []
        for block in self.blocks():
            if block.key() in blocks:
                size.append(block.start()-startpos[-1])
                startpos.append(block.end())
        in_file = open(self._filename, 'rb')
        out = self._filename.parent/'removed.UNRST'
        out_file = open(out, 'wb')
        while (startpos and size):
            pos = startpos.pop(0)
            self_file.seek(pos)
            #print('pos: '+str(pos)+', '+str(self_file.tell()))
            read = size.pop(0)
            write=out_file.write(self_file.read(read))
            #print('read: '+str(read)+', '+str(write))
        in_file.close()
        out_file.close()
        return out
    
    
#====================================================================================
class Section:
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
    def open_file(self):
    #--------------------------------------------------------------------------------
        self._fh = open(self._filename, 'rb')
        
    #--------------------------------------------------------------------------------
    def close_file(self):
    #--------------------------------------------------------------------------------
        self._fh.close()
        
    #--------------------------------------------------------------------------------
    def filename(self):
    #--------------------------------------------------------------------------------
        return self._filename
        
    #--------------------------------------------------------------------------------
    def size(self):
    #--------------------------------------------------------------------------------
        return len(self._units)
        
    #--------------------------------------------------------------------------------
    def pop_unit(self, nr):
    #--------------------------------------------------------------------------------
        return self._units.pop(nr)
        
    #--------------------------------------------------------------------------------
    def write_next_unit(self, out_file):
    #--------------------------------------------------------------------------------
        if len(self._units)==0:
            return False
        unit = self._units.pop(0)
        for pos, size in zip(unit[::2],unit[1::2]):
            self._fh.seek(pos)
            out_file.write( self._fh.read(size) )
        return True
    
    #--------------------------------------------------------------------------------
    def print(self):
    #--------------------------------------------------------------------------------
        for unit in self._units:
            print(unit)

                             
    #--------------------------------------------------------------------------------
    def startpos_and_size(self):
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
class check_blocks:                                                # output_checker
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, filename, start=None, end=None, var=None):           # output_checker
    #--------------------------------------------------------------------------------
        #self.reader = reader(file)
        self.file = unfmt_file(filename)
        self.key = {'start':encode(start), 'end':encode(end)}
        self._var = var
        self.out = {k:[] for k in ('start', 'end', 'startpos')+(self._var,)}
        self.out['end'] = 0
        self.datalist = [self.key['start']]
        if var:
            self.datalist.append(var2key[var])
        self.last_block = None
    
    #--------------------------------------------------------------------------------
    def var(self, _var):                                             # output_checker
    #--------------------------------------------------------------------------------
        return self.out[_var][0]
            
    #--------------------------------------------------------------------------------
    def filename(self):                                              # output_checker
    #--------------------------------------------------------------------------------
        return self.file.filename

    #--------------------------------------------------------------------------------
    def start_values(self):                                          # output_checker
    #--------------------------------------------------------------------------------
        return {'start':self.key['start'].decode(), 'values':self.out['start']}

    
    #--------------------------------------------------------------------------------
    def info(self):                                                          # output_checker
    #--------------------------------------------------------------------------------
        txt = '   {} : {}'.format(self.key['start'].decode(), list2str(self.out['start']))
        if self._var:
            txt += ', {} : {}'.format(self._var, list2str(self.out[self._var]))
        return txt

    #--------------------------------------------------------------------------------
    def update_startpos(self, nblocks):        # output_checker
    #--------------------------------------------------------------------------------
        startpos = self.file.endpos
        # if > nblocks are read, set startpos to end of nblock timestep
        if len(self.out['start']) > nblocks: 
            startpos = self.out['startpos'][nblocks] 
        # update reader with startpos for next run 
        self.file.set_startpos(startpos)

        
    #--------------------------------------------------------------------------------
    def blocks_complete(self, nblocks=None): # output_checker
    #--------------------------------------------------------------------------------
        self.reset_out()
        try:
            #b = None
            #startpos = None
            #if self.last_block:
            #    startpos = self.last_block.startpos
            #for b in self.reader.parse_forward(data=False, datalist=self.datalist, startpos=startpos):
            for b in self.file.blocks(data=False, datalist=self.datalist):
                if b._key == self.key['start']:
                    #b.print()
                    #self.out['start'].append(b.get_data_array()[0])
                    self.out['start'].append(b.data()[0])
                    self.out['startpos'].append(b.startpos)
                if b._key == self.key['end']:
                    #b.print()
                    self.out['end'] += 1 
                if self._var and b._key == var2key[self._var]:
                    self.out[self._var].append(b.get_value(self._var))
            #print(self.out['end'])
        except (ValueError, AttributeError):
            #if b:
            #    self.last_block = b
            return None
        except:
            raise
        try:
            if b._key == self.key['end'] and len(self.out['start']) >= nblocks and len(self.out['start']) == self.out['end']:
                self.update_startpos(nblocks)
                return True
        except UnboundLocalError:
            #if b:
            #    self.last_block = b
            return None

        
    #--------------------------------------------------------------------------------
    def reset_out(self):                                             # output_checker
    #--------------------------------------------------------------------------------
        self.out = {k:[] for k in self.out.keys()}
        self.out['end'] = 0


#====================================================================================
class fmt_block:
    #
    # formatted block of Eclipse data
    #
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, keyword=None, length=None, datatype=None, data=zeros(0)):
    #--------------------------------------------------------------------------------
        self.keyword = keyword
        self.length = length
        self.datatype = datatype
        self.data = data
        self.max_length = max_block_length
        #self.max_size = 1000*datasize[self.datatype]
        #self.max_size = 4000 #2**31

    # #--------------------------------------------------------------------------------
    # def update(self, keyword=None, length=None, datatype=None, data=zeros(0)):
    # #--------------------------------------------------------------------------------
    #     self.keyword = keyword
    #     self.length = length
    #     self.datatype = datatype
    #     self.data = data
    #     self.max_length = max_block_length
        
    #--------------------------------------------------------------------------------
    def key(self):
    #--------------------------------------------------------------------------------
        return self.keyword.strip()
    
    #--------------------------------------------------------------------------------
    def set_key(self, keyword):
    #--------------------------------------------------------------------------------
        self.keyword = keyword #.ljust(8)
    
    #--------------------------------------------------------------------------------
    def unformatted(self):
    #--------------------------------------------------------------------------------
        #print(endian + 'i8si4sii{}{}i'.format(self.length, pack_char[self.datatype]))
        dtype = self.datatype.encode()
        length = self.length
        bytes_ = bytearray()
        # header
        bytes_ += struct.pack(endian + 'i8si4si', 16, self.keyword.encode(), length, dtype, 16)
        # data is split in multiple records if length > 1000 
        data = self.data
        typesize = datasize[dtype]
        #start = 0
        #self.print()
        #while len(data[start:]) > 0:
        while data.size > 0:
            length = min(len(data), self.max_length)
            #length = min(len(data[start:]), self.max_length)
            #print(start, start+length)
            size = typesize*length
            bytes_ += struct.pack(endian + 'i{}{}i'.format(length, unpack_char[dtype]), size, *data[:length], size)
            #bytes_ += struct.pack(endian + 'i{}{}i'.format(length, unpack_char[dtype]), size, *data[start:start+length], size)
            data = data[length:]
            #start += length
        return bytes_
                
    #--------------------------------------------------------------------------------
    def print(self):
    #--------------------------------------------------------------------------------
        data = ''
        if len(self.data) > 0:
            data = 'data[0,-1] = ['+str(self.data[0])+', '+str(self.data[-1])+']'
        print(self.keyword, self.length, self.datatype, data)

        
#====================================================================================
class fmt_file:
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
    def __init__(self, filename):
    #--------------------------------------------------------------------------------
        self.name = Path(filename)
        self.fh = None

    #--------------------------------------------------------------------------------
    def is_file(self):
    #--------------------------------------------------------------------------------
        if self.name.is_file():
            return True
        else:
            print('File {} does not exist!'.format(self.name))
            return False
        
    #--------------------------------------------------------------------------------
    def blocks(self, warn_missing=False):
    #--------------------------------------------------------------------------------
        if not self.is_file():
             return
        #block = fmt_block()
        with open(self.name) as self.fh:
            for line in self.fh:
                try:
                    keyword, length, dtype = self.read_header(line)
                    data = self.read_data(length, dtype)
                except StopIteration:
                    if warn_missing:
                        print("\n  WARNING: Missing data in '{}' block, file {} not complete!".format(keyword,self.name.name))
                    return
                except TypeError:
                    return
                else:
                    #block.update(keyword, length, dtype, data)
                    #yield block
                    yield fmt_block(keyword, length, dtype, data)
                           
                
    #--------------------------------------------------------------------------------
    def read_header(self, line): 
    #--------------------------------------------------------------------------------
        words = line.rstrip().split("\'")
        try:
            return words[1], int(words[2]), words[3]
        except IndexError:
            return
        
    #--------------------------------------------------------------------------------
    def read_data(self, length, dtype, skip=False):
    #--------------------------------------------------------------------------------
        if skip:
            data = None
        else:
            data = zeros(length, dtype=datatype[dtype])
        n = 0
        while n < length:
            line = next(self.fh)
            cols = line.rstrip().split()
            m = len(cols)
            #print(n,m,cols)
            if not skip:
                if dtype=='DOUB':
                    cols = [c.replace('D','E') for c in cols]
                try:
                    data[n:n+m] = cols
                except ValueError:
                    return data
            n += m
        return data
            
    #--------------------------------------------------------------------------------
    def start_key(self):
    #--------------------------------------------------------------------------------
        if not self.is_file():
             return
        with open(self.name) as f:
            start = f.readline().split("'")[1]
        return start.strip()
        
        

    #----------------------------------------------------------------------------
    def convert(self, ext='UNRST', init_key='SEQNUM', rename_duplicate=True,
                rename_key=None, echo=False, progress=False): 
    #--------------------------------------------------------------------------------
        #  
        #
        #start = self.start_key().ljust(8)
        if rename_key and len(rename_key)<2:
            raise SystemError('ERROR in convert: Format of rename_keyword options is ' +
                              "('old name', 'new name'), but "+
                              '{} were given'.format(rename_key))
        stem = self.name.stem.upper()
        fname = str(self.name.parent/stem)+'.'+ext
        out_file = open(fname, 'wb')
        bytes_ = bytearray()
        n = 0
        count = {}
        for block in self.blocks():
            #block.print()
            key = block.key()
            if key==init_key and len(bytes_)>0:
                # write previous block to file, and reset bytes_
                n += 1
                out_file.write(bytes_)
                # reset bytes for next section
                bytes_ = bytearray()
                if progress:
                    progress(n)
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
            print('{} converted to {}'.format(self.name.name,Path(fname)))
        # if message and any([n>1 for n in num.values()]):
        #         message(('info',"Duplicate keyword '{}' in {} ignored during convert".
        #                  format(', '.join([k for k,v in num.items() if v>1]),self.name.name)))            
        return Path(fname)

    # #----------------------------------------------------------------------------
    # def convert(self, ext='UNRST', duplicate=None, ignore=None, echo=False, progress=False, message=False): 
    # #--------------------------------------------------------------------------------
    #     #  
    #     #
    #     start = self.start_key().ljust(8)
    #     if duplicate:
    #         duplicate = duplicate.ljust(8)
    #     fname = str(self.name.parent/self.name.stem)+'.'+ext
    #     unformatted_file = open(fname, 'wb')
    #     bytes_ = bytearray()
    #     n = 0
    #     num = {}
    #     for block in self.blocks():
    #         #block.print()
    #         key = block.key()
    #         if key==start and len(bytes_)>0:
    #             # write previous block to file, and reset bytes_
    #             n += 1
    #             unformatted_file.write(bytes_)
    #             if progress:
    #                 progress(n)
    #             bytes_ = bytearray()
    #             num = {}
    #         num[key] = 1 + (num.get(key) or 0)
    #         #bytes_ += block.unformatted()
    #         if duplicate and key==duplicate:
    #             if num[key] != ignore:
    #                 #print(key, str(ignore))
    #                 bytes_ += block.unformatted()
    #         elif num[key] < 2:
    #             bytes_ += block.unformatted()
    #             #print(key)
    #     unformatted_file.close()
    #     if echo:
    #         print('{} converted to {}'.format(self.name.name,Path(fname)))
    #     if message and any([n>1 for n in num.values()]):
    #             message(('info',"Duplicate keyword '{}' in {} ignored during convert".
    #                      format(', '.join([k for k,v in num.items() if v>1]),self.name.name)))            
    #     return fname

    
    # ----------------------------------------------------------------------------
    # def find_duplicate(self, init_key='SEQNUM'):
    # ----------------------------------------------------------------------------
    #     n = 0
    #     count = {}
    #     for block in self.blocks():
    #         key = block.key()
    #         if key==init_key:
    #             n += 1
    #             if n>2:
    #                 break
    #         count[key] = 1 + (count.get(key) or 0)
    #     return [k for k,v in count.items() if v>1]
        
    # #--------------------------------------------------------------------------------
    # def records(self, skip=0, warn_missing=False):
    # #--------------------------------------------------------------------------------
    #     if not self.name.is_file():
    #         print('File {} does not exist!'.format(self.name))
    #         return
    #     num = -1
    #     record = []
    #     end_key = None
    #     header_line = ''
    #     with open(self.name) as self.fh:
    #         for line in self.fh:
    #             try:
    #                 keyword, length, dtype = self.read_header(line)
    #                 if num==-1:
    #                     start = keyword
    #                 if keyword==start:
    #                     num += 1
    #                     if num < skip:
    #                         skip_this = True
    #                     else:
    #                         skip_this = False
    #                     if record and not skip_this:
    #                         if end_key and end_key != record[-1].key():
    #                             print('WARNING! End keyword difference in records')
    #                             raise StopIteration
    #                         yield record
    #                         end_key = record[-1].key()
    #                         record = []
    #                 data = self.read_data(length, dtype, skip=skip_this) #, line=line)
    #                 if not skip_this:
    #                     record.append( Block(keyword, length, dtype, data) )
    #             except StopIteration:
    #                 if warn_missing:
    #                     print("\n  WARNING: Missing data in '{}' block, file {} not complete!".format(keyword,self.name.name))
    #                 return
    #         yield record
    #         if end_key and end_key != record[-1].key():
    #             print('WARNING! End keyword difference in records')
            

    
#====================================================================================
class RSM_block:
#====================================================================================
    #--------------------------------------------------------------------------------
    def __init__(self, var, unit, well, data):
    #--------------------------------------------------------------------------------
        self.var = var
        self.unit = unit
        self.well = well
        self.data = data
        self.nrow = len(self.data)
        
    #--------------------------------------------------------------------------------
    def get_data(self):
    #--------------------------------------------------------------------------------
        for col,(v,u,w) in enumerate(zip(self.var, self.unit, self.well)):
            yield (v, u, w, [self.data[row][col] for row in range(self.nrow)])
        
        
#====================================================================================
class RSM_file:
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
    def get_data(self):
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
    def read_block(self):
    #--------------------------------------------------------------------------------
        self.skip_lines(3)
        var, unit, well = self.read_var_unit_well()
        self.skip_lines(2)
        data = self.read_data(ncol=len(var))
        yield RSM_block(var, unit, well, data)
                
    #--------------------------------------------------------------------------------
    def skip_lines(self, n): 
    #--------------------------------------------------------------------------------
        for i in range(n): 
            next(self.fh) 
        
    #--------------------------------------------------------------------------------
    def read_data(self, ncol=None):
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
    def get_columns_by_position(self, line=None):
    #--------------------------------------------------------------------------------
        '''
        Return None if column is empty
        '''
        #line = line.rstrip()
        n = len(self.colpos)
        words = [None]*n
        for i in range(n-1):
            a, b = self.colpos[i], self.colpos[i+1]
            words[i] = line[a:b].strip() or None
            #print(a, b, words[i])
        return words

    
    #--------------------------------------------------------------------------------
    def read_var_unit_well(self):
    #--------------------------------------------------------------------------------
        line = next(self.fh)#.rstrip()
        var = line.split()
        start = 0
        self.colpos = []
        #print(var)
        for v in var:
            i = line.index(v,start)
            self.colpos.append(i)
            start = i+len(var)
        self.colpos.append(len(line))
        #self.colpos = [line.index(v) for v in var]
        #print(self.colpos)
        unit = self.get_columns_by_position(line=next(self.fh))
        well = self.get_columns_by_position(line=next(self.fh))
        return var, unit, well

    #--------------------------------------------------------------------------------
    def block_length(self): 
    #--------------------------------------------------------------------------------
        with open(self.name) as fh: 
            nb, n = 0, 0 
            for line in fh: 
                n += 1 
                if line[0]==self.tag: 
                    nb += 1 
                    if nb==2: 
                        return int(n) 
