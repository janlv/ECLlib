#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import errno
from pathlib import Path, PurePath
from re import findall, finditer, DOTALL, MULTILINE
from time import sleep, time
from datetime import timedelta, datetime
from mmap import mmap, ACCESS_READ
from struct import unpack

#--------------------------------------------------------------------------------
def safeindex(alist, value):
#--------------------------------------------------------------------------------
    return alist.index(value) if value in alist else None

#--------------------------------------------------------------------------------
def flatten_list(alist):
#--------------------------------------------------------------------------------
    return [item for sublist in alist for item in sublist]

#--------------------------------------------------------------------------------
def upper_and_lower(alist):
#--------------------------------------------------------------------------------
    return flatten_list([[item.upper(),item.lower()] for item in alist])

#--------------------------------------------------------------------------------
def file_contains(fname, text='', comment='#'):
#--------------------------------------------------------------------------------
    if not Path(fname).is_file():
        raise SystemError('ERROR ' + fname + ' is not found')    
    with open(fname, 'r') as f:
        for line in f:
            line = line.lstrip()
            if not line.startswith(comment) and line.startswith(text):
                return True 
    return False

#--------------------------------------------------------------------------------
def delete_all(folder):
#--------------------------------------------------------------------------------
    for child in Path(folder).iterdir():
        if child.is_file():
            child.unlink()
        else:
            delete_all(child)
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
    
#--------------------------------------------------------------------------------
def exit_without_atexit():
#--------------------------------------------------------------------------------
    #from os import _exit as os_exit
    os._exit(0)

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
    
    ### check python version
    sysmajor = version_info.major
    sysminor = version_info.minor
    if sysmajor<major or (sysmajor==major and sysminor<minor):
        raise SystemExit('\nThis script requires python{}.{} or higher, you are using python{}.{}\n'.
                         format(major, minor, sysmajor, sysminor))
    return True

#------------------------------------------------
def silentdelete(fname, echo=False):
#------------------------------------------------
    if isinstance(fname, (str,Path)):
        fname = list(str(fname))
    if isinstance(fname, (list, tuple)): # or isinstance(fname, tuple):
        for f in fname:
            file = Path(f)
            try:
                file.is_file() and file.unlink()
            except (PermissionError, FileNotFoundError) as e:
                if echo:
                    print('Unable to delete {}: {}'.format(f, e))
                else:
                    pass
            else:
                if echo:
                    print('Deleted {}'.format(f))
    else:
        raise SystemError('silentdelete: Unknown format {} passed'.format(type(fname)))


#--------------------------------------------------------------------------------
def float_or_str(word): 
#--------------------------------------------------------------------------------
    try:
        return float(word)
    except (TypeError, ValueError):
        return str(word)

    
#------------------------------------------------
def delete_files_matching(pattern, echo=False):
#------------------------------------------------
    pattern = Path(pattern)
    for file in pattern.parent.glob(pattern.name):
        if echo:
            print('Removing ' + str(file))
        try:
            file.unlink()
        except PermissionError:
            raise SystemError('Unable to delete file '+str(file)+', maybe it belongs to another process')



#------------------------------------------------
def loop_until(func, *args, limit=None, pause=None, error=None, loop_func=None, **kwargs):
#------------------------------------------------
    n = 0
    if not loop_func:
        loop_func = lambda:None
    while True:
        if func(**kwargs):
            return n
        if pause:
            sleep(pause)
        #print(n, limit, pause, error, loop_func)
        if limit:
            n += 1
            if n > limit:
                if error:
                    raise SystemError(error)
                else:
                    return -1
        loop_func()


#------------------------------------------------
def list2str(alist, start='', end='', sep=''):
#------------------------------------------------
    return start + '%s'%', '.join(sep+'{}'.format(i)+sep for i in alist) + end


#------------------------------------------------
def tail_file(fname, nchars=300):
#------------------------------------------------
    if Path(fname).is_file():
        with open(fname, 'rb') as f:
            f.seek(-nchars, 2)
            return f.read(nchars).decode()

#------------------------------------------------
def safeopen(filename, mode):
#------------------------------------------------
    try:
        filehandle = open(filename, mode)
        return filehandle
    except OSError as err:
        raise SystemError('Unable to open file {} in mode {}: {}'.format(filename, mode, err.errno))
        
#--------------------------------------------------------------------------------
def warn_empty_file(file, comment=''):
#--------------------------------------------------------------------------------
    with open(file, 'r') as f:
        for line in f:
            if not line.startswith(comment) and not line.isspace():
                return
    print('WARNING! {} is empty'.format(file))

#--------------------------------------------------------------------------------
def matches(file=None, pattern=None, length=0, multiline=False):
#--------------------------------------------------------------------------------
    flags = 0
    if multiline:
        flags = DOTALL
    with open(file) as f:
        with mmap(f.fileno(), length=length, access=ACCESS_READ) as data:
            for match in finditer(pattern.encode(), data, flags=flags):
                yield match
        

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
    def __init__(self, N=1, update=1, format='%', indent=3):
    #--------------------------------------------------------------------------------
        self.start_time = time()
        self.update = update
        self.N = N
        if '%' in format:
            self.format = self.format_percent
        if '#' in format:
            self.format = self.format_bar
            try: n = int(format.split('#')[0])
            except ValueError: n = 1
            self.bar_length = n
        self.indent = indent*' '

    #--------------------------------------------------------------------------------
    def reset(self, N=1):
    #--------------------------------------------------------------------------------
        self.N = N
        self.start_time = time()

    #--------------------------------------------------------------------------------
    def calc_estimated_arrival(self, n):
    #--------------------------------------------------------------------------------
        Dt = time()-self.start_time
        self.ela = timedelta(seconds=int(Dt))
        self.eta = timedelta(seconds=int((self.N-n)*Dt/n))

    #--------------------------------------------------------------------------------
    def format_percent(self, n):
    #--------------------------------------------------------------------------------
        return 'Progress {: 4d}/{:4d} = {:.0f} %   ETA: {}'.format(n, self.N, 100*n/self.N, self.eta) 

    #--------------------------------------------------------------------------------
    def format_bar(self, n):
    #--------------------------------------------------------------------------------
        hash = int(self.bar_length*n/self.N)
        rest = self.bar_length - hash
        return '{:4d}/{:4d}  [{}{}]  {}'.format(n, self.N, hash*'#', rest*'-', self.eta) 

    #--------------------------------------------------------------------------------
    def set_N(self, N):
    #--------------------------------------------------------------------------------
        self.N = N

    #--------------------------------------------------------------------------------
    def print(self, n):
    #--------------------------------------------------------------------------------
        if n>0 and n%self.update==0:
            self.calc_estimated_arrival(n)
            print('\r'+self.indent+self.format(n)+10*' ', end='', flush=True)
 
    #--------------------------------------------------------------------------------
    def remaining_time(self, n):
    #--------------------------------------------------------------------------------
        eta = 0        
        if n==0:
            self.start_time = time()
            #self.steps = [0]
        if n>0:
            #if n>self.steps[-1]:
            #    self.steps.append(n) 
            Dt = time()-self.start_time
            eta = max(int((self.N-n)*Dt/n), 0)
            #t_step = Dt/len(self.steps)
            #interval = 1
            #if len(self.steps)>1:
            #    interval=self.steps[-1] - self.steps[-2]
            #rem_steps = (self.N-n)/interval
            #eta = int(rem_steps*t_step)
            #print('steps:{}, int:{}, rem:{}, t: {}, eta:{}'.format(self.steps, interval, rem_steps, t_step, eta))
        #print(self.start_time)
        return str(timedelta(seconds=eta))
    
    #--------------------------------------------------------------------------------
    def elapsed_time(self):
    #--------------------------------------------------------------------------------
        Dt = time()-self.start_time
        ela = timedelta(seconds=int(Dt))
        return str(ela)


    # #--------------------------------------------------------------------------------
    # def total_time(self, n):
    # #--------------------------------------------------------------------------------
    #     Dt = time()-self.start_time
    #     tot = timedelta(seconds=int(Dt*self.N/n))
    #     return str(tot)


#====================================================================================
class check_endtag:
#====================================================================================

    #--------------------------------------------------------------------------------
    def __init__(self, file=None, endtag='', comment=''):
    #--------------------------------------------------------------------------------
        self._file = Path(file)        
        self._endtag = endtag
        self._endtag_size = len(endtag)
        self._comment = comment
        
    #--------------------------------------------------------------------------------
    def file(self):
    #--------------------------------------------------------------------------------
        return self._file
        
    #--------------------------------------------------------------------------------
    def find_endtag(self, binary=False):     
    #--------------------------------------------------------------------------------
        if binary:
            return self.find_endtag__binary()
        else:
            return self.find_endtag__ascii()
        
    #--------------------------------------------------------------------------------
    def find_endtag__ascii(self):     
    #--------------------------------------------------------------------------------
        try:
            if self._file.stat().st_size < self._endtag_size:
                return False
            with open(self._file, 'r') as f:
                for line in f:
                    if self._endtag in line:
                        return True
        except FileNotFoundError:
            return None

    #--------------------------------------------------------------------------------
    def find_endtag__binary(self):  
    #--------------------------------------------------------------------------------
        try:
            size = self._endtag_size+2 # 2 = '\r\n'
            if self._file.stat().st_size < size:
                return False
            endtag = self._endtag.encode()
            with open(self._file, 'rb') as f:
                f.seek(-size, os.SEEK_END)
                end = unpack('>%ds'%size, f.read(size))[0]
                if endtag in end:
                    return True
        except FileNotFoundError:
            return None




#====================================================================================
class Timer:
#====================================================================================

    #--------------------------------------------------------------------------------    
    def __init__(self, filename=None):
    #--------------------------------------------------------------------------------    
        self.counter = 0
        self.timefile = Path('{}_timer.dat'.format(filename))
        self.timefile.write_text('# step \t seconds\n')
        self.starttime = time()
        self.info = 'Execution time is recorded in {}'.format(self.timefile.name)


    #--------------------------------------------------------------------------------    
    def start(self):
    #--------------------------------------------------------------------------------    
        self.counter += 1
        self.starttime = time()


    #--------------------------------------------------------------------------------    
    def stop(self):
    #--------------------------------------------------------------------------------    
        with self.timefile.open('a') as f:
            f.write('{:d}\t{:.3e}\n'.format(self.counter, time()-self.starttime))

