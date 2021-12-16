
# -*- coding: utf-8 -*-

#import os
from pathlib import Path
from re import RegexFlag, findall, compile, DOTALL, search
from time import sleep, time
from datetime import timedelta, datetime
from mmap import mmap, ACCESS_READ

#from struct import unpack

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
def get_keyword(file, keyword, end='', comment='#', ignore_case=True, raise_error=True):
#-----------------------------------------------------------------------
    #print(f'get_keyword({file}, {keyword}, end={end})')
    if not Path(file).is_file():
        return []
    flags = 0
    if ignore_case:
        flags = RegexFlag.IGNORECASE
    data = remove_comments(file, comment=comment, raise_error=raise_error)
    if data == []:
        return []
    #print(data)
    space = '\s'
    slash = '/'
    if end in (' ','\s','\n','\t'):
        end = space
        space = ''
    if end == slash:
        slash = ''
    # Lookahead used at the end to mark end without consuming
    regex = compile(fr"{keyword}\s+([0-9A-Za-z._+:{space}{slash}\\-]+)(?={end})", flags=flags)   
    values = [v.split() for v in regex.findall(data)]
    #print(keyword, values)
    return [float_or_str(v) for v in values]
    #return list(regex.finditer(data))

#--------------------------------------------------------------------------------
def get_python_version():
#--------------------------------------------------------------------------------
    from sys import version_info
    return version_info


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
def read_file(file, raise_error=True):
#--------------------------------------------------------------------------------
    if not Path(file).is_file():
        if raise_error:
            raise SystemError(f'ERROR {file} not found in read_file()')
        else:   
            return ''    
    lines = ''
    try:
        with open(file) as f:
        #with open(file, encoding='ascii', errors='surrogateescape') as f:
            lines = f.readlines()
    except UnicodeDecodeError as e:
        #print(e)
        #with open(file, encoding='ISO-8859-1') as f:
        with open(file, encoding='latin-1') as f:
        #with open(file) as f:
            lines = f.readlines()
    except OSError as err:
        raise SystemError(f'Unable to read {file}: {err}')
    return ''.join(lines)

#--------------------------------------------------------------------------------
def write_file(file, text):
#--------------------------------------------------------------------------------
    try:
        with open(file, 'w') as f:
            f.write(text)
    except UnicodeEncodeError:
        #with open(file, 'w', encoding='ISO-8859-1') as f:
        with open(file, 'w', encoding='latin-1') as f:
            f.write(text)
    except OSError as err:
        raise SystemError(f'Unable to write to {file}: {err}')



#--------------------------------------------------------------------------------
def remove_comments(file, comment='--', end=None, raise_error=True):
#--------------------------------------------------------------------------------
    if not Path(file).is_file():
        if raise_error:
            raise SystemError(f'ERROR {file} not found in remove_comments()')    
        else:
            return []
    try:
        with open(file) as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        #with open(file, encoding='ISO-8859-1') as f:
        with open(file, encoding='latin-1') as f:
            lines = f.readlines()
    lines = ''.join([l.split(comment)[0]+'\n' if comment in l else l for l in lines])
    if end:
        pos = [m.end() for m in compile(rf'\b{end}\b').finditer(lines)]
        if pos:
            lines = lines[:pos[0]]+'\n'
    return lines

#--------------------------------------------------------------------------------
def safeindex(alist, value):
#--------------------------------------------------------------------------------
    return alist.index(value) if value in alist else None

#--------------------------------------------------------------------------------
def flat_list(alist):
#--------------------------------------------------------------------------------
    return [item for sublist in alist for item in sublist]

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
def file_contains(fname, text='', regex='', comment='#', end=None, raise_error=True):
#--------------------------------------------------------------------------------
    #print(f'file_contains({fname}, {text})')
    if not Path(fname).is_file():
        if raise_error:
            raise SystemError('ERROR ' + fname + ' not found in file_contains()')    
        else:
            return False
    if text:
        regex = rf'\b{text}\b'
    lines = remove_comments(fname, comment=comment, end=end)
    if search(regex, lines): 
        return True
    return False

#--------------------------------------------------------------------------------
def delete_all(folder, keep_folder=False):
#--------------------------------------------------------------------------------
    for child in Path(folder).iterdir():
        if child.is_file():
            child.unlink()
        else:
            delete_all(child)
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
        fname = [str(fname)]
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
def delete_files_matching(pattern, echo=False, raise_error=False):
#------------------------------------------------
    msg = ''
    if not isinstance(pattern, (list, tuple)):
        pattern = (pattern,)
    for pat in pattern:
        pat = Path(pat)
        for file in pat.parent.glob(pat.name):
            if echo:
                print('Removing ' + str(file))
            try:
                file.unlink()
            except PermissionError:
                msg = 'WARNING Unable to delete file '+str(file)+', maybe it belongs to another process'
                if raise_error:
                    raise SystemError(msg)
    return msg


#------------------------------------------------
def loop_until(func, *args, limit=None, pause=None, loop_func=None, **kwargs):
#------------------------------------------------
    n = 0
    if not loop_func:
        loop_func = lambda:None
    while True:
        if func(**kwargs):
            return n
        if pause:
            sleep(pause)
        n += 1
        if limit and n > limit:
            # if error:
            #     raise SystemError(error)
            # else:
            return -1
        loop_func()


#------------------------------------------------
def list2str(alist, start='', end='', sep=''):
#------------------------------------------------
    return start + '%s'%', '.join(sep+'{}'.format(i)+sep for i in alist) + end

#------------------------------------------------
def list2text(alist):
#------------------------------------------------
    text = ', '.join([str(a) for a in alist])
    return ' and'.join(text.rsplit(',',1))

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
def matches(file=None, pattern=None, length=0, multiline=False, pos=None):
#--------------------------------------------------------------------------------
    flags = 0
    if multiline:
        flags = DOTALL
    regexp = compile(pattern.encode(), flags=flags)
    with open(file) as f:
        with mmap(f.fileno(), length=length, access=ACCESS_READ) as data:
            if pos:
                data = data[pos:]
            for match in regexp.finditer(data):
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
    def __init__(self, N=1, update=1, format='%', indent=3, min=0):
    #--------------------------------------------------------------------------------
        self.start_time = time()
        self.update = update
        self.N = N
        self.n0 = 0       # n0 != 0 if time-counter is reset
        self.min = min    # sets a minimum for the progress bar
        if '%' in format:
            self.format = self.format_percent
        if '#' in format:
            self.format = self.format_bar
            try: n = int(format.split('#')[0])
            except ValueError: n = 1
            self.bar_length = n
        self.indent = indent*' '

    #--------------------------------------------------------------------------------
    def set_min(self, min):
    #--------------------------------------------------------------------------------
        self.reset_time()
        self.min = self.n0 = min

    #--------------------------------------------------------------------------------
    def reset(self, N=1):
    #--------------------------------------------------------------------------------
        self.N = N
        self.n0 = 0
        self.reset_time()

    #--------------------------------------------------------------------------------
    def reset_time(self, n=0):
    #--------------------------------------------------------------------------------
        #print('reset_time', n)
        self.start_time = time()
        self.n0 = n
        self.min = 0

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
        return 'Progress {: 4d} / {:4d} = {:.0f} %   ETA: {}'.format(int(n), int(self.N), percent, self.eta) 

    #--------------------------------------------------------------------------------
    def format_bar(self, n):
    #--------------------------------------------------------------------------------
        nn = max(n-self.min, 0)
        hash = int(self.bar_length*nn/(self.N-self.min))
        rest = self.bar_length - hash
        count = f'{int(n)}'
        if self.min > 0:
            count = f'({int(self.min)} + {int(nn)})'
        return f'{count} / {int(self.N)}  [{hash*"#"}{rest*"-"}]  {self.eta}'

    #--------------------------------------------------------------------------------
    def set_N(self, N):
    #--------------------------------------------------------------------------------
        self.N = N

    #--------------------------------------------------------------------------------
    def print(self, n, text=None, trail_space=3):
    #--------------------------------------------------------------------------------
        #print(n, self.min)
        if n>self.min and int(n)%self.update==0:
            #self.calc_estimated_arrival(n)
            self.remaining_time(n)
            print(f'\r{text or ""}'+self.indent+self.format(n)+trail_space*' ', end='', flush=True)
            #print(f'\r{text or ""}{self.indent}{self.format(n)}{trail_space*" "}', end='', flush=True)

    #--------------------------------------------------------------------------------
    def remaining_time(self, n):
    #--------------------------------------------------------------------------------
        #print('remaining_time: ',n, time()-self.start_time)
        eta = 0
        if n==0:
            self.reset_time()
        elif n > self.n0:
            nn = n-self.n0
            eta = max( int( (self.N-n) * (time()-self.start_time)/nn ) , 0)
        self.eta = timedelta(seconds=eta)
        return str(self.eta)
    
    # #--------------------------------------------------------------------------------
    # def elapsed_time(self):
    # #--------------------------------------------------------------------------------
    #     Dt = time()-self.start_time
    #     ela = timedelta(seconds=int(Dt))
    #     return str(ela)


    # #--------------------------------------------------------------------------------
    # def total_time(self, n):
    # #--------------------------------------------------------------------------------
    #     Dt = time()-self.start_time
    #     tot = timedelta(seconds=int(Dt*self.N/n))
    #     return str(tot)


# #====================================================================================
# class check_endtag:
# #====================================================================================

#     #--------------------------------------------------------------------------------
#     def __init__(self, file=None, endtag='', comment=''):
#     #--------------------------------------------------------------------------------
#         self._file = Path(file)        
#         self._endtag = endtag
#         self._endtag_size = len(endtag)
#         self._comment = comment
        
#     #--------------------------------------------------------------------------------
#     def file(self):
#     #--------------------------------------------------------------------------------
#         return self._file
        
#     #--------------------------------------------------------------------------------
#     def find_endtag(self, binary=False):     
#     #--------------------------------------------------------------------------------
#         if binary:
#             return self.find_endtag__binary()
#         else:
#             return self.find_endtag__ascii()
        
#     #--------------------------------------------------------------------------------
#     def find_endtag__ascii(self):     
#     #--------------------------------------------------------------------------------
#         try:
#             if self._file.stat().st_size < self._endtag_size:
#                 return False
#             with open(self._file, 'r') as f:
#                 for line in f:
#                     if self._endtag in line:
#                         return True
#         except FileNotFoundError:
#             return None

#     #--------------------------------------------------------------------------------
#     def find_endtag__binary(self):  
#     #--------------------------------------------------------------------------------
#         try:
#             size = self._endtag_size+2 # 2 = '\r\n'
#             if self._file.stat().st_size < size:
#                 return False
#             endtag = self._endtag.encode()
#             with open(self._file, 'rb') as f:
#                 f.seek(-size, os.SEEK_END)
#                 end = unpack('>%ds'%size, f.read(size))[0]
#                 if endtag in end:
#                     return True
#         except FileNotFoundError:
#             return None




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

