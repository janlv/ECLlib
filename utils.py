
# -*- coding: utf-8 -*-

from pathlib import Path
from re import RegexFlag, findall, compile, DOTALL, search, sub, IGNORECASE
from threading import Thread
from time import sleep, time
from datetime import timedelta, datetime, time as dt_time
from mmap import mmap, ACCESS_READ, ACCESS_WRITE
from numpy import array, sum as npsum
from psutil import Process, NoSuchProcess, wait_procs
from signal import SIGTERM
from contextlib import contextmanager
from itertools import chain, islice, takewhile, tee, zip_longest
from collections import deque

# Short Python regexp guide:
#   \s : whitespace, [ \t\n\r\f\v]
#   \w : alphanumeric, [a-zA-Z0-9_]
#   \d : decimal digit, [0-9]
#   \b : word-delimiter
#    ? : 0 or 1 repetitions
#    + : 1 or more rep.
#    * : 0 or more rep.


### pairwise is new in python 3.10, define it for older versions
# try:
#     from itertools import pairwise
# except ImportError:
#     from itertools import tee

#-----------------------------------------------------------------------
def pairwise(iterable): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

#-----------------------------------------------------------------------
def take(n, iterable): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

#-----------------------------------------------------------------------
def tail(n, iterable): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    "Return an iterator over the last n items"
    # tail(3, 'ABCDEFG') --> E F G
    return iter(deque(iterable, maxlen=n))

#------------------------------------------------
def tail_file(fname, n=0):
#------------------------------------------------
    'Return the last n lines of a file'
    if fname and Path(fname).is_file():   
        with open(fname) as f:
            return tail(n,f)
    return iter(())

#-----------------------------------------------------------------------
def prepend(value, iterator): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    "Prepend a single value in front of an iterator"
    # prepend(1, [2, 3, 4]) -> 1 2 3 4
    return chain([value], iterator)

#-----------------------------------------------------------------------
def flatten(list_of_lists): # From Itertools Recipes at docs.python.org
#-----------------------------------------------------------------------
    "Flatten one level of nesting"
    try:
        return list(chain.from_iterable(list_of_lists))
    except TypeError:
        return list_of_lists
        
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

# #--------------------------------------------------------------------------------
# def flat_list(alist):
# #--------------------------------------------------------------------------------
#     return [item for sublist in alist for item in sublist]

#-----------------------------------------------------------------------
def split_by_words(string, words, comment=None): #, wb=r'\b'):
#-----------------------------------------------------------------------
    '''
    Split a string, with comments, into sections based on a list of unique words.
    Returns a dict with words as keys and a tuple of begin and end positins
    '''
    regex =  (comment and rf'(?<!{comment})' or '') + r'\s*\b' + r'\b|\b'.join(words) + r'\b'
    #regex =  (comment and rf'(?<!{comment})' or '') + r'\s*' + wb + rf'{wb}|{wb}'.join(words) + wb
    matches = compile(regex, flags=IGNORECASE).finditer(string)
    ### Append string end pos as tuple of tuple
    tag_pos = chain( ((m.group(), m.start()) for m in matches), (('', len(string)),) )
    return ((tag, a, b) for (tag, a), (_, b) in pairwise(tag_pos))
    #return [(a[0],a[1],b[1]) for a,b in pairwise(tag_pos)]


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
    ### Lookahead used at the end to mark end without consuming
    regex = compile(fr"{keyword}\s+([0-9A-Za-z._+:{space}{slash}\\-]+)(?={end})", flags=flags)   
    #values = [v.split() for v in regex.findall(data)]
    #values = (v.split() for v in regex.findall(data))
    #print(keyword, values)
    #return [float_or_str(v) for v in values]
    return [list(convert_float_or_str(v.split())) for v in regex.findall(data)]
    #return list(regex.finditer(data))


#--------------------------------------------------------------------------------
def convert_float_or_str(words): 
#--------------------------------------------------------------------------------
    for w in words: 
        try:
            v = float(w)
        except ValueError:
            v = str(w)
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
    '''
    Zip generators and close them if the zip exits. Zip exits when the first generator is exhausted.
    The __exit__() function for the non-exhausted generators will not be called. 
    This routine closes the generators explicitly.
    '''
    try:
        yield zip(*gen)
    finally:
        [g.close() for g in gen]


#-----------------------------------------------------------------------
def remove_leading_nondigits(txt):
#-----------------------------------------------------------------------
    return sub(r'^[a-zA-Z-+._]*', '', txt)  

#-----------------------------------------------------------------------
def try_except_loop(*args, limit=1, pause=0.05, error=None, raise_error=True, func=None, **kwargs):
#-----------------------------------------------------------------------
    for i in range(limit):
        #print(f'{func.__qualname__}({args},{kwargs}): {i}')
        try:
            result = func(*args, **kwargs)
            break
        except error as e:
            sleep(pause)
    if i==limit-1 and raise_error:
        raise SystemError(f'Unable to complete {func.__qualname__} within {limit} tries during {limit*pause} seconds: {e}')
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
def read_file(file, raise_error=True):
#--------------------------------------------------------------------------------
    file = Path(file)
    if not file.is_file():
        if raise_error:
            raise SystemError(f'ERROR {file} not found in read_file()')
        else:   
            return ''    
    lines = ''
    try:
        with open(file, encoding='utf-8') as f:
        #with open(file, encoding='ascii', errors='surrogateescape') as f:
            lines = f.readlines()
    except UnicodeDecodeError as e:
        #print(e)
        #with open(file, encoding='ISO-8859-1') as f:
        with open(file, encoding='latin-1') as f:
        #with open(file, encoding=getdefaultlocale()[1]) as f:
            lines = f.readlines()
    except OSError as err:
        raise SystemError(f'Unable to read {file}: {err}')
    # except:
    #     e = exc_info()
    #     print(e)
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
def remove_comments(path, comment='--', join=True, raise_error=True, encoding=None, end=None):
#--------------------------------------------------------------------------------
    try:
        path = Path(path)
        if not path.is_file:
            if raise_error:
                raise SystemError(f'ERROR {path} not found in remove_comments()')    
            return []
        with open(path, encoding=encoding) as file:
            lines = (line.split(comment)[0].strip() for l in file if (line:=l.strip()) and not line.startswith(comment))
            if end:
                # If end should be included
                #lines = chain(takewhile(lambda x: x != end, lines), (end,))
                lines = takewhile(lambda x: x != end, lines)
            if join:
                return '\n'.join(lines)+'\n'
            return list(lines)
    except (FileNotFoundError, PermissionError):
        return []
    except UnicodeDecodeError as e:
        if encoding:
            raise SystemError('ERROR {path} raised UnicodeDecodeError for both UTF-8 and latin-1 encodings: {e}')
        return remove_comments(path, encoding='latin-1', comment=comment, join=join, raise_error=raise_error, end=end)



# #--------------------------------------------------------------------------------
# def remove_comments_old(file=None, lines=None, comment='--', end=None, raise_error=True, newline=True):
# #--------------------------------------------------------------------------------
#     if file:
#         try:
#             if not Path(file).is_file():
#                 if raise_error:
#                     raise SystemError(f'ERROR {file} not found in remove_comments()')    
#                 else:
#                     return []
#             with open(file) as f:
#                 lines = f.readlines()
#         except UnicodeDecodeError:
#             #with open(file, encoding='ISO-8859-1') as f:
#             with open(file, encoding='latin-1') as f:
#                 lines = f.readlines()
#         except (FileNotFoundError, PermissionError):
#             return []
#     lf = ''
#     if newline:
#         lf = '\n'
#     lines = ''.join([l.split(comment)[0]+lf if comment in l else l for l in lines])
#     if end:
#         pos = [m.end() for m in compile(rf'\b{end}\b').finditer(lines)]
#         if pos:
#             lines = lines[:pos[0]]+lf
#     return lines

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
    return flatten([[item.upper(),item.lower()] for item in alist])

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


#--------------------------------------------------------------------------------
def file_contains(fname, text='', regex='', comment='#', end=None, raise_error=True):
#--------------------------------------------------------------------------------
    #print(f'file_contains({fname}, {text})')
    if not Path(fname).is_file():
        if raise_error:
            raise SystemError('ERROR ' + fname + ' not found in file_contains()')    
        else:
            return False
    if isinstance(text, str):
        text = [text]
    regex = [rf'\b{t}\b' for t in text]
    lines = remove_comments(fname, comment=comment, end=end)
    if any(search(r, lines) for r in regex):
        return True
    # for reg in regex:
    #     #regex = rf'\b{text}\b'
    #     if search(reg, lines): 
    #         return True
    return False

#--------------------------------------------------------------------------------
def delete_all(folder, keep_folder=False):
#--------------------------------------------------------------------------------
    if not Path(folder).is_dir():
        return
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
            else:
                pass
        else:
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
            return -1
        loop_func()



#------------------------------------------------
def list2str(alist, start='', end='', sep='', count=False):
#------------------------------------------------
    #return start + '%s'%', '.join(f'{sep}{i}{sep}' for i in alist) + end
    if count:
        return ', '.join([f'{v} (n={npsum(array(alist)==v)})' for v in set(alist)])
    else:
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
    except OSError as err:
        raise SystemError(f'Unable to open file {filename} in mode {mode}: {err.errno}')
        
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
    regexp = compile(pattern.encode(), flags=flags)
    with open(file) as f:
        with mmap(f.fileno(), length=length, access=ACCESS_READ) as data:
            if pos:
                data = data[pos:]
            if check and not check in data:
                data = b''
            for match in regexp.finditer(data):
                yield match

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
        self.start_time = datetime.now()
        #self.update = update
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
        #self.eta = 0
        self.eta = None
        self.last_eta = None
        self.time_str = '--:--:--'
        self.length = 0
        self.prev_n = -1
        #print('Progress:', N, update, format, indent, min)

    #--------------------------------------------------------------------------------
    def __repr__(self):
    #--------------------------------------------------------------------------------
        return f'<Progress start_time:{self.start_time}, N:{self.N}, n0:{self.n0}, min:{self.min}, eta:{self.eta}, last_eta:{self.last_eta}, {self.time_str}>'

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
        self.eta = None
        self.last_eta = None
        self.time_str = '--:--:--'
        self.reset_time(**kwargs)

    #--------------------------------------------------------------------------------
    def reset_time(self, n=0, min=None):
    #--------------------------------------------------------------------------------
        #print('reset_time', n)
        self.start_time = datetime.now() #time()
        #self.start_time = None
        self.min = min and min or 0
        # self.n0 = n
        self.n0 = max(n, self.min)

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
        #print('format_bar', n, self.min)
        hash = 0
        nn = max(n-self.min, 0)
        if (diff := self.N-self.min) > 0:
            hash = int(self.bar_length*nn/diff)
        rest = self.bar_length - hash
        # count = f'{int(n)}'
        t, T = strip_zero((n, self.N))
        if self.min > 0 and n >= self.min:
            #count = f'({int(self.min)} + {int(nn)})'
            a, b = strip_zero((self.min, nn))
            t = f'({a} + {b})'
            #print('format_bar',t)
        bar = hash <= self.bar_length and f'{hash*"#"}{rest*"-"}' or f'-- E R R O R, n:{n}, N:{self.N}, min:{self.min} --'
        return f'{t} / {T}  [{bar}]  {self.time_str}'
        #return f'{t} / {T}  [{bar}]  {self.eta}'

    #--------------------------------------------------------------------------------
    def set_N(self, N):
    #--------------------------------------------------------------------------------
        self.N = N

    #--------------------------------------------------------------------------------
    def print(self, n, text=None):
    #--------------------------------------------------------------------------------
        #print(n, self)
        if self.prev_n < self.min:
            self.start_time = datetime.now()
        if n>self.prev_n and n>self.min and n>self.n0:
            #self.remaining_time(n)
            ### Calculate estimated time of arrival, eta
            nn = n-self.n0
            eta = max( int( (self.N-n) * (datetime.now()-self.start_time).total_seconds()/nn ) , 0)
            self.eta = timedelta(seconds=eta)
            ### Time of this estimate (used if progress printed more often than estimated)
            self.last_eta = datetime.now()
            self.time_str = f'{self.eta}'.split('.')[0]
        elif self.eta:
            self.time_str = f'{self.eta-(datetime.now()-self.last_eta)}'.split('.')[0]
        line = self.format(n)
        trail_space = max(1, self.length - len(line))
        self.length = len(line)
        print(f'\r{text or ""}' + self.indent + line + trail_space*' ', end='', flush=True)
        self.prev_n = n

    # #--------------------------------------------------------------------------------
    # def remaining_time(self, n):
    # #--------------------------------------------------------------------------------
    #     #print('remaining_time: ',n, time()-self.start_time)
    #     #print(n, self)
    #     #eta = None
    #     # if n==0:
    #     #     self.reset_time()
    #     # elif n > self.n0:
    #     if n > self.n0:
    #         nn = n-self.n0
    #         #eta = max( int( (self.N-n) * (time()-self.start_time)/nn ) , 0)
    #         eta = max( int( (self.N-n) * (datetime.now()-self.start_time).total_seconds()/nn ) , 0)
    #         self.eta = timedelta(seconds=eta)
    #     return self.eta
    #     # return str(self.eta)
    
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
class timer_thread:
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
        #self._is_alive = False
        self._starttime = None
        self._endtime = None
        self._thread = Thread(target=self._timer, daemon=True)
        self.DEBUG and print(f'Creating {self}')
        #print(self._limit, self._idle)

    #--------------------------------------------------------------------------------    
    def __str__(self):
    #--------------------------------------------------------------------------------    
        return f'<timer_thread(limit={self._limit}, prec={self._idle}, func={self._func.__qualname__}, thread={self._thread})>'

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
        # self._is_alive = True
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
            #self._is_alive = False
            self._endtime = self.time()
            return True
        return False

    #--------------------------------------------------------------------------------    
    def is_alive(self):
    #--------------------------------------------------------------------------------    
        #return self._is_alive
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
            #if self._is_alive:
            if not self._endtime:
                time = self.time()
                if time >= self._limit:
                    self._call_func()
                    #self._is_alive = False
                    self._endtime = time
                    #print('Called '+self._caller.__qualname__+f' at {sec}')

