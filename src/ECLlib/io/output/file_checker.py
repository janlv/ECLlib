from numpy import asarray
from ECLlib.io.unformatted.base import unfmt_file
from ECLlib.utils import list2str

DEBUG = False

#==================================================================================================
class File_checker:                                                                  # File_checker
#==================================================================================================
    """
    Class to check if a file contains complete block-sequences.
    """

    #----------------------------------------------------------------------------------------------
    def __init__(self, file, start=None, end=None, wait_func=None, 
                 warn_offset=True, timer=False):                                     # File_checker
    #----------------------------------------------------------------------------------------------
        """Initialize the File_checker."""
        if isinstance(file, unfmt_file):
            self._unfmt = file
        else:
            self._unfmt = unfmt_file(file)
        start = start or self._unfmt.start
        end = end or self._unfmt.end
        self._keys = [start.ljust(8).encode(), [], end.ljust(8).encode(),  0]
        self._data = None
        self._wait_func = wait_func
        self._warn_offset = warn_offset
        self._timer = timer
        if DEBUG:
            print(f'Creating {self}')

    #----------------------------------------------------------------------------------------------
    def __repr__(self):                                                              # File_checker
    #----------------------------------------------------------------------------------------------
        """Return a developer-friendly representation."""
        return f'<{type(self)}, file={self._unfmt}>'

    #----------------------------------------------------------------------------------------------
    def __del__(self):                                                               # File_checker
    #----------------------------------------------------------------------------------------------
        """Handle object cleanup."""
        if DEBUG:
            print(f'Deleting {self}')

    #----------------------------------------------------------------------------------------------
    def data(self):                                                                  # File_checker
    #----------------------------------------------------------------------------------------------
        """Return decoded block data."""
        #return self._data and self._data[1]
        return self._data[1] if self._data else None

    #----------------------------------------------------------------------------------------------
    def not_in_sync(self, time, prec=0.1):                                           # File_checker
    #----------------------------------------------------------------------------------------------
        """Return whether the files appear out of sync."""
        data = self.data()
        if data and any(abs(asarray(data)-time) > prec):
            return True
        return False

    #----------------------------------------------------------------------------------------------
    def info(self, data=None, count=False):                                          # File_checker
    #----------------------------------------------------------------------------------------------
        """Return a textual summary of the file."""
        return f"  {self._data[0].decode()} : {list2str(data and data or self._data[1], count=count)}"
        
    #----------------------------------------------------------------------------------------------
    def blocks_complete(self, nblocks=1, only_new=True):                             # File_checker
    #----------------------------------------------------------------------------------------------
        """Return whether the file contains every expected block."""
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

    #----------------------------------------------------------------------------------------------
    def steps_complete(self):                                                        # File_checker
    #----------------------------------------------------------------------------------------------
        """Return whether all report steps are present."""
        # 1: start_list, 3: end_count
        return len(self._keys[1]) == self._keys[3]


    #----------------------------------------------------------------------------------------------
    def warn_if_offset(self):                                                        # File_checker
    #----------------------------------------------------------------------------------------------
        """Warn about detected offsets."""
        msg = ''
        if (offset := self._unfmt.offset()):
            msg = f'WARNING {self._unfmt} not at end after check, offset is {offset}'
        return msg

    #----------------------------------------------------------------------------------------------
    def data_saved_maxmin(self, nblocks=1, niter=100, **kwargs):                     # File_checker
    #----------------------------------------------------------------------------------------------
        """
            Loop for 'niter' iterations until 'nblocks' start/end-blocks are found 
            or end-of-file reached.
        """
        if nblocks == 0:
            return []
        msg = []
        data = []
        n = nblocks
        v = 2
        while n > 0:
            passed = self._wait_func(self.blocks_complete, nblocks=n, limit=niter,
                                     timer=self._timer, v=v, **kwargs)
            if self._unfmt.at_end() and self.steps_complete():
                ### blocks <= max_blocks
                break
            if passed:
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

    #----------------------------------------------------------------------------------------------
    def data_saved(self, nblocks=1, wait_func=None, **kwargs):                       # File_checker
    #----------------------------------------------------------------------------------------------
        """Return whether all data has been written to disk."""
        msg = ''
        wait_func = self._wait_func or wait_func
        is_complete = wait_func(self.blocks_complete, nblocks=nblocks, log=self.info,
                                timer=self._timer, **kwargs)
        msg += not is_complete and f'WARNING Check of {self._unfmt.path.name} failed!' or ''
        msg += self._warn_offset and self.warn_if_offset() or ''
        return msg
