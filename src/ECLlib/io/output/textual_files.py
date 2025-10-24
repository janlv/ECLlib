"""
Eclipse/INTERSECT textual output files

Text-based Eclipse output files are written in plain ASCII and provide human-readable 
diagnostic and runtime information from the simulator. These files complement the 
binary (unformatted) outputs and are primarily used for monitoring, debugging, and 
audit purposes. They typically follow the same naming convention as binary files, 
sharing the same root name as the simulation case.

File types:
- MSG: General message log containing warnings, notes, and simulation status updates.
- PRT: Main printer output file containing a full textual record of the run, including 
  summary tables, convergence data, timestep reports, and keyword echoes.
- PRTX (or .dbprtx): Structured message database used by Petrel’s message service; 
  contains machine-readable event logs and metadata linked to simulation entities.
"""

from datetime import datetime
from re import findall

from ...core import File
from ...utils import matches

__all__ = ["text_file", "MSG_file", "PRT_file", "PRTX_file"]

#==================================================================================================
class text_file(File):                                                                  # text_file
#==================================================================================================
    """Base class for textual output helpers."""
    #----------------------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                                 # text_file
    #----------------------------------------------------------------------------------------------
        """Initialize the text_file."""
        super().__init__(file, **kwargs)
        self._pattern = {} 
        self._convert = {}
        # self._flavor = None          # 'ecl' or 'ix'

    #----------------------------------------------------------------------------------------------
    def __contains__(self, key):                                                        # text_file
    #----------------------------------------------------------------------------------------------
        """Return whether the value exists."""
        return key.encode() in self.binarydata()
        
    #----------------------------------------------------------------------------------------------
    def contains_any(self, *keys, head=None, tail=None):                                # text_file
    #----------------------------------------------------------------------------------------------
        """Return whether any pattern exists in the file."""
        if head:
            data = self.head(size=head)
        elif tail:
            data = self.tail(size=tail)
        else:
            data = self.binarydata()
            keys = (key.encode() for key in keys)
        return any(key in data for key in keys)

    #----------------------------------------------------------------------------------------------
    def read(self, *var_list):                                                          # text_file
    #----------------------------------------------------------------------------------------------
        """Read the complete file contents."""
        values = []
        for var in var_list:
            match = matches(file=self.path, pattern=self._pattern[var])
            values.append([self._convert[var](m.group(1)) for m in match])
        return list(zip(*values))



#==================================================================================================
class MSG_file(text_file):                                                               # MSG_file
#==================================================================================================
    """
    MSG (Message File)
    Plain-text log file containing messages from the simulator during execution.
    Includes informational messages, warnings, and error summaries for each run step.
    Useful for quick review of runtime issues or keyword-related warnings.
    """

    #----------------------------------------------------------------------------------------------
    def __init__(self, file):                                                            # MSG_file
    #----------------------------------------------------------------------------------------------
        """Initialize the MSG_file."""
        super().__init__(file, suffix='.MSG')
        #'time' : r'<\s*\bmessage\b\s+\bdate\b="[0-9/]+"\s+time="([0-9.]+)"\s*>',
        self._pattern = {'date' : r'<message date="([0-9/]+)"',
                         'time' : r'<message date="[0-9/]+" time="([0-9.]+)"',
                         'step' : r'\bRESTART\b\s+\bFILE\b\s+\bWRITTEN\b\s+\bREPORT\b\s+([0-9]+)'}
        self._convert = {'date' : lambda x: datetime.strptime(x.decode(),'%d/%m/%Y'),
                         'time' : float,
                         'step' : int}



#==================================================================================================
class PRT_file(text_file):                                                               # PRT_file
#==================================================================================================
    """
    PRT (Printer Output File)
    Primary human-readable report generated by the simulator.
    Includes a complete textual record of the run: input echo, keyword processing,
    timestep summaries, material balance tables, convergence diagnostics, and
    overall performance statistics. Typically the first file to inspect after a run.
    """

    #----------------------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                                  # PRT_file
    #----------------------------------------------------------------------------------------------
        """Initialize the PRT_file."""
        super().__init__(file, suffix='.PRT', **kwargs)
        self._pattern['time'] = r'TIME(?:[ a-zA-Z\s/%-]+;|=) +([\d.]+)'
        #self._pattern['time'] = r' (?:Rep    ;|Init   ;|TIME=)\s*([0-9.]+)\s+'
        self._pattern['days'] = self._pattern['time']
        self._convert = {key:float for key in self._pattern}

    #----------------------------------------------------------------------------------------------
    def end_time(self):                                                                  # PRT_file
    #----------------------------------------------------------------------------------------------
        """Return the final timestamp."""
        chunks = (txt for txt in self.reversed(size=10*1024) if 'TIME' in txt)
        if data:=next(chunks, None):
            days = findall(self._pattern['time'], data)
            return float(days[-1]) if days else 0
        return 0



#==================================================================================================
class PRTX_file(text_file):                                                             # PRTX_file
#==================================================================================================
    """
    PRTX (Petrel Message Database)
    Structured message output file (often with `.dbprtx` extension) used by Petrel's
    message service framework. Contains messages and metadata in a machine-readable
    format, allowing Petrel and other SLB tools to map messages to simulation objects
    (e.g., wells, regions, or grids). Not intended for manual editing or direct viewing.
    """

    #----------------------------------------------------------------------------------------------
    def __init__(self, file, **kwargs):                                                 # PRTX_file
    #----------------------------------------------------------------------------------------------
        """Initialize the PRTX_file."""
        super().__init__(file, suffix='.PRTX', **kwargs)
        self._var_index = {}

    #----------------------------------------------------------------------------------------------
    def var_index(self):                                                                # PRTX_file
    #----------------------------------------------------------------------------------------------
        """Return the index for the requested variable."""
        if not self._var_index:
            names = next(self.lines(), '').split(',')
            self._var_index = {name:i for i,name in enumerate(names)}
        return self._var_index

    #----------------------------------------------------------------------------------------------
    def end_time(self):                                                                 # PRTX_file
    #----------------------------------------------------------------------------------------------
        """
        Note that time in PRTX seems to be delayed compared to PRT and RFT
        """
        time = 0
        if (line:=self.last_line()) and (index:=self.var_index()):
            time = line.split(',')[index['Simulation Time']]
            time = float(time) if time[0] != 'S' else 0
        return time
