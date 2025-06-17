# src/ECLlib/__init__.py
__version__ = "0.1.0"

from .Files import (File, DATA_file, EGRID_file, INIT_file, UNRST_file, RFT_file, UNSMRY_file, 
                    SMSPEC_file, text_file, MSG_file, PRT_file, PRTX_file, FUNRST_file, 
                    RSM_file, AFI_file, IXF_file, IX_input, unfmt_file, fmt_file, Restart)
from .File_checker import File_checker
from .GSG import read_GSG, write_GSG, change_resolution