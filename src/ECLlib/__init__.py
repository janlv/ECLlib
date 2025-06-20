# src/ECLlib/__init__.py

# Version-number generated by setuptools_scm from git tags.
# Do not edit this file manually, it will be overwritten by the build process.
# To update the version, create a new git tag with the format vX.Y.Z (e.g., v1.0.0).
# setuptools_scm will automatically detect this tag and set the version accordingly.
from ._version import version as __version__

# Importing necessary modules and classes from ECLlib
# This allows users to write code like 
#       from ECLlib import DATA_file 
# instead of 
#       from ECLlib.Files import DATA_file
from .Files import (File, DATA_file, EGRID_file, INIT_file, UNRST_file, RFT_file, UNSMRY_file, 
                    SMSPEC_file, text_file, MSG_file, PRT_file, PRTX_file, FUNRST_file, 
                    RSM_file, AFI_file, IXF_file, IX_input, unfmt_file, fmt_file, Restart,
                    unfmt_block)
from .File_checker import File_checker
from .GSG import read_GSG, write_GSG, change_resolution

# Exporting the public API of the ECLlib module
# This is what will be available if ECLlib is imported as 
#       from ECLlib import * 
# but this is not recommended practice.
__all__ = [
    "__version__",
    "File", "DATA_file", "EGRID_file", "INIT_file", "UNRST_file", "RFT_file", "UNSMRY_file",
    "SMSPEC_file", "text_file", "MSG_file", "PRT_file", "PRTX_file", "FUNRST_file",
    "RSM_file", "AFI_file", "IXF_file", "IX_input", "unfmt_file", "fmt_file", "Restart",
    "unfmt_block",
    "File_checker",
    "read_GSG", "write_GSG", "change_resolution",
]
