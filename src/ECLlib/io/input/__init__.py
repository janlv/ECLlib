"""Input-related helpers for ECLlib."""

from .eclipse import DATA_file
from .intersect import AFI_file, IXF_file, IX_input
from .gsgfile import PROP_data, read_prop_file, write_prop_file, change_resolution, read_GSG, write_GSG

__all__ = ["DATA_file", 
           "AFI_file", "IXF_file", "IX_input",
           "PROP_data", "read_prop_file", "write_prop_file",
           "change_resolution", "read_GSG", "write_GSG"]
