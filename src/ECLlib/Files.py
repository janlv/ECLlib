"""Compatibility layer re-exporting file handlers from the new package layout."""

from .constants import DEBUG, ENDIAN
from .core import DTYPE, File, RefreshIterator, Restart
from .input import DATA_file
from .output import (
    EGRID_file,
    FUNRST_file,
    INIT_file,
    MSG_file,
    PRTX_file,
    PRT_file,
    RFT_file,
    RSM_block,
    RSM_file,
    SMSPEC_file,
    UNRST_file,
    UNSMRY_file,
    fmt_block,
    fmt_file,
    text_file,
)
from .unformatted import unfmt_block, unfmt_file

__all__ = [
    "File",
    "Restart",
    "RefreshIterator",
    "DTYPE",
    "DEBUG",
    "ENDIAN",
    "DATA_file",
    "EGRID_file",
    "INIT_file",
    "UNRST_file",
    "RFT_file",
    "UNSMRY_file",
    "SMSPEC_file",
    "text_file",
    "MSG_file",
    "PRT_file",
    "PRTX_file",
    "fmt_block",
    "fmt_file",
    "FUNRST_file",
    "RSM_block",
    "RSM_file",
    "unfmt_block",
    "unfmt_file",
]
