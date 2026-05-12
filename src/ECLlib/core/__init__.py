"""Core primitives used throughout :mod:`ECLlib`."""
from .datatypes import DTYPE, DTYPE_LIST, Dtyp
from .file import File
from .iterators import AutoRefreshIterator
from .restart import Restart
from .unformatted import ENDSOL, unfmt_block, unfmt_file

__all__ = [
    "DTYPE",
    "DTYPE_LIST",
    "Dtyp",
    "File",
    "AutoRefreshIterator",
    "Restart",
    "ENDSOL",
    "unfmt_file",
    "unfmt_block",
]
