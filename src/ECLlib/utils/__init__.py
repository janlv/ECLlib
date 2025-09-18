"""Utility helpers aggregated for backward compatibility."""
from __future__ import annotations

from . import array_ops as _array_ops
from . import conversions as _conversions
from . import file_ops as _file_ops
from . import iterables as _iterables
from . import progress as _progress
from . import string_ops as _string_ops
from . import system as _system
from . import time_utils as _time_utils

from .array_ops import *  # noqa: F401,F403
from .conversions import *  # noqa: F401,F403
from .file_ops import *  # noqa: F401,F403
from .iterables import *  # noqa: F401,F403
from .progress import *  # noqa: F401,F403
from .string_ops import *  # noqa: F401,F403
from .system import *  # noqa: F401,F403
from .time_utils import *  # noqa: F401,F403

__all__ = (
    _array_ops.__all__
    + _conversions.__all__
    + _file_ops.__all__
    + _iterables.__all__
    + _progress.__all__
    + _string_ops.__all__
    + _system.__all__
    + _time_utils.__all__
)

__all__ = sorted(set(__all__))

__modules__ = {
    "array_ops": _array_ops,
    "conversions": _conversions,
    "file_ops": _file_ops,
    "iterables": _iterables,
    "progress": _progress,
    "string_ops": _string_ops,
    "system": _system,
    "time_utils": _time_utils,
}

