from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


#---------------------------------------------------------------------------------------------------
def test_ecllib_imports_without_optional_runtime_dependencies():
#---------------------------------------------------------------------------------------------------
    """Import ECLlib and common utilities without removed optional dependencies."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), env.get("PYTHONPATH", "")))
    code = """
import builtins

real_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    blocked = ("molmass", "proclib", "psutil")
    if name in blocked or name.startswith(tuple(f"{module}." for module in blocked)):
        raise ModuleNotFoundError(f"No module named {name!r}")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import
import ECLlib
import ECLlib.utils
from ECLlib import IX_input
from ECLlib.io.input import IX_input as input_IX_input
from ECLlib.utils import batched, ceildiv

assert IX_input is input_IX_input
assert list(batched(range(3), 2)) == [(0, 1), (2,)]
assert ceildiv(5, 2) == 3
print("ok")
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=True,
        text=True,
    )

    assert result.stdout.strip() == "ok"


#---------------------------------------------------------------------------------------------------
def test_ecllib_utils_no_longer_exports_molmass_conversions():
#---------------------------------------------------------------------------------------------------
    """Remove chemistry conversion helpers from the ECLlib utility API."""
    import ECLlib.utils as utils

    assert "ppm2molL" not in utils.__all__
    assert "molL2ppm" not in utils.__all__
    assert not hasattr(utils, "ppm2molL")
    assert not hasattr(utils, "molL2ppm")


#---------------------------------------------------------------------------------------------------
def test_ecllib_no_longer_exports_process_or_conversion_helpers():
#---------------------------------------------------------------------------------------------------
    """Remove process and ecl2ix helpers from the ECLlib public API."""
    import ECLlib.utils as utils
    from ECLlib import IX_input

    assert "kill_process" not in utils.__all__
    assert not hasattr(utils, "kill_process")
    assert not hasattr(IX_input, "from_eclipse")
    assert not hasattr(IX_input, "need_convert")
