# ECLlib

ECLlib is a Python toolkit for reading, writing, and analysing files produced by Schlumberger's Eclipse and Intersect reservoir simulators. It gathers convenience wrappers for both formatted and unformatted data so you can assemble custom workflows for quality control, post-processing, and visualisation.

## Key capabilities

- Parse common Eclipse output files such as `.EGRID`, `.INIT`, `.UNRST`, `.RFT`, `.SMSPEC`, `.UNSMRY`, and more.
- Work with Intersect (`.AFI`, `.IXF`, `.GSG`) and Eclipse (`.DATA`) input files.

## Installation

The project requires Python 3.10 or newer. Installing ECLlib also installs the
`ecl-unrst` command-line tool.

See [INSTALL.md](INSTALL.md) for complete clone, download, Linux/macOS, and Windows
installation instructions.

Quick Linux/macOS path from a local checkout:

```bash
cd ECLlib
chmod +x install.sh
./install.sh
source .venv_ECLlib/bin/activate
```

Quick Windows Command Prompt path from a local checkout:

```bat
cd ECLlib
install.bat
call .venv_ECLlib\Scripts\activate.bat
```

Verify:

```bash
python -c "import ECLlib; print(ECLlib.__version__)"
ecl-unrst --help
```

## `ecl-unrst` command-line tool

`ecl-unrst` is the installed command for one-off unified restart (`.UNRST`) workflows.
Use it from a terminal after installing ECLlib and activating the environment:

```bash
ecl-unrst --help
```

Common commands:

| Command | Purpose |
| --- | --- |
| `ecl-unrst inspect CASE` | Print the `SEQNUM` sections and keys in `CASE.UNRST`. |
| `ecl-unrst merge CASE DONOR SWAT SGAS -o CASE_MERGED` | Write a new UNRST file by copying `SWAT` and `SGAS` from `DONOR.UNRST` into matching sections of `CASE.UNRST`. |
| `ecl-unrst merge CASE records.txt -o CASE_RECORDS` | Write a new UNRST file by reading extra keys from records text input and matching rows by `DOUBHEAD` simulation time. |

`merge` always writes a new output file. It does not modify the input UNRST file. Use
`-o` or `--output` to choose the output path, and add `--overwrite` only when replacing
an existing output file is intended.

`inspect` is read-only. It prints the resolved UNRST path, the number of `SEQNUM`
sections, and the keys in each selected section. Use `--steps` to inspect only selected
report steps:

```bash
ecl-unrst inspect CASE --steps 0 10 20
```

After a successful merge, the CLI prints a short summary:

```text
wrote=CASE_RECORDS.UNRST
mode=records
source=records.txt
keys=WATER_SA,OIL_SATU
sections=12
```

A records text format template is available at
`examples/format_templates/records.txt`. Use it as a reference for preparing the
file passed to `merge`. The file may have any name if it follows that format.

Use the CLI for manual file operations. Use the Python API, for example
`UNRST_file("CASE").merge_keys_from_file(...)`, when the same operation belongs inside
a script, GUI, or larger workflow. See the [UNRST tool guide](docs/unrst-tool.md) for
the full CLI and Python API details.

## Documentation

See the [user manual](docs/user-manual.md) for an overview of the core, I/O, and utility APIs.
For command-line restart workflows, see the [UNRST tool guide](docs/unrst-tool.md).

## Quick start

```python
import numpy as np
from ECLlib import UNRST_file, RFT_file

unrst = UNRST_file('rootname')
rft = RFT_file('rootname')

for block in unrst.blocks():
    if block.key() == 'SWAT':
        swat = block.data()
        break

welldata = rft.blockdata('TIME', 'WELLETC')
print(next(welldata))

# Append one new block to the current last section
unrst.append_blocks(
    step=int(unrst.end_step()),
    keys=('XTEST',),
    blocks=(np.array([1.0], dtype=np.float32),),
)

# Merge donor keys into a new file
unrst.merge_keys_from_file('rootname_donor', keys=('XTEST',))
```

See `src/ECLlib/__init__.py` for a full overview of the public API. The project is licensed under the MIT License and versioned via git tags managed by `setuptools_scm`.
