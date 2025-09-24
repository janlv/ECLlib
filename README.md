# ECLlib

ECLlib is a Python toolkit for reading, writing, and analysing files produced by Schlumberger's Eclipse and Intersect reservoir simulators. It gathers convenience wrappers for both formatted and unformatted data so you can assemble custom workflows for quality control, post-processing, and visualisation.

## Key capabilities

- Parse common Eclipse output files such as `.EGRID`, `.INIT`, `.UNRST`, `.RFT`, `.SMSPEC`, `.UNSMRY`, and more.
- Work with Intersect (`.AFI`, `.IXF`, `.GSG`) and Eclipse (`.DATA`) input files.

## Installation

The project requires Python 3.10 or newer. The supplied installation scripts create an isolated virtual environment (`venv`), activate it, and install ECLlib together with its dependencies via `pip install .`.

### Linux and macOS

1. **Make the script executable** (first run only):
   ```bash
   chmod +x install.sh
   ```
2. **Run the installation** from the project root:
   ```bash
   ./install.sh
   ```

The script determines the package name from `pyproject.toml`, provisions a `.venv_ECLlib` directory, activates the environment, and installs the project in editable mode.

### Windows

1. Open *Command Prompt* or *PowerShell*.
2. Change into the project root if required using `cd`.
3. Run the installer:
   ```bat
   install.bat
   ```

The batch script mirrors the UNIX workflow by creating a `.venv_ECLlib` environment, activating it for the duration of the session, and installing the project with `pip install .`.

### After installation

- **Reactivate the environment** whenever you return to the project:
  - Linux/macOS:
    ```bash
    source .venv_ECLlib/bin/activate
    ```
  - Windows:
    ```bat
    call .venv_ECLlib\Scripts\activate.bat
    ```
- **Deactivate the environment** when you are done:
  ```bash
  deactivate
  ```

## Quick start

```python
from ECLlib import UNRST_file, RFT_file

unrst = UNRST_file('rootname')
rft = RFT_file('rootname')

for block in unrst.blocks():
    if block.key() == 'SWAT':
        swat = block.data()
        break

welldata = rft.blockdata('TIME', 'WELLETC')
print(next(welldata))
```

See `src/ECLlib/__init__.py` for a full overview of the public API. The project is licensed under the MIT License and versioned via git tags managed by `setuptools_scm`.
