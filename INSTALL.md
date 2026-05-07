# Install ECLlib

This guide installs ECLlib from a local checkout and makes both the Python package and
the `ecl-unrst` command available in the same virtual environment.

## Requirements

- Python 3.10 or newer.
- A local checkout of this repository.

Verify Python:

```bash
python --version
```

On Windows, `py --version` may be the right command if Python was installed with the
Python Launcher.

## Linux And macOS

From the ECLlib repository root:

```bash
python -m venv .venv_ECLlib
source .venv_ECLlib/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Verify the Python package:

```bash
python -c "import ECLlib; print(ECLlib.__version__)"
```

Verify the command-line tool:

```bash
ecl-unrst --help
```

The `ecl-unrst` command does not have to be run from the ECLlib repository folder after
the environment is activated. Run it from the folder where your reservoir files are
located, or pass explicit paths to the input and output files.

When returning to the project later:

```bash
source .venv_ECLlib/bin/activate
```

## Windows

Open PowerShell in the ECLlib repository root:

```powershell
py -m venv .venv_ECLlib
.\.venv_ECLlib\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

If PowerShell blocks activation because script execution is restricted, use Command
Prompt instead:

```bat
py -m venv .venv_ECLlib
call .venv_ECLlib\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e .
```

Verify the Python package:

```powershell
python -c "import ECLlib; print(ECLlib.__version__)"
```

Verify the command-line tool:

```powershell
ecl-unrst --help
```

After activation, `ecl-unrst` can be run from any folder. Use the folder containing your
case files as the current directory, or pass full paths.

When returning later in PowerShell:

```powershell
.\.venv_ECLlib\Scripts\Activate.ps1
```

Or in Command Prompt:

```bat
call .venv_ECLlib\Scripts\activate.bat
```

## Using ECLlib From Python

After installation, import ECLlib in Python:

```python
from ECLlib import UNRST_file, EGRID_file

unrst = UNRST_file("CASE")
print(unrst.count_sections())
```

## Using `ecl-unrst`

Inspect a restart file:

```bash
ecl-unrst inspect CASE
```

Merge keys from another UNRST file:

```bash
ecl-unrst merge CASE DONOR SWAT SGAS -o CASE_MERGED
```

Merge keys from records text input:

```bash
ecl-unrst merge CASE records.txt -o CASE_RECORDS
```

A template for this input format is available at
`examples/format_templates/records.txt`. The input file may have any name if it follows
that format.

## Deactivate

When finished:

```bash
deactivate
```

## Installer Scripts

The repository also includes installer scripts that run the same editable installation
workflow and install the `ecl-unrst` command into `.venv_ECLlib`.

Linux and macOS:

```bash
chmod +x install.sh
./install.sh
```

Windows:

```bat
install.bat
```

After either script finishes, activate the environment and verify the CLI:

```bash
ecl-unrst --help
```
