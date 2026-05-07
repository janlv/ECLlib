# Merge And Append UNRST Keys

## Quick Access

There are two ways to use these features:

| Use case | Use this | What it means |
| --- | --- | --- |
| You want to run a command in a terminal | CLI: `ecl-unrst` | No Python code is needed. |
| You are writing a Python script or application | Python API: `UNRST_file(...)` | Import ECLlib and call methods from Python. |

### CLI: Terminal Commands

Use the CLI for one-off file operations after installing ECLlib and activating the
environment:

```bash
ecl-unrst --help
ecl-unrst inspect CASE
ecl-unrst merge CASE DONOR SWAT SGAS -o CASE_MERGED
ecl-unrst merge CASE records.txt -o CASE_RECORDS
```

Use `merge` for both completed-file workflows. If the source is records text, the tool
detects that format automatically. Otherwise it treats the source as a donor UNRST and
requires key names. `merge` writes a new output file; it does not modify `CASE.UNRST`.
Add `--overwrite` only when the output file already exists and should be replaced.

A sample `records.txt` format template is available at
`examples/format_templates/records.txt`.

### Python API: Code Usage

Use the Python API when this is part of a script, workflow, GUI, or another application:

```python
from ECLlib import UNRST_file

UNRST_file("CASE").merge_keys_from_file(
    "DONOR",
    keys=("SWAT", "SGAS"),
    name="CASE_MERGED",
)
```

Python methods use `name=...` for the output file. CLI commands use `-o` or `--output`.

ECLlib can write new unified restart (`.UNRST`) files by appending new solution blocks
to existing `SEQNUM` sections. Prefer the Python API when building scripts or
applications. Use the CLI for one-off donor-file or records text merges.

There are two different workflows:

| Workflow | Method | Changes input file | Typical use |
| --- | --- | --- | --- |
| Append to an active UNRST | `append_blocks()` | Yes | Add keys to the current last section while a workflow is still producing restart data. |
| Update a completed UNRST copy | `merge_keys_from_file()` or `merge_keys_from_blocks()` | No | Add extra keys to every matching `SEQNUM` section in a finished restart file. |

## Python API

Import the restart reader with the case root. Full `.UNRST` paths are accepted, but the
case root keeps calls consistent with the rest of ECLlib:

```python
from ECLlib import UNRST_file, unfmt_block

unrst = UNRST_file("CASE")
```

### Append Keys To An Active UNRST

Use `append_blocks()` when you want to modify the current UNRST file in place. This is
for active or incremental workflows where only the last completed section should receive
new keys.

```python
import numpy as np
from ECLlib import UNRST_file

unrst = UNRST_file("ACTIVE")
unrst.append_blocks(
    step=int(unrst.end_step()),
    keys=("XAPP",),
    blocks=(np.array([1.0, 2.0], dtype=np.float32),),
)
```

Important limits:

- The input file is modified in place.
- Only the last section can be updated.
- The requested `step` must match the current last `SEQNUM`.
- This is not a full-file merge operation.

### Merge Keys From Another UNRST

Use `merge_keys_from_file()` for a completed UNRST file when the new data already exists as
blocks in another UNRST file. This writes a new output file with selected donor keys
inserted into each host section.

```python
merged = UNRST_file("CASE").merge_keys_from_file(
    "DONOR",
    keys=("SWAT", "SGAS"),
    name="CASE_MERGED",
    overwrite=True,
)
```

This writes a new file. The host and donor files are not modified. Donor sections are
paired with host sections in file order.

`overwrite=True` means the output path may already exist and will be replaced. Leave it
as `False` to fail instead of replacing an existing output file.

Rename donor keys while merging:

```python
UNRST_file("CASE").merge_keys_from_file(
    "DONOR",
    keys=("TEMP",),
    rename={"TEMP": "TEMP_IOR"},
    name="CASE_TEMP",
)
```

### Merge Serialized Blocks By Simulation Time

Use `merge_keys_from_blocks()` for a completed UNRST file when the new data is available
as serialized unformatted blocks and should be matched by `DOUBHEAD[0]` simulation day.
This also writes a new output file.

```python
keys = ("WATER_SA", "OIL_SATU")
rows = (
    (
        0.0,
        (
            unfmt_block.from_data("WATER_SA", [0.10, 0.20], "float").as_bytes(),
            unfmt_block.from_data("OIL_SATU", [0.90, 0.80], "float").as_bytes(),
        ),
    ),
    (
        10.0,
        (
            unfmt_block.from_data("WATER_SA", [0.11, 0.21], "float").as_bytes(),
            unfmt_block.from_data("OIL_SATU", [0.89, 0.79], "float").as_bytes(),
        ),
    ),
)

out = UNRST_file("CASE").merge_keys_from_blocks(
    keys=keys,
    rows=rows,
    name="CASE_RECORDS",
    tolerance=1e-6,
)
```

`rows` may be a generator. Each row is `(time, block_row)`, where `block_row[j]`
contains serialized bytes for `keys[j]`. Rows must be ordered by increasing simulation
time in the same order as the target UNRST sections.

The method writes a new file, leaves the input unchanged, and removes partial output on
failure. Missing or ambiguous time matches fail. Duplicate target keys in a matched
section also fail.

## CLI

The CLI supports the completed-file workflows through one `merge` command. It always
writes a new output file; it does not expose the in-place active-file `append_blocks()`
operation.

Install ECLlib first. See [the install guide](../INSTALL.md) for Linux, macOS, and
Windows setup.

Then run:

```bash
ecl-unrst --help
```

All CLI commands accept either a case root or a full `.UNRST` path. Prefer the case root
unless you specifically need to point at a non-standard filename.

`inspect` is read-only. It prints the resolved file path, total section count, and the
keys in each selected `SEQNUM` section. Use `--steps` to limit the output.

| Command | Source type | Purpose | Writes a new file |
| --- | --- | --- | --- |
| `merge INPUT DONOR KEY [KEY ...]` | Donor UNRST | Append selected donor blocks into host sections. | Yes |
| `merge INPUT SOURCE` | Records text | Append parsed blocks into sections matched by `DOUBHEAD` time. | Yes |

### CLI Merge From Another UNRST

```bash
ecl-unrst merge INPUT DONOR KEY [KEY ...] [-o OUTPUT] [--rename OLD=NEW] [--overwrite]
```

Example:

```bash
ecl-unrst merge CASE DONOR SWAT SGAS -o CASE_MERGED
```

`-o` and `--output` are equivalent. `--overwrite` allows the output path to replace an
existing file. Without `--overwrite`, the command fails if the output path already exists.
`--rename` only applies to donor UNRST merges.

### CLI Merge Records Text

```bash
ecl-unrst merge INPUT SOURCE [-o OUTPUT] [--overwrite] \
    [--time-tolerance DAYS]
```

Example:

```bash
ecl-unrst merge CASE records.txt -o CASE_RECORDS
```

`-o` and `--output` are equivalent. `--overwrite` has the same meaning here: replace the
output file if it already exists. It never means modifying `INPUT`; the original UNRST
file is not overwritten. `--time-tolerance` only applies to records text merges.

Successful merges print a compact summary:

```text
wrote=CASE_RECORDS.UNRST
mode=records
source=records.txt
keys=WATER_SA,OIL_SATU
sections=12
```

## Records Text Format

The CLI `merge` command detects records text input with this shape:

```text
NBLOCK 2

TIME = 0 d

WATER SATURATIONS
0.10 0.20

OIL SATURATIONS
0.90 0.80
```

The repository includes the same shape as a template at
`examples/format_templates/records.txt`. The input file may have any name if it follows
this format.

Rules:

- First non-empty line: `NBLOCK <count>`.
- Each group starts with `TIME = <value> d`.
- Headings become UNRST keys by replacing spaces with `_` and truncating to 8 characters.
- Each heading must provide exactly `NBLOCK` numeric values.
- The file is assumed to be machine written with the same headings in the same order for every `TIME`.

Examples:

| Heading | UNRST key |
| --- | --- |
| `WATER SATURATIONS` | `WATER_SA` |
| `OIL SATURATIONS` | `OIL_SATU` |
| `GAS SATURATIONS` | `GAS_SATU` |

Additional source formats can be added later as separate script-side adapters that
produce `(keys, rows)` for `UNRST_file.merge_keys_from_blocks()`.
