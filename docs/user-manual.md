# ECLlib User Manual

This document introduces the public interface of ECLlib and explains how to work with the
core abstractions, I/O helpers, and utility modules. All examples assume Python 3.10+
(as required by the project) and that the package has been installed into an active
virtual environment.

## Overview

ECLlib is a toolkit for reading, writing, and analysing Eclipse and Intersect simulator
files. It groups the API into three main areas:

- **Core** primitives such as `File` and `Restart` that other modules build upon.
- **I/O** helpers under `ECLlib.io` that parse binary, formatted, and textual artefacts.
- **Utility** functions in `ECLlib.utils` for array manipulation, filesystem access,
  iteration, logging, and time handling.

Import the package to expose the public API defined in `src/ECLlib/__init__.py`:

```python
from ECLlib import (
    File,
    UNRST_file,
    DATA_file,
    EGRID_file,
    RSM_file,
    Progress,
)
```

## Getting Started

```python
from ECLlib import UNRST_file, RFT_file

unrst = UNRST_file("MYCASE")
rft = RFT_file("MYCASE")

# Iterate through simulation steps and pull cell data
for step in unrst.steps():
    swat = step.get("SWAT")  # NumPy array from the current block

# Fetch a report-time series from an RFT file
welldata = rft.blockdata("TIME", "WELLETC", singleton=False)
print(next(welldata))
```

Most parser classes accept either the full filename or just the Eclipse/Intersect case
root. File suffixes are inferred when omitted, and the `role` attribute controls how
instances format themselves when printed.

---

## Core API (`ECLlib.core`)

### `File`

`File` is a convenience wrapper around `pathlib.Path` that centralises filesystem access.
It transparently proxies unknown attributes to the underlying path object and adds
helpers that are reused across the code base:

- `mmap(write=False)` yields a context-managed `mmap` object for zero-copy reads/writes.
- `binarydata(pos=None, raise_error=False)` loads raw bytes optionally slicing by a
  `(start, end)` tuple.
- `as_text(**kwargs)` decodes bytes returned by `binarydata`.
- `resize(start, end)` removes byte ranges in-place and `delete()` unlinks the path.
- `head()`, `tail()`, and `last_line()` are provided through `ECLlib.utils.file_ops` and
  remain available because `File` exposes them via attribute forwarding.

### `Restart`

A frozen `dataclass` storing restart metadata (`start`, `days`, and `step`) extracted from
Eclipse decks or output files. Instances are returned by helpers such as
`DATA_file.restart()`.

### `RefreshIterator`

Wraps an iterator factory that accepts an `only_new` flag. The wrapper restarts the
underlying iterator whenever it is exhausted, making it ideal for monitoring files that
grow over time (for example unformatted restart files).

### Datatype descriptors (`Dtyp`, `DTYPE`, `DTYPE_LIST`)

`ECLlib.core.datatypes` defines the binary layouts that Eclipse uses for unformatted
files. `DTYPE` maps raw header keywords (for example `b"REAL"`) to `Dtyp` descriptors,
while `DTYPE_LIST` exposes the human-readable names. These mappings drive the
unformatted readers described later.

---

## I/O API (`ECLlib.io`)

### Input decks (`ECLlib.io.input`)

#### `DATA_file`

Parses Eclipse input decks and provides high-level access to sections and keywords.
Key capabilities include:

- Enforces a `.DATA` suffix (case sensitive by default, case-insensitive via
  `ignore_suffix_case`).
- `check(include=True, uppercase=False)` verifies that referenced INCLUDE files exist
  and enforces uppercase filenames on Linux when requested.
- `include_files()` and `grid_files()` return generators over referenced files.
- `search(key, regex, comments=False)` and `match()` helpers enable regex-based lookups
  in specific sections.
- `restart()` returns a `Restart` dataclass populated from the deck.
- `read()` and `blockdata()` stream values keyed by the `var_pos` mapping that ships
  with the class, letting you iterate through schedule steps or grid parameters.

#### `AFI_file`

Represents the top-level INTERSECT `.afi` deck. It decodes INCLUDE statements, exposes
referenced `.ixf` files through `ixf_files()`, and recursively loads nested include
files with `included_file_data()`.

#### `IXF_file`, `IXF_node`, and `IX_input`

- `IXF_file` reads INTERSECT `.ixf` files, normalises their contents, and provides the
  `node()` generator for both context and table nodes. Nodes are exposed as `IXF_node`
  objects with helpers such as `.rows()`, `.as_dict()`, `.get()`, and `.update()`.
- `IX_input` orchestrates an entire INTERSECT case. It aggregates the `AFI_file` and
  all referenced `IXF_file` objects, implements case validation through `check()`,
  surfaces grid dimensions via `dim()`, and offers `from_eclipse()` to regenerate
  INTERSECT input using `ecl2ix` when the Eclipse deck changes.

#### GSG helpers (`read_GSG`, `write_GSG`, `change_resolution`)

`ECLlib.io.input.gsgfile` handles Petrel/INTERSECT `.GSG` property files. Use
`read_GSG()` (or the lower-level `read_prop_file()`) to stream `PROP_data` tuples
containing alias, datatype, and NumPy arrays. `write_GSG()` serialises new property
payloads, while `change_resolution(dim, rundir)` rewrites every `.GSG` in a run
directory to a new grid resolution, keeping backups under `GSG_backup`.

### Output readers (`ECLlib.io.output`)

#### Unformatted binaries

`INIT_file`, `UNRST_file`, `RFT_file`, `UNSMRY_file`, and `SMSPEC_file` all inherit from
`unfmt_file` (see below). They expose methods like:

- `blocks(only_new=False)` / `tail_blocks()` for streaming raw binary blocks.
- `blockdata(*keys, singleton=False)` to pull arrays keyed by Eclipse keywords.
- `read(*variables)` for convenience access to entries listed in each class’ `var_pos`
  dictionary.
- Specialised helpers per file type—for example `INIT_file.cell_ijk()` converts cell
  numbers to `(i, j, k)` indices, and `UNRST_file.celldata()` aggregates per-cell time
  series.

`EGRID_file` also builds on `unfmt_file` but focuses on grid geometry. It adds
`nijk()`, `coord_zcorn()`, and `grid()` helpers and can assemble a `pyvista.UnstructuredGrid`
for visualisation.

#### Formatted binaries

`fmt_file` provides formatted (ASCII) block traversal; it yields `fmt_block` objects
with `formatted()`, `as_binary()`, and header metadata. Concrete subclasses include:

- `FUNRST_file` for formatted restart files.
- `RSM_file` alongside its `RSM_block` companion for summary data that pairs with
  `SMSPEC_file` metadata.

#### Textual outputs

`text_file` wraps textual reports such as `.MSG`, `.PRT`, and `.PRTX`. Subclasses set up
regular-expression patterns and converters:

- `MSG_file` extracts report dates, times, and restart steps from XML-style MSG logs.
- `PRT_file` reads report times and exposes `end_time()`.
- `PRTX_file` parses comma-separated PRTX exports, lazily building a column index via
  `var_index()` and exposing `end_time()`.

#### Diagnostics

`File_checker` monitors unformatted files to ensure complete report-step sequences. It
works with any `unfmt_file` instance, tracking matching start/end markers via
`blocks_complete()` and reporting offsets through `warn_if_offset()`.

### Unformatted base layer (`ECLlib.io.unformatted.base`)

The unformatted foundation defines:

- `unfmt_header` and `unfmt_block`, which describe raw block metadata and payloads.
- `unfmt_file`, the core reader/writer matching Eclipse’s 4-byte record markers. It
  implements `blocks()`, `tail_blocks()`, `blockdata()`, `read()`, `fix_errors()`, and
  other low-level routines leveraged by all unformatted subclasses.
- `ENDSOL`, a named tuple used to represent `ENDSOL` keyword occurrences.

---

## Utility Modules (`ECLlib.utils`)

Utilities are re-exported at the package root for backward compatibility, but they are
organised into thematic modules:

- **`array_ops`** – Grid-centric NumPy helpers such as `neighbour_connections()` for
  six-face connectivity, `index_array()` for `(i, j, k)` indices, `cumtrapz()` for
  trapezoidal integration, and `run_length_encode()` to compress arrays.
- **`conversions`** – Unit conversions (`ppm2molL()`, `molL2ppm()`) and `ceildiv()` for
  integer ceiling division.
- **`file_ops`** – Filesystem utilities: `empty_folder()`, `has_write_access()`,
  `head_file()`/`tail_file()`, `read_file()`, `write_file()`, `remove_comments()`, and
  `is_file_ignore_suffix_case()` among others.
- **`iterables`** – Iterator building blocks such as `batched()`, `sliding_window()`,
  `flatten_all()`, `group_indices()`, `index_limits()`, and `take()`.
- **`progress`** – Terminal and file-based progress reporting through `Progress`,
  `Timer`, and `TimerThread`.
- **`string_ops`** – String processing helpers including `decode()`, `float_or_str()`,
  `expand_pattern()`, `split_by_words()`, and `upper_and_lower()`.
- **`system`** – Runtime helpers: `try_except_loop()`, `loop_until()`,
  `kill_process()`, `safezip()`, and environment inspection utilities.
- **`time_utils`** – Date/time helpers (`date_range()`, `dates_after()`, `day2time()`,
  `delta_timestring()`).

Each module exposes an `__all__` list with the supported public names. Import from the
submodule for clarity, e.g.

```python
from ECLlib.utils.array_ops import neighbour_connections
```

---

## Additional Notes

- All readers honour `File.exists(raise_error=True)` to provide meaningful error
  messages when files are missing.
- Many classes accept an `only_new=True` flag to limit iteration to data appended since
  the last read, which is useful in long-running monitoring scripts.
- Functions returning NumPy arrays follow Eclipse’s Fortran ordering; reshape with
  `order="F"` when comparing against simulator output.

Refer to the docstrings inside each module for further details and usage examples.
