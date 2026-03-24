#!/usr/bin/env python3
"""Benchmark UNRST merge and append throughput on synthetic files."""
from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from ECLlib import UNRST_file, unfmt_block


#---------------------------------------------------------------------------------------------------
def _write_unrst(path: Path, *, sections: int, cells: int, donor: bool = False):
#---------------------------------------------------------------------------------------------------
    """Write a synthetic UNRST fixture for merge and append benchmarks."""
    with open(path, "wb") as file:
        for step in range(sections):
            temp = np.arange(cells, dtype=np.float32) + step
            pressure = np.arange(cells, dtype=np.float32) + 100.0 + step
            specs = [
                ("SEQNUM", [step], "int"),
                ("STARTSOL", [], "mess"),
                ("TEMP", temp, "float"),
                ("PRESSURE", pressure, "float"),
            ]
            if donor:
                specs.extend(
                    [
                        ("KEY1", temp + 0.5, "float"),
                        ("KEY2", np.arange(cells, dtype=np.int32) + step, "int"),
                    ]
                )
            specs.append(("ENDSOL", [], "mess"))
            for key, data, dtype in specs:
                file.write(unfmt_block.from_data(key, data, dtype).as_bytes())


#---------------------------------------------------------------------------------------------------
def _format_rate(bytes_processed: int, elapsed: float):
#---------------------------------------------------------------------------------------------------
    """Return a simple MiB/s rate string."""
    rate = 0.0 if elapsed <= 0 else bytes_processed / elapsed / (1024 * 1024)
    return f"{rate:0.2f} MiB/s"


#---------------------------------------------------------------------------------------------------
def _benchmark_merge(root: Path, *, sections: int, cells: int):
#---------------------------------------------------------------------------------------------------
    """Benchmark donor merge throughput on synthetic UNRST files."""
    host = root / "host.UNRST"
    donor = root / "donor.UNRST"
    out = root / "merged.UNRST"
    _write_unrst(host, sections=sections, cells=cells)
    _write_unrst(donor, sections=sections, cells=cells, donor=True)
    bytes_processed = host.stat().st_size + donor.stat().st_size
    start = perf_counter()
    UNRST_file(host).merge_keys_from(donor, keys=("KEY1", "KEY2"), name=out, overwrite=True)
    elapsed = perf_counter() - start
    return elapsed, bytes_processed, out.stat().st_size


#---------------------------------------------------------------------------------------------------
def _benchmark_append(root: Path, *, sections: int, cells: int):
#---------------------------------------------------------------------------------------------------
    """Benchmark last-section append throughput on a synthetic UNRST file."""
    host = root / "append.UNRST"
    _write_unrst(host, sections=sections, cells=cells)
    unrst = UNRST_file(host)
    before = host.stat().st_size
    start = perf_counter()
    unrst.append_blocks(
        step=sections - 1,
        keys=("XAPP", "IAPP"),
        blocks=(
            np.arange(cells, dtype=np.float32) + 0.25,
            np.arange(cells, dtype=np.int32),
        ),
    )
    elapsed = perf_counter() - start
    return elapsed, host.stat().st_size - before


#---------------------------------------------------------------------------------------------------
def build_parser():
#---------------------------------------------------------------------------------------------------
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections", type=int, default=100, help="number of report sections to generate")
    parser.add_argument("--cells", type=int, default=20000, help="array length per solution block")
    return parser


#---------------------------------------------------------------------------------------------------
def main():
#---------------------------------------------------------------------------------------------------
    """Run the synthetic merge and append benchmarks."""
    args = build_parser().parse_args()
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        merge_elapsed, merge_bytes, merged_size = _benchmark_merge(root, sections=args.sections, cells=args.cells)
        append_elapsed, append_bytes = _benchmark_append(root, sections=args.sections, cells=args.cells)

    print(
        f"merge:  sections={args.sections} cells={args.cells} "
        f"elapsed={merge_elapsed:0.3f}s throughput={_format_rate(merge_bytes, merge_elapsed)} "
        f"output={merged_size / (1024 * 1024):0.2f} MiB"
    )
    print(
        f"append: sections={args.sections} cells={args.cells} "
        f"elapsed={append_elapsed:0.3f}s throughput={_format_rate(append_bytes, append_elapsed)} "
        f"added={append_bytes / (1024 * 1024):0.2f} MiB"
    )


if __name__ == "__main__":
    main()
