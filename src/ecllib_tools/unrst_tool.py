#!/usr/bin/env python3
"""Inspect, merge, and augment unified UNRST files."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from ECLlib import UNRST_file
from .unrst_records_txt import is_records_txt, parse_records_txt


#---------------------------------------------------------------------------------------------------
def iter_sections(unrst: UNRST_file):
#---------------------------------------------------------------------------------------------------
    """Yield ``(step, section)`` pairs from a UNRST file."""
    for section in unrst.section_blocks():
        step = next((int(block.data()[0]) for block in section if block.key() == "SEQNUM"), None)
        if step is None:
            raise SystemExit(f"Section without SEQNUM in {unrst.path}")
        yield step, section


#---------------------------------------------------------------------------------------------------
def parse_rename_args(values):
#---------------------------------------------------------------------------------------------------
    """Return a donor-key rename mapping from ``OLD=NEW`` arguments."""
    return dict(value.split("=", 1) for value in values)


#---------------------------------------------------------------------------------------------------
def _source_suffix(path):
#---------------------------------------------------------------------------------------------------
    """Return a sanitized uppercase suffix derived from an external source path."""
    suffix = re.sub(r"[^A-Z0-9]+", "_", Path(path).stem.upper()).strip("_")
    return suffix or "RECORDS"


#---------------------------------------------------------------------------------------------------
def _counting_rows(rows):
#---------------------------------------------------------------------------------------------------
    """Return a row iterator and mutable count."""
    count = {"value": 0}

    def counted():
        """Yield rows while counting them."""
        for row in rows:
            count["value"] += 1
            yield row

    return counted(), count


#---------------------------------------------------------------------------------------------------
def print_merge_summary(*, outfile, mode, source, keys, sections):
#---------------------------------------------------------------------------------------------------
    """Print a compact merge result summary."""
    print(f"wrote={outfile}")
    print(f"mode={mode}")
    print(f"source={source}")
    print(f"keys={','.join(keys)}")
    print(f"sections={sections}")


#---------------------------------------------------------------------------------------------------
def merge_records_to_unrst(input_file, source_file, *, output=None, overwrite=False, time_tolerance=1e-6):
#---------------------------------------------------------------------------------------------------
    """Return output path, keys, and section count for a records text merge."""
    unrst = UNRST_file(input_file)
    unrst.exists(raise_error=True)
    outfile = (
        UNRST_file(output).path
        if output
        else unrst.path.with_name(f"{unrst.path.stem}_{_source_suffix(source_file)}.UNRST")
    )
    if outfile == unrst.path:
        raise ValueError("Records text merge output path must differ from the input UNRST path")
    if outfile.exists() and not overwrite:
        raise FileExistsError(f"{outfile} already exists; pass --overwrite to replace it")
    if time_tolerance <= 0:
        raise ValueError("--time-tolerance must be positive")
    if not is_records_txt(source_file):
        raise ValueError(
            "Unsupported merge source format; expected records text starting with "
            "'NBLOCK <count>' and 'TIME = <value> d'"
        )

    keys, rows = parse_records_txt(source_file)
    counted_rows, count = _counting_rows(rows)
    merged = unrst.merge_keys_from_blocks(
        keys=keys,
        rows=counted_rows,
        name=outfile,
        overwrite=overwrite,
        tolerance=time_tolerance,
    )
    return merged.path, keys, count["value"]


#---------------------------------------------------------------------------------------------------
def inspect_command(args):
#---------------------------------------------------------------------------------------------------
    """Print section keys for a UNRST file."""
    unrst = UNRST_file(args.input)
    selected = set(args.steps) if args.steps else None
    print(f"file={unrst.path}")
    print(f"sections={unrst.count_sections()}")
    shown = 0
    for step, section in iter_sections(unrst):
        if selected is not None and step not in selected:
            continue
        shown += 1
        keys = ", ".join(block.key() for block in section)
        print(f"step={step} keys={keys}")
    if selected is not None and shown == 0:
        raise SystemExit(f"No matching steps found in {unrst.path}")


#---------------------------------------------------------------------------------------------------
def merge_command(args):
#---------------------------------------------------------------------------------------------------
    """Merge donor UNRST keys or records text rows into a new UNRST file."""
    if is_records_txt(args.source):
        if args.keys:
            raise ValueError("Records text merge does not take key arguments")
        if args.rename:
            raise ValueError("--rename only applies to donor UNRST merges")
        outfile, keys, sections = merge_records_to_unrst(
            args.input,
            args.source,
            output=args.output,
            overwrite=args.overwrite,
            time_tolerance=args.time_tolerance or 1e-6,
        )
        print_merge_summary(
            outfile=outfile,
            mode="records",
            source=args.source,
            keys=keys,
            sections=sections,
        )
        return

    if not args.keys:
        raise ValueError("Merge source is not records text; donor UNRST merge requires KEY [KEY ...]")
    if args.time_tolerance is not None:
        raise ValueError("--time-tolerance only applies to records text merges")
    merged = UNRST_file(args.input).merge_keys_from_file(
        args.source,
        keys=tuple(args.keys),
        name=args.output,
        rename=parse_rename_args(args.rename) if args.rename else None,
        overwrite=args.overwrite,
    )
    print_merge_summary(
        outfile=merged.path,
        mode="donor",
        source=UNRST_file(args.source).path,
        keys=tuple(args.keys),
        sections=UNRST_file(args.input).count_sections(),
    )


#---------------------------------------------------------------------------------------------------
def build_parser():
#---------------------------------------------------------------------------------------------------
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="print section keys")
    inspect_parser.add_argument("input", help="input UNRST file or case root")
    inspect_parser.add_argument("--steps", nargs="*", type=int, help="optional SEQNUM filter")
    inspect_parser.set_defaults(func=inspect_command)

    merge_parser = subparsers.add_parser("merge", help="merge donor UNRST keys or records text rows")
    merge_parser.add_argument("input", help="input host UNRST file or case root")
    merge_parser.add_argument("source", help="donor UNRST file/case root or records text source")
    merge_parser.add_argument("keys", nargs="*", help="donor keys to append to each host section")
    merge_parser.add_argument("-o", "--output", help="optional output UNRST file or case root")
    merge_parser.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="optional donor-to-output key rename",
    )
    merge_parser.add_argument("--overwrite", action="store_true", help="replace the output file if it exists")
    merge_parser.add_argument(
        "--time-tolerance",
        type=float,
        default=None,
        help="absolute day tolerance for matching external rows to DOUBHEAD time",
    )
    merge_parser.set_defaults(func=merge_command)

    return parser


#---------------------------------------------------------------------------------------------------
def main():
#---------------------------------------------------------------------------------------------------
    """Run the command-line tool."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
