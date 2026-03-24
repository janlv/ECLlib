#!/usr/bin/env python3
"""Inspect and merge unified UNRST files."""
from __future__ import annotations

import argparse

from ECLlib import UNRST_file


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
    """Merge selected donor keys into a new UNRST file."""
    merged = UNRST_file(args.input).merge_keys_from(
        args.donor,
        keys=tuple(args.keys),
        name=args.output,
        rename=parse_rename_args(args.rename) if args.rename else None,
        overwrite=args.overwrite,
    )
    print(f"wrote={merged.path}")


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

    merge_parser = subparsers.add_parser("merge", help="merge donor solution keys into a new UNRST file")
    merge_parser.add_argument("input", help="input host UNRST file or case root")
    merge_parser.add_argument("donor", help="input donor UNRST file or case root")
    merge_parser.add_argument("keys", nargs="+", help="donor keys to append to each host section")
    merge_parser.add_argument("--output", help="optional output UNRST file or case root")
    merge_parser.add_argument(
        "--rename",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help="optional donor-to-output key rename",
    )
    merge_parser.add_argument("--overwrite", action="store_true", help="replace the output file if it exists")
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
