#!/usr/bin/env python3
"""Inspect and augment unified UNRST files using BlockSpec providers."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path

from ECLlib import UNRST_file


def load_module(reference: str):
    """Load a Python module from a file path or dotted module path."""
    path = Path(reference)
    if path.is_file():
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"Unable to load spec module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(reference)


def iter_sections(unrst: UNRST_file):
    """Yield `(step, section)` pairs from a UNRST file."""
    for section in unrst.section_blocks():
        step = next((int(block.data()[0]) for block in section if block.key() == "SEQNUM"), None)
        if step is None:
            raise SystemExit(f"Section without SEQNUM in {unrst.path}")
        yield step, section


def inspect_command(args):
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


def augment_command(args):
    """Write a new UNRST file using a Python BlockSpec provider."""
    module = load_module(args.spec_module)
    build_blocks = getattr(module, "build_blocks", None)
    if not callable(build_blocks):
        raise SystemExit("Spec module must define build_blocks(step, section)")
    steps = tuple(args.steps) if args.steps is not None else getattr(module, "STEPS", None)
    replace_keys = tuple(args.replace_keys) if args.replace_keys else tuple(getattr(module, "REPLACE_KEYS", ()))
    augmented = UNRST_file(args.input).augment(
        args.output,
        build_blocks,
        steps=steps,
        replace_keys=replace_keys,
        overwrite=args.overwrite,
    )
    print(f"wrote={augmented.path}")


def build_parser():
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="print section keys")
    inspect_parser.add_argument("input", help="input UNRST file or case root")
    inspect_parser.add_argument("--steps", nargs="*", type=int, help="optional SEQNUM filter")
    inspect_parser.set_defaults(func=inspect_command)

    augment_parser = subparsers.add_parser("augment", help="write an augmented UNRST file")
    augment_parser.add_argument("input", help="input UNRST file or case root")
    augment_parser.add_argument("output", help="output UNRST file or case root")
    augment_parser.add_argument("spec_module", help="Python file or dotted module defining build_blocks()")
    augment_parser.add_argument("--steps", nargs="*", type=int, help="optional SEQNUM filter")
    augment_parser.add_argument(
        "--replace-key",
        dest="replace_keys",
        action="append",
        default=[],
        help="existing in-section key to remove before writing new blocks",
    )
    augment_parser.add_argument("--overwrite", action="store_true", help="replace the output file if it exists")
    augment_parser.set_defaults(func=augment_command)

    return parser


def main():
    """Run the command-line tool."""
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
