#!/usr/bin/env python3
"""
00_validate_input.py — Stage 0: Input validator.

Reads the pipeline input SDF, checks that required property tags are present
and well-typed, and emits a clean copy.  This is NOT a curation step — it is
a contract gate.

What it does:
  - Parses the input SDF (sanitize=False — no chemical validation).
  - Checks required tags (Energy_Ha, FormalCharge, Multiplicity) exist and
    have the correct Python type.
  - Warns about missing recommended tags (SMILES, SourceID).
  - Validates optional tags (HOMO_Ha, …) if present.
  - Writes the passing molecules to a clean SDF for stage 1.
  - Writes failing molecules (with original tags) to rejects/00_validate/.
  - Exits 0 if no failures; exits 1 if any molecule is rejected.

Usage:
    python3 00_validate.py input.sdf
    python3 00_validate.py input.sdf -o clean.sdf --rejects-dir rejects/
"""

import os
import sys
import argparse

# Ensure the lib/ directory is importable when the script is run from
# anywhere relative to the scripts folder.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rdkit import Chem
from rdkit.Chem import SDWriter

from lib.schema import REQUIRED_TAGS, RECOMMENDED_TAGS, OPTIONAL_TAGS
from lib.sdf_io import read_sdf, validate_tags


# -- Main ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AIM4ML Stage 0 — Validate input SDF contract."
    )
    p.add_argument("input_sdf", type=str, help="Path to pipeline input SDF.")
    p.add_argument("-o", "--output", type=str, default=None,
                   help="Clean output SDF (default: same name with _valid.sdf).")
    p.add_argument("--rejects-dir", type=str, default="rejects/00_validate",
                   help="Directory for rejected molecules (default: rejects/00_validate/).")
    p.add_argument("--lenient", action="store_true",
                   help="Warn on required-tag failures instead of rejecting.")
    return p.parse_args()


def write_sdf(path, entries):
    """Write a list of (mol, tags) to an SDF file, preserving all tags."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    writer = SDWriter(path)
    for mol, tags in entries:
        # Reinject tags (SDMolSupplier strips them on read; we kept them)
        for name, value in tags.items():
            mol.SetProp(name, value)
        writer.write(mol)
    writer.close()


def main():
    args = parse_args()

    # Resolve output path
    if args.output:
        out_sdf = args.output
    else:
        base, ext = os.path.splitext(args.input_sdf)
        out_sdf = f"{base}_valid{ext}"

    # -- Read -------------------------------------------------------------
    print(f"Reading {args.input_sdf} ...")
    entries = read_sdf(args.input_sdf)
    total = len(entries)
    print(f"  {total} molecules found\n")

    # -- Validate ---------------------------------------------------------
    passed, rejected, warnings = validate_tags(
        entries,
        required=REQUIRED_TAGS,
        recommended=RECOMMENDED_TAGS,
        optional=OPTIONAL_TAGS,
        strict=not args.lenient,
    )

    # -- Write clean SDF --------------------------------------------------
    passed_entries = [entries[i] for i in passed]
    write_sdf(out_sdf, passed_entries)
    print(f"  {len(passed)} valid → {out_sdf}")

    # -- Write rejects -----------------------------------------------------
    if rejected:
        reject_entries = [(mol, tags) for mol, tags, _errs in rejected.values()]
        reject_sdf = os.path.join(args.rejects_dir, "rejected.sdf")
        write_sdf(reject_sdf, reject_entries)
        print(f"  {len(rejected)} rejected → {reject_sdf}")

    # -- Report -----------------------------------------------------------
    print(f"\nReport")
    print(f"  Total:        {total}")
    print(f"  Valid:        {len(passed)}")
    print(f"  Rejected:     {len(rejected)}")
    print(f"  Warnings:     {len(warnings)}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for idx in warnings:
            mol_name = entries[idx][0].GetProp("_Name") if entries[idx][0] is not None else f"entry_{idx}"
            for w in warnings[idx]:
                print(f"  {mol_name} — {w}")

    if rejected:
        print(f"\nFailures ({len(rejected)}):")
        for idx, (mol, _, errs) in rejected.items():
            mol_name = mol.GetProp("_Name") if mol is not None else f"entry_{idx}"
            for e in errs:
                print(f"  {mol_name} — {e}")

    # -- Exit -------------------------------------------------------------
    if rejected:
        print(f"\nValidation FAILED — {len(rejected)} molecule(s) rejected.")
        # Also write an errors file so Makefile/orchestrator can detect failure
        err_path = os.path.join(args.rejects_dir, ".FAILED")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(err_path, "w") as f:
            f.write(f"{len(rejected)} molecule(s) rejected\n")
        sys.exit(1)
    else:
        print(f"\nValidation PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
