#!/usr/bin/env python3
"""
01_split.py — Stage 1: Split validated SDF into Parquet batches.

Reads the output of Stage 0 (a clean SDF with all required tags validated),
extracts the SDF V2000 mol block and all metadata columns, and writes
Parquet batch files ready for downstream curation stages.

The mol_block column stores the raw V2000 text block (atoms + bonds + M END)
for each molecule.  Metadata tags are stored in typed Parquet columns.

Usage:
    python3 01_split.py input_valid.sdf -o batches/ --batch-size 5000
"""

import os
import sys
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rdkit import Chem
from rdkit.Chem import rdchem

from lib.schema import PARQUET_COLUMNS
from lib.parquet_io import write_batch


# -- SDF → row extraction ------------------------------------------------

def extract_row(mol):
    """
    Extract a Parquet row dict from a validated RDKit molecule.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        A validated molecule with all required SDF tags.

    Returns
    -------
    dict
        Keys match PARQUET_COLUMNS.
    """
    block_full = Chem.MolToMolBlock(mol)
    m_end = block_full.index("M  END") + len("M  END")
    mol_block = block_full[:m_end]

    row = {
        "CompoundID":   None,         # computed in stage 4 (dedup)
        "mol_block":    mol_block,
        "num_atoms":    mol.GetNumAtoms(),
        "num_bonds":    mol.GetNumBonds(),
    }

    # Required tags — guaranteed by Stage 0
    row["Energy_Ha"]    = float(mol.GetProp("Energy_Ha"))
    row["FormalCharge"] = int(mol.GetProp("FormalCharge"))
    row["Multiplicity"] = int(mol.GetProp("Multiplicity"))

    # Recommended + optional — safe getter
    def _get(tag, cast=None):
        try:
            val = mol.GetProp(tag)
            return cast(val) if cast else val
        except KeyError:
            return None

    row["SMILES"]         = _get("SMILES")
    row["SourceID"]       = _get("SourceID")
    row["HOMO_Ha"]        = _get("HOMO_Ha", float)
    row["LUMO_Ha"]        = _get("LUMO_Ha", float)
    row["HL_Gap_Ha"]      = _get("HL_Gap_Ha", float)
    row["PartialCharges"] = _get("PartialCharges")

    return row


# -- Main ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AIM4ML Stage 1 — Split validated SDF into Parquet batches."
    )
    p.add_argument("input_sdf", type=str, help="Validated input SDF (Stage 0 output).")
    p.add_argument("-o", "--output-dir", type=str, default="batches",
                   help="Directory for Parquet batch files (default: batches/).")
    p.add_argument("-b", "--batch-size", type=int, default=5000,
                   help="Molecules per batch (default: 5000).")
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -- Read SDF ---------------------------------------------------------
    print(f"Reading {args.input_sdf} ...")
    supplier = Chem.SDMolSupplier(args.input_sdf, sanitize=False, removeHs=False)
    mols = [m for m in supplier if m is not None]
    total = len(mols)
    print(f"  {total} molecules loaded.")

    # -- Split into batches ------------------------------------------------
    batch_num = 0
    batch_rows = []
    written = 0

    for mol in mols:
        row = extract_row(mol)
        batch_rows.append(row)

        if len(batch_rows) >= args.batch_size:
            path = os.path.join(args.output_dir, f"batch_{batch_num:04d}.parquet")
            write_batch(path, batch_rows)
            written += len(batch_rows)
            print(f"  Wrote {path} ({len(batch_rows)} mols)")
            batch_rows = []
            batch_num += 1

    # Flush last partial batch
    if batch_rows:
        path = os.path.join(args.output_dir, f"batch_{batch_num:04d}.parquet")
        write_batch(path, batch_rows)
        written += len(batch_rows)
        print(f"  Wrote {path} ({len(batch_rows)} mols)")

    # -- Report ------------------------------------------------------------
    n_batches = batch_num + (1 if batch_rows else 0) if total else 0
    print(f"\nReport")
    print(f"  Total molecules:  {total}")
    print(f"  Written:          {written}")
    print(f"  Batches:          {n_batches}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Output dir:       {args.output_dir}")


if __name__ == "__main__":
    main()
