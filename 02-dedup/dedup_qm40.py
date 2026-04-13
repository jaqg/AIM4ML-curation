#!/usr/bin/env python3
"""
dedup_qm40.py — Assign compound IDs and rename XYZ files for QM40.

Steps:
    1. Verify no conformers (each Zinc_id appears exactly once in main.csv)
        - If there are no conformers, assign ICONF=1 to all molecules
        - If conformers are found, stop (not yet implemented)
    2. Canonicalise SMILES with RDKit
    3. Compute MD5 hash of canonical SMILES → compound ID
    4. Copy/rename {Zinc_id}.xyz → mol_{ID}_{ICONF}.xyz
    5. Write a mapping CSV: Zinc_id, canonical_SMILES, ID, ICONF, NAT

Usage:
    python3 dedup_qm40.py               # local sample (default)
    python3 dedup_qm40.py --full-data   # full dataset on cluster
"""

import os
import argparse
import hashlib
import shutil
import pandas as pd
from rdkit import Chem

HASH_LENGTH = 12   # characters of MD5 to use as ID (full MD5 = 32)

# --- Paths -------------------------------------------------------------------
PATHS = {
    "sample": {
        "main_csv":    "../../samples/qm40/sample_main.csv",
        "xyz_dir":     "../../samples/qm40/xyz_files",
        "out_dir":     "../../samples/qm40/mol_files",
        "mapping_csv": "../../samples/qm40/qm40_mapping.csv",
        "file_mode":   "copy",
    },
    "full": {
        "main_csv":    "/datos_pool/mldata1/QMdatasets/QM40/main.csv",
        "xyz_dir":     "/home/joseq/AIM4ML/QM40/xyz_files",
        "out_dir":     "/home/joseq/AIM4ML/QM40/mol_files",
        "mapping_csv": "/home/joseq/AIM4ML/QM40/qm40_mapping.csv",
        "file_mode":   "move",
    },
}
# -----------------------------------------------------------------------------


def canonical_smiles(smiles):
    """Return RDKit canonical SMILES, or None if the molecule cannot be parsed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def compound_id(canonical_smi):
    """Return the first HASH_LENGTH characters of the MD5 hash of the canonical SMILES."""
    return hashlib.md5(canonical_smi.encode()).hexdigest()[:HASH_LENGTH]


def parse_args():
    parser = argparse.ArgumentParser(description="Assign compound IDs and rename XYZ files for QM40.")
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Run on the full QM40 dataset on the cluster (default: local sample).",
    )
    return parser.parse_args()


def transfer_file(src, dst, file_mode):
    """Copy or move a file depending on file_mode."""
    if file_mode == "move":
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def main():
    args  = parse_args()
    mode  = "full" if args.full_data else "sample"
    paths = PATHS[mode]

    MAIN_CSV    = paths["main_csv"]
    XYZ_DIR     = paths["xyz_dir"]
    OUT_DIR     = paths["out_dir"]
    MAPPING_CSV = paths["mapping_csv"]
    FILE_MODE   = paths["file_mode"]

    print(f"Mode: {mode}")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Reading {MAIN_CSV} ...")
    main_df = pd.read_csv(MAIN_CSV)

    # -------------------------------------------------------------------------
    # Step 1 — conformer check
    # -------------------------------------------------------------------------

    # First check if there are conformers in QM40 by checking if there are
    # multiple rows for the same Zinc_id (duplicates)
    # If there are no duplicates, assign ICONF=1 to all rows

    # count total number of lines
    n_rows    = len(main_df)
    # count number of unique Zinc_ids 
    n_unique  = main_df["Zinc_id"].nunique()
    # check if there is any Zinc_id duplicated (boolean)
    has_dupes = main_df["Zinc_id"].duplicated().any()

    print(f"\nConformer check:")
    print(f"  Total rows:       {n_rows}")
    print(f"  Unique Zinc_ids:  {n_unique}")
    print(f"  Duplicates found: {has_dupes}")

    if has_dupes:
        # Get the list of Zinc_ids that appear more than once
        # If duplicates are found, warns the user and stops
        dupes = main_df[main_df["Zinc_id"].duplicated(keep=False)]["Zinc_id"].unique()
        print(f"  WARNING: {len(dupes)} Zinc_ids appear more than once.")
        print(f"  Conformer handling is not yet implemented — stopping.")
        return
    else:
        print(f"  OK — no conformers. Assigning ICONF=1 to all molecules.")
        iconf = 1

    # -------------------------------------------------------------------------
    # Steps 2–4 — canonicalise, hash, rename
    # -------------------------------------------------------------------------
    records        = []
    skipped_smiles = []   # RDKit could not parse SMILES
    skipped_xyz    = []   # XYZ file not found in XYZ_DIR

    print(f"\nProcessing {n_rows} molecules ...")
    for _, row in main_df.iterrows():
        zinc_id = row["Zinc_id"]
        smiles  = row["smile"]

        # Step 2 — canonicalise
        can_smi = canonical_smiles(smiles)
        if can_smi is None:
            skipped_smiles.append(zinc_id)
            continue

        # Step 3 — compute compound ID
        cid   = compound_id(can_smi)

        # Step 4 — transfer XYZ file
        src = os.path.join(XYZ_DIR, f"{zinc_id}.xyz")
        dst = os.path.join(OUT_DIR, f"mol_{cid}_{iconf}.xyz")

        if not os.path.exists(src):
            skipped_xyz.append(zinc_id)
            continue

        # Read NAT from line 1 before transferring (src may be moved away)
        with open(src) as f:
            nat = int(f.readline().strip())

        transfer_file(src, dst, FILE_MODE)

        records.append({
            "Zinc_id":          zinc_id,
            "canonical_SMILES": can_smi,
            "ID":               cid,
            "ICONF":            iconf,
            "NAT":              nat,
        })

    # -------------------------------------------------------------------------
    # Step 5 — write mapping CSV
    # -------------------------------------------------------------------------
    mapping_df = pd.DataFrame(records)
    mapping_df.to_csv(MAPPING_CSV, index=False)

    n_skipped = len(skipped_smiles) + len(skipped_xyz)
    print(f"\nDone.")
    print(f"  Processed: {len(records)} molecules")
    print(f"  Skipped:   {n_skipped}")
    print(f"  Mapping:   {MAPPING_CSV}")
    print(f"  XYZ files: {OUT_DIR}/")

    if skipped_smiles:
        print(f"\n  Skipped — RDKit could not parse SMILES ({len(skipped_smiles)}):")
        for z in skipped_smiles:
            print(f"    {z}")

    if skipped_xyz:
        print(f"\n  Skipped — XYZ file not found ({len(skipped_xyz)}):")
        for z in skipped_xyz:
            print(f"    {z}")

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------
    # Check if there are any ID collisions
    n_ids = mapping_df["ID"].nunique()
    if n_ids < len(mapping_df):
        n_collisions = len(mapping_df) - n_ids
        print(f"\n  WARNING: {n_collisions} ID collision(s) detected.")
        print(f"  Consider increasing HASH_LENGTH (currently {HASH_LENGTH}).")
    else:
        print(f"\n  ID collision check: OK — all {n_ids} IDs are unique.")


if __name__ == "__main__":
    main()
