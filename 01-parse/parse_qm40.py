#!/usr/bin/env python3
"""
parse_qm40.py — Extract XYZ files from QM40 xyz.csv.

For each molecule in xyz.csv, writes one standard XYZ file named {Zinc_id}.xyz
using the DFT-optimised coordinates (final_x/y/z).

Usage:
    python3 parse_qm40.py               # local sample (default)
    python3 parse_qm40.py --full-data   # full dataset on cluster

Output:
    One .xyz file per molecule in OUTPUT_DIR, named {Zinc_id}.xyz.
"""

import os
import argparse
import pandas as pd
from tqdm import tqdm

# --- Paths -------------------------------------------------------------------
PATHS = {
    "sample": {
        "xyz_csv":    "../../samples/qm40/sample_xyz.csv",
        "output_dir": "../../samples/qm40/xyz_files",
    },
    "full": {
        "xyz_csv":    "/datos_pool/mldata1/QMdatasets/QM40/xyz.csv",
        "output_dir": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/xyz_files",
    },
}
# -----------------------------------------------------------------------------


def write_xyz(path, zinc_id, smiles, atoms, coords):
    """
    Write a standard XYZ file.

    Format:
        <NAT>
        <Zinc_id>  <SMILES>
        <El>  <x>  <y>  <z>
        ...
    """
    nat = len(atoms)
    with open(path, "w") as f:
        f.write(f"{nat}\n")
        f.write(f"{zinc_id}  {smiles}\n")
        for el, (x, y, z) in zip(atoms, coords):
            f.write(f"{el:<3s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Extract XYZ files from QM40 xyz.csv.")
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Run on the full QM40 dataset on the cluster (default: local sample).",
    )
    return parser.parse_args()


def main():
    args   = parse_args()
    mode   = "full" if args.full_data else "sample"
    paths  = PATHS[mode]

    XYZ_CSV    = paths["xyz_csv"]
    OUTPUT_DIR = paths["output_dir"]

    print(f"Mode: {mode}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading {XYZ_CSV} ...")
    df = pd.read_csv(XYZ_CSV)

    groups  = df.groupby("Zinc_id", sort=False)
    n_total = len(groups)
    print(f"Found {n_total} molecules. Writing XYZ files to {OUTPUT_DIR}/")

    for zinc_id, group in tqdm(groups, total=n_total, desc="Writing XYZ", unit="mol"):
        smiles = group["smile"].iloc[0]
        atoms  = group["atom"].tolist()
        coords = list(zip(
            group["final_x"].astype(float),
            group["final_y"].astype(float),
            group["final_z"].astype(float),
        ))

        out_path = os.path.join(OUTPUT_DIR, f"{zinc_id}.xyz")
        write_xyz(out_path, zinc_id, smiles, atoms, coords)

    print(f"\nDone. {n_total} XYZ files written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
