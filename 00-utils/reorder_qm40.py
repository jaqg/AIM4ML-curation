#!/usr/bin/env python3
"""
reorder_qm40.py — AMBER canonical atom reordering for QM40 mol_*.xyz files (D17).

For each molecule in qm40_mapping.csv, calls xyz_reord_fixed.sh (AMBER antechamber).
On success the XYZ is overwritten in-place with the reordered geometry.
On failure the XYZ (renamed to *_ORIG by the shell script) and the corresponding
SDF are removed, and the molecule is logged as discarded.

Adds a 'reorder_status' column to qm40_mapping.csv:
    success  — reordering succeeded
    failed   — antechamber could not process the structure (discarded)
    missing  — mol_*.xyz not found before reordering (should not happen)

Requires: AMBER23 (antechamber) and OpenBabel installed on the cluster.

Usage:
    python3 reorder_qm40.py               # local sample (default)
    python3 reorder_qm40.py --full-data   # full dataset on cluster
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REORD_SCRIPT = os.path.join(SCRIPT_DIR, "xyz_reord_qm40.sh")
ANTECHAMBER  = "/opt/amber23/bin/antechamber"

PATHS = {
    "sample": {
        "mol_dir":     "../../samples/qm40/mol_files",
        "mapping_csv": "../../samples/qm40/qm40_mapping.csv",
        "log_dir":     "../../samples/qm40/logs",
    },
    "full": {
        "mol_dir":     "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/mol_files",
        "mapping_csv": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/qm40_mapping.csv",
        "log_dir":     "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/logs",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="AMBER canonical atom reordering for QM40 (D17).")
    parser.add_argument("--full-data", action="store_true",
                        help="Run on the full QM40 dataset on the cluster (default: local sample).")
    return parser.parse_args()


def preflight_check():
    """Abort early if AMBER antechamber is not available."""
    if not os.path.exists(ANTECHAMBER):
        print(f"ERROR: antechamber not found at {ANTECHAMBER}")
        print("       Atom reordering requires AMBER23 — run on the cluster.")
        sys.exit(1)
    if not os.path.exists(REORD_SCRIPT):
        print(f"ERROR: reordering script not found at {REORD_SCRIPT}")
        sys.exit(1)


def reorder_xyz(xyz_path):
    """
    Call xyz_reord_fixed.sh on xyz_path.

    Returns True if reordering succeeded (xyz_path still exists with reordered content),
    False if antechamber failed (xyz_path renamed to xyz_path_ORIG by the shell script).
    """
    subprocess.run(
        [REORD_SCRIPT, xyz_path],
        capture_output=True,
        timeout=60,
    )
    return os.path.exists(xyz_path)


def main():
    args = parse_args()
    mode = "full" if args.full_data else "sample"
    paths = PATHS[mode]

    MOL_DIR     = paths["mol_dir"]
    MAPPING_CSV = paths["mapping_csv"]
    LOG_DIR     = paths["log_dir"]

    preflight_check()
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Mode: {mode}")
    print(f"Reading {MAPPING_CSV} ...")
    mapping_df = pd.read_csv(MAPPING_CSV)
    n_total = len(mapping_df)
    print(f"  {n_total} molecules")

    statuses = []
    failed   = []

    print(f"\nReordering atoms (AMBER canonical, D17) ...")
    for _, row in tqdm(mapping_df.iterrows(), total=n_total, desc="Reordering", unit="mol", file=sys.stdout):
        cid   = row["ID"]
        iconf = row["ICONF"]

        xyz_path = os.path.join(MOL_DIR, f"mol_{cid}_{iconf}.xyz")
        sdf_path = os.path.join(MOL_DIR, f"mol_{cid}_{iconf}.sdf")

        if not os.path.exists(xyz_path):
            statuses.append("missing")
            continue

        success = reorder_xyz(xyz_path)

        if success:
            statuses.append("success")
        else:
            # Shell script renamed xyz → xyz_ORIG on failure; clean up both.
            orig_path = xyz_path + "_ORIG"
            for path in (orig_path, sdf_path):
                if os.path.exists(path):
                    os.remove(path)
            statuses.append("failed")
            failed.append(row.to_dict())

    mapping_df["reorder_status"] = statuses
    mapping_df.to_csv(MAPPING_CSV, index=False)

    n_success      = statuses.count("success")
    n_failed  = statuses.count("failed")
    n_missing = statuses.count("missing")

    print(f"\nDone.")
    print(f"  Reordered success: {n_success}")
    print(f"  Failed:       {n_failed}")
    if n_missing:
        print(f"  Missing XYZ:  {n_missing}  (unexpected — check dedup output)")
    print(f"  Mapping  →  {MAPPING_CSV}  (reorder_status column added)")

    if failed:
        failed_path = os.path.join(LOG_DIR, "reorder_failed.csv")
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f"  Failed   →  {failed_path}")


if __name__ == "__main__":
    main()
