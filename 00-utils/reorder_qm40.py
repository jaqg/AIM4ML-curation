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
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REORD_SCRIPT = os.path.join(SCRIPT_DIR, "xyz_reord_qm40.sh")
ANTECHAMBER  = "/opt/amber23/bin/antechamber"

PATHS = {
    "sample": {
        "mol_dir":     "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/mol_files",
        "mapping_csv": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/qm40_mapping.csv",
        "log_dir":     "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/logs",
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
    parser.add_argument("--sample", action="store_true",
                            help="Run on sample dataset (cluster absolute paths).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1). Uses threads — safe for subprocess.")
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


def _reorder_one(args):
    """Worker: reorder one molecule. Returns (mol_id, status, row_dict_or_None)."""
    mol_id, xyz_path, sdf_path, row_dict = args
    if not os.path.exists(xyz_path):
        return mol_id, "missing", None
    success = reorder_xyz(xyz_path)
    if success:
        return mol_id, "success", None
    orig_path = xyz_path + "_ORIG"
    for path in (orig_path, sdf_path):
        if os.path.exists(path):
            os.remove(path)
    return mol_id, "failed", row_dict


def main():
    args      = parse_args()
    mode = "full" if args.full_data else "sample"
    n_workers = args.workers
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

    all_ids    = set(mapping_df["ID"])
    to_process = mapping_df[mapping_df["stereo_status"] == "kept"].copy()
    n_process  = len(to_process)
    n_skip     = n_total - n_process
    print(f"  {n_total} total, {n_process} to process (stereo_status==kept), {n_skip} skipped")

    print(f"\nReordering atoms (AMBER canonical, D17) — {n_workers} worker(s) ...")

    work_items = []
    for _, row in to_process.iterrows():
        mol_id   = row["ID"]
        iconf    = row["ICONF"]
        xyz_path = os.path.join(MOL_DIR, f"mol_{mol_id}_{iconf}.xyz")
        sdf_path = os.path.join(MOL_DIR, f"mol_{mol_id}_{iconf}.sdf")
        work_items.append((mol_id, xyz_path, sdf_path, row.to_dict()))

    status_by_id = {mol_id: "skipped" for mol_id in all_ids}
    failed = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_reorder_one, item): item[0] for item in work_items}
        for future in tqdm(as_completed(futures), total=n_process, desc="Reordering", unit="mol", file=sys.stdout):
            mol_id, status, row_dict = future.result()
            status_by_id[mol_id] = status
            if row_dict is not None:
                failed.append(row_dict)

    mapping_df["reorder_status"] = mapping_df["ID"].map(status_by_id)
    mapping_df.to_csv(MAPPING_CSV, index=False)

    n_success = sum(1 for s in status_by_id.values() if s == "success")
    n_failed  = sum(1 for s in status_by_id.values() if s == "failed")
    n_missing = sum(1 for s in status_by_id.values() if s == "missing")
    n_skipped = sum(1 for s in status_by_id.values() if s == "skipped")

    print(f"\nDone.")
    print(f"  Reordered success: {n_success}")
    print(f"  Failed:       {n_failed}")
    if n_missing:
        print(f"  Missing XYZ:  {n_missing}  (unexpected — check dedup output)")
    print(f"  Skipped (enantiomers): {n_skipped}")
    print(f"  Mapping  →  {MAPPING_CSV}  (reorder_status column added)")

    if failed:
        failed_path = os.path.join(LOG_DIR, "reorder_failed.csv")
        pd.DataFrame(failed).to_csv(failed_path, index=False)
        print(f"  Failed   →  {failed_path}")


if __name__ == "__main__":
    main()
