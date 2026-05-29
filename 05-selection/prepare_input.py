#!/usr/bin/env python3
"""
prepare_input.py: Build standardized selection input CSV for QM40 (D29/D31).

Reads qm40_stats.csv, applies kept-molecule filter (must stay in sync with
build_extxyz_qm40.py), adds source_dataset column.

Structural twins (diastereomers with identical Morgan FP, ~771 molecules) are
intentionally kept — MaxMin excludes the second twin naturally (D30 revised).

Output columns:
    ID, canonical_SMILES, NAT, charge, MolWt, TPSA, Internal_E(0K), source_dataset

Usage:
    python3 prepare_input.py              # full dataset on cluster
    python3 prepare_input.py --nrows 500  # dev/testing: limit to N rows
"""

import argparse
from pathlib import Path

import pandas as pd

STATS_CSV      = "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/stats/qm40_stats.csv"
ENERGY_CSV     = "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/stats/qm40_energy_status.csv"
OUT_CSV        = "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/selection/qm40_selection_input.csv"

OUT_COLS = [
    "ID", "canonical_SMILES", "NAT", "charge",
    "MolWt", "TPSA", "Internal_E(0K)", "source_dataset",
]


def parse_args():
    p = argparse.ArgumentParser(description="Build QM40 standardized selection input CSV (D29).")
    p.add_argument("--nrows", type=int, default=None,
                   help="Read only first N rows (dev/testing).")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Reading {STATS_CSV} ...")
    df = pd.read_csv(STATS_CSV, nrows=args.nrows)
    print(f"  Rows loaded:  {len(df):,}")

    # optional energy outlier filter (from qm40_energy_prefilter.py)
    energy_path = Path(ENERGY_CSV)
    if energy_path.exists():
        energy_df = pd.read_csv(energy_path, usecols=["ID", "energy_status"])
        df = df.merge(energy_df, on="ID", how="left")
        n_flagged = int((df["energy_status"] == "flagged").sum())
        print(f"  Energy status merged: {n_flagged:,} flagged molecules")
    else:
        df["energy_status"] = "ok"
        print(f"  {ENERGY_CSV} not found — energy outlier filter skipped")

    mask = (
        (df["reorder_status"] == "success") &
        (df["stereo_status"]  == "kept") &
        (df["sdf_status"]     != "sdf_failed") &
        (df["energy_status"]  != "flagged")
    )
    df = df[mask].copy()
    print(f"  After filter: {len(df):,}")

    df["source_dataset"] = "QM40"
    df = df[OUT_COLS]  # drops energy_status and all other non-schema columns

    Path(OUT_CSV).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    written = sum(1 for _ in open(OUT_CSV)) - 1  # verify by counting lines on disk
    print(f"  Output     →  {OUT_CSV}")
    print(f"  Written rows (verified): {written:,}")
    if written != len(df):
        raise RuntimeError(f"Write mismatch: in-memory {len(df):,} vs on-disk {written:,}")


if __name__ == "__main__":
    main()
