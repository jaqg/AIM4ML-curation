#!/usr/bin/env python3
"""
outlier_prefilter.py: Step 0 — flag structural/descriptor outliers before selection.

Checks theory-level-independent, structure-derived columns present in all
standardized selection CSVs: NAT, MolWt, TPSA. Energy-based checks are
dataset-specific (see qm40_energy_prefilter.py and analogues for other datasets).

Outputs:
    selection/qm40_clean_pool.csv    — molecules passing all checks (input to Layer 1)
    selection/outlier_flagged.csv    — flagged molecules + outlier_reason column

Usage:
    python3 outlier_prefilter.py
    python3 outlier_prefilter.py --threshold 3.0 --nrows 500
"""

import argparse
from pathlib import Path

import pandas as pd

IN_CSV      = "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/selection/qm40_selection_input.csv"
CLEAN_CSV   = "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/selection/qm40_clean_pool.csv"
FLAGGED_CSV = "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/selection/outlier_flagged.csv"

CHECK_COLS  = ["NAT", "MolWt", "TPSA"]
REPORT_COLS = ["ID", "NAT", "MolWt", "TPSA", "Internal_E(0K)", "outlier_reason"]
REPORT_N    = 15


def parse_args():
    p = argparse.ArgumentParser(description="Step 0 structural outlier pre-filter.")
    p.add_argument("--threshold", type=float, default=3.0,
                   help="Z-score threshold for outlier flagging (default: 3.0).")
    p.add_argument("--nrows", type=int, default=None,
                   help="Read only first N rows (dev/testing).")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Reading {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, nrows=args.nrows)
    n_total = len(df)
    print(f"  Loaded: {n_total:,}")

    present = [c for c in CHECK_COLS if c in df.columns]
    missing = [c for c in CHECK_COLS if c not in df.columns]
    if missing:
        print(f"  WARNING: columns not found, skipped: {missing}")

    print(f"\nChecking columns: {present}  (threshold |z| > {args.threshold})")

    outlier_flag  = pd.Series(False, index=df.index)
    outlier_parts = pd.Series("", index=df.index)

    for col in present:
        vals     = df[col]
        nan_mask = vals.isna()
        mean     = vals.mean()
        std      = vals.std()
        z        = (vals - mean).abs() / std

        col_flag = nan_mask | (z > args.threshold)
        n_col    = int(col_flag.sum())
        n_nan    = int(nan_mask.sum())
        n_zscore = int((~nan_mask & (z > args.threshold)).sum())

        print(f"  {col}: mean={mean:.4g}  std={std:.4g}  "
              f"flagged={n_col} (NaN={n_nan}, |z|>{args.threshold}={n_zscore})")

        outlier_parts[nan_mask] += col + ":NaN,"

        zscore_mask = ~nan_mask & (z > args.threshold)
        if zscore_mask.any():
            outlier_parts[zscore_mask] += (
                col + ":z=" + z[zscore_mask].round(2).astype(str) + ","
            )

        outlier_flag |= col_flag

    outlier_parts = outlier_parts.str.rstrip(",")

    df_flagged = df[outlier_flag].copy()
    df_flagged["outlier_reason"] = outlier_parts[outlier_flag].values
    df_clean   = df[~outlier_flag].copy()

    n_flagged = len(df_flagged)
    n_clean   = len(df_clean)

    print(f"\n--- Outlier summary ---")
    print(f"  Total input : {n_total:,}")
    print(f"  Flagged     : {n_flagged:,}  ({100*n_flagged/n_total:.2f}%)")
    print(f"  Clean pool  : {n_clean:,}  ({100*n_clean/n_total:.2f}%)")

    if n_flagged:
        print(f"\n--- Flagged sample (first {REPORT_N}) ---")
        show = [c for c in REPORT_COLS if c in df_flagged.columns]
        print(df_flagged[show].head(REPORT_N).to_string(index=False))
        print(f"\nReason counts (top 10):")
        print(df_flagged["outlier_reason"].value_counts().head(10).to_string())

    Path(CLEAN_CSV).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(CLEAN_CSV, index=False)
    df_flagged.to_csv(FLAGGED_CSV, index=False)

    print(f"\n  Clean pool  → {CLEAN_CSV}")
    print(f"  Flagged     → {FLAGGED_CSV}")


if __name__ == "__main__":
    main()
