#!/usr/bin/env python3
"""
report_stats.py — Extract structured statistics from stats_summary.csv.

Produces funnel table, descriptor summary, and Tanimoto statistics
suitable for pasting into the AIM4ML report.

Usage:
    python3 report_stats.py stats/stats_summary.csv --total-input 162954
    python3 report_stats.py stats/stats_summary.csv --total-input 162954 --rejects-dir rejects/
    python3 report_stats.py stats/stats_summary.csv --funnel-only --total-input 162954
"""

import argparse
import os
import pandas as pd
import numpy as np


# -- SDF counting ---------------------------------------------------------

# Map stage number → (label, reject_sdf_name, keep_status_value)
# Order must match pipeline flow for cumulative pass.
_STAGE_DEFS = [
    ("02", "Energy prefilter",   "energy_flagged.sdf"),
    ("03", "Chemical filter",    "filter_rejected.sdf"),
    ("06", "Stereo filter",      "stereo_removed.sdf"),
    ("07", "Reorder",            "reorder_failed.sdf"),
    ("08", "Conformer filter",   "conformer_removed.sdf"),
]


def count_sdf_entries(path):
    """Count molecules in an SDF file by counting $$$$ delimiters."""
    if not os.path.isfile(path):
        return 0
    with open(path, "r") as fh:
        return fh.read().count("$$$$")


def load_reject_counts(rejects_dir):
    """Return dict[stage_label] = count from reject SDF files."""
    counts = {}
    if not rejects_dir or not os.path.isdir(rejects_dir):
        return counts
    for stage_num, label, sdf_name in _STAGE_DEFS:
        sdf_path = os.path.join(rejects_dir,
                                f"{stage_num}_{sdf_name.split('_')[0]}",
                                sdf_name)
        # Also try the full stage name format: rejects/02_energy_prefilter/
        if not os.path.isfile(sdf_path):
            # Search for any directory starting with the stage number
            for entry in os.listdir(rejects_dir):
                if entry.startswith(f"{stage_num}_"):
                    alt_path = os.path.join(rejects_dir, entry, sdf_name)
                    if os.path.isfile(alt_path):
                        sdf_path = alt_path
                        break
        counts[label] = count_sdf_entries(sdf_path)
    return counts


# -- Funnel ---------------------------------------------------------------

def print_funnel(df, total_input, rejects_dir):
    reject_counts = load_reject_counts(rejects_dir)

    # Sequential cumulative pass: start from total_input, subtract
    # per-stage reject counts
    cum_pass = total_input
    print("Curation funnel")
    print("  Stage                  Cum. pass  New rejected")
    print("  ---------------------- ---------- -------------")
    print(f"  Input                 {cum_pass:10d}  {'—':>13s}")

    # Final unique count from CSV (trusted)
    n_unique = df["CompoundID"].nunique() if "CompoundID" in df.columns else len(df)

    for _, label, _ in _STAGE_DEFS:
        n_rejected = reject_counts.get(label, 0)
        if "Conformer" in label:
            # Conformer removes duplicate conformers of same CompoundID.
            # n_rejected = molecules_in - unique_compounds_out
            n_rejected_conformer = cum_pass - n_unique
            if n_rejected_conformer < 0:
                print(f"  WARNING: cum_pass={cum_pass} < n_unique={n_unique}."
                      f"  Reject SDFs may be from mixed runs —"
                      f"  try: make clean && make MODE=full")
                # Fallback: use SDF count (from current run)
                n_rejected = reject_counts.get(label, 0)
            else:
                n_rejected = n_rejected_conformer
            cum_pass = n_unique
        else:
            cum_pass = cum_pass - n_rejected
        print(f"  {label:22s} {cum_pass:10d} {n_rejected:13d}")

    # Final unique by CompoundID from CSV
    if "CompoundID" in df.columns:
        n_unique = df["CompoundID"].nunique()
    else:
        n_unique = len(df)
    print(f"  Final unique          {n_unique:10d}  {'—':>13s}")
    print()


# -- Descriptors ----------------------------------------------------------

def print_descriptors(df):
    print("Descriptors")
    if len(df) == 0:
        print("  (empty pool — no molecules pass all filters)")
        print()
        return

    if "num_atoms" in df.columns:
        nat = df["num_atoms"].dropna().astype(int)
        print(f"  NAT           mean={nat.mean():.0f} median={nat.median():.0f} "
              f"min={nat.min()} max={nat.max()}")
        bins = [(12, 40), (41, 65), (66, 92)]
        for lo, hi in bins:
            n = int(((nat >= lo) & (nat <= hi)).sum())
            pct = 100 * n / len(nat)
            print(f"  NAT {lo}-{hi}      {n:6d} ({pct:.1f}%)")
        for lo, hi in [(25, 35), (45, 55), (65, 75)]:
            n = int(((nat >= lo) & (nat <= hi)).sum())
            print(f"  NAT peak {lo}-{hi}: {n}")

    if "MolWt" in df.columns:
        mw = df["MolWt"].dropna()
        if len(mw):
            print(f"  MolWt         mean={mw.mean():.1f} median={mw.median():.1f} "
                  f"min={mw.min():.1f} max={mw.max():.1f}")

    if "TPSA" in df.columns:
        tpsa = df["TPSA"].dropna()
        if len(tpsa):
            pct_lt_140 = 100 * (tpsa < 140).sum() / len(tpsa)
            print(f"  TPSA          mean={tpsa.mean():.1f} median={tpsa.median():.1f} "
                  f"% < 140 = {pct_lt_140:.1f}%")

    if "Energy_Ha" in df.columns:
        ene = df["Energy_Ha"].dropna()
        if len(ene):
            print(f"  Energy (Ha)   mean={ene.mean():.4f} median={ene.median():.4f} "
                  f"min={ene.min():.4f} max={ene.max():.4f}")

    print()


# -- Tanimoto -------------------------------------------------------------

def print_tanimoto(df):
    if "max_tanimoto" not in df.columns or len(df) == 0:
        return
    # Use unique CompoundIDs
    if "CompoundID" in df.columns:
        df = df.drop_duplicates(subset="CompoundID")

    tani = df["max_tanimoto"].dropna()
    if len(tani) < 2:
        return

    n_t1 = int((tani == 1.0).sum())
    n_t85 = int((tani > 0.85).sum())

    print("Tanimoto (unique CompoundIDs)")
    print(f"  mean={tani.mean():.3f} median={tani.median():.3f} "
          f"max={tani.max():.3f}")
    print(f"  T=1.000 (twins): {n_t1} ({100*n_t1/len(tani):.1f}%)")
    print(f"  T>0.85  (near-dups): {n_t85} ({100*n_t85/len(tani):.1f}%)")
    print()


# -- Main ----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Extract structured statistics from stats_summary.csv."
    )
    p.add_argument("csv", type=str, help="Path to stats_summary.csv.")
    p.add_argument("--total-input", type=int, required=True,
                   help="Total molecules entering curation pipeline (stage 0 count).")
    p.add_argument("--rejects-dir", type=str, default="",
                   help="Path to rejects/ directory for per-stage SDF counts.")
    p.add_argument("--funnel-only", action="store_true",
                   help="Print only the curation funnel.")
    p.add_argument("--debug", action="store_true",
                   help="Print first rows and column names.")
    args = p.parse_args()

    df = pd.read_csv(args.csv)

    if args.debug:
        print(f"Columns: {sorted(df.columns.tolist())}")
        print(f"Rows: {len(df)}")
        for col in ["filter_status", "stereo_status", "energy_status",
                     "reorder_status", "conformer_status"]:
            if col in df.columns:
                print(f"{col}: {df[col].value_counts().to_dict()}")
        print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")
        print()

    print_funnel(df, args.total_input, args.rejects_dir)

    if not args.funnel_only:
        print_descriptors(df)
        print_tanimoto(df)


if __name__ == "__main__":
    main()
