#!/usr/bin/env python3
"""
energy_prefilter_qm40.py: QM40-specific energy outlier detection.

Fits OLS E_total ~ sum(n_i * e_i) over atom types {C,H,N,O,S,F,Cl,Br,P,I}
(no intercept — physically motivated: atomic energies are additive).
Flags molecules where the residual |z| > threshold.

Reads filtered_main.csv, writes energy_status column back to it.
Logs flagged rows to logs/qm40_energy_flagged.csv.

Usage:
    python3 energy_prefilter_qm40.py --full-data
    python3 energy_prefilter_qm40.py --sample
    python3 energy_prefilter_qm40.py --full-data --threshold 3.0
    python3 energy_prefilter_qm40.py --full-data --nrows 500
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

AIM4ML     = "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML"
SAMPLE_DIR = f"{AIM4ML}/samples"

CONFIGS = {
    "sample": {
        "filtered_csv": f"{SAMPLE_DIR}/filtered_sample_main.csv",
        "flagged_csv":  f"{SAMPLE_DIR}/logs/qm40_energy_flagged.csv",
    },
    "full": {
        "filtered_csv": f"{AIM4ML}/filtered_main.csv",
        "flagged_csv":  f"{AIM4ML}/logs/qm40_energy_flagged.csv",
    },
}

ATOM_SYMS = ["C", "H", "N", "O", "S", "F", "Cl", "Br", "P", "I"]
FEAT_COLS = [f"n_{s}" for s in ATOM_SYMS]

REPORT_COLS = ["Zinc_id", "smile", "NAT", "S_count",
               "Internal_E(0K)", "E_per_atom", "z_residual"]
REPORT_N    = 15


def count_atoms(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {f"n_{s}": 0 for s in ATOM_SYMS}
    mol = Chem.AddHs(mol)
    counts = {f"n_{s}": 0 for s in ATOM_SYMS}
    for atom in mol.GetAtoms():
        key = f"n_{atom.GetSymbol()}"
        if key in counts:
            counts[key] += 1
    return counts


def parse_args():
    p = argparse.ArgumentParser(
        description="QM40 energy outlier detection via atom-type OLS regression.")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full-data", dest="mode", action="store_const", const="full",
                      help="Run on full cluster dataset.")
    mode.add_argument("--sample", dest="mode", action="store_const", const="sample",
                      help="Run on cluster sample data.")
    p.add_argument("--threshold", type=float, default=3.5,
                   help="MAD-based robust z-score threshold (default: 3.5).")
    p.add_argument("--nrows", type=int, default=None,
                   help="Read only first N rows (dev/testing).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = CONFIGS[args.mode]

    print(f"Reading {cfg['filtered_csv']} ...")
    df = pd.read_csv(cfg["filtered_csv"], nrows=args.nrows)
    n_total = len(df)
    print(f"  Loaded: {n_total:,}")

    if "Internal_E(0K)" not in df.columns:
        print("  Internal_E(0K) absent in filtered CSV — skipping energy prefilter.")
        df["energy_status"] = "ok"
        df.to_csv(cfg["filtered_csv"], index=False)
        return

    print("Counting atoms per molecule ...")
    atom_counts = df["smile"].map(count_atoms).apply(pd.Series)
    df = pd.concat([df, atom_counts], axis=1)
    df["NAT"]        = atom_counts.sum(axis=1)
    df["S_count"]    = df["n_S"]
    df["E_per_atom"] = df["Internal_E(0K)"] / df["NAT"]

    print("\nAtom-type distribution:")
    for col in FEAT_COLS:
        n_nonzero = int((df[col] > 0).sum())
        print(f"  {col[2:]:3s}: present in {n_nonzero:,} molecules")

    active_feat_cols = [c for c in FEAT_COLS if (df[c] > 0).any()]
    active_syms      = [c[2:] for c in active_feat_cols]

    print(f"\nFitting OLS: E_total ~ sum(n_i * e_i), no intercept  [{len(active_syms)} atom types: {', '.join(active_syms)}] ...")
    X = df[active_feat_cols].values.astype(float)
    y = df["Internal_E(0K)"].values
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    print("  Fitted atomic energies (Ha):")
    for sym, e in zip(active_syms, beta):
        print(f"    e_{sym:2s} = {e:12.6f}")

    residuals  = y - X @ beta
    res_median = float(np.median(residuals))
    mad        = float(np.median(np.abs(residuals - res_median)))
    z          = 0.6745 * (residuals - res_median) / mad
    df["z_residual"]    = z
    df["energy_status"] = "ok"
    df.loc[np.abs(z) > args.threshold, "energy_status"] = "flagged"

    n_flagged = int((df["energy_status"] == "flagged").sum())
    n_ok      = n_total - n_flagged

    print(f"\n  Residual stats: median={res_median:.4g} Ha  MAD={mad:.4g} Ha  (robust z-score, k=0.6745)")
    print(f"\n--- Energy outlier summary ---")
    print(f"  Total    : {n_total:,}")
    print(f"  Flagged  : {n_flagged:,}  ({100*n_flagged/n_total:.2f}%)")
    print(f"  OK       : {n_ok:,}  ({100*n_ok/n_total:.2f}%)")

    if n_flagged:
        df_flagged = df[df["energy_status"] == "flagged"].copy()
        print(f"\n--- Flagged sample (first {REPORT_N}) ---")
        show = [c for c in REPORT_COLS if c in df_flagged.columns]
        print(df_flagged[show].head(REPORT_N).to_string(index=False))

    # Write energy_status back to filtered_csv; drop temporary computation columns
    tmp_cols = FEAT_COLS + ["NAT", "S_count", "E_per_atom", "z_residual"]
    df_out = df.drop(columns=[c for c in tmp_cols if c in df.columns])
    df_out.to_csv(cfg["filtered_csv"], index=False)
    print(f"\n  Updated filtered_csv → {cfg['filtered_csv']}  (energy_status column added)")

    if n_flagged:
        df_flagged = df[df["energy_status"] == "flagged"]
        Path(cfg["flagged_csv"]).parent.mkdir(parents=True, exist_ok=True)
        flagged_cols = ["Zinc_id", "energy_status", "S_count", "E_per_atom", "z_residual",
                        "smile", "NAT", "Internal_E(0K)"]
        df_flagged[[c for c in flagged_cols if c in df_flagged.columns]].to_csv(
            cfg["flagged_csv"], index=False
        )
        print(f"  Flagged CSV  → {cfg['flagged_csv']}")


if __name__ == "__main__":
    main()
