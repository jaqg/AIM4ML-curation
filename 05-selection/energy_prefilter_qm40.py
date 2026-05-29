#!/usr/bin/env python3
"""
energy_prefilter_qm40.py: QM40-specific energy outlier detection.

Fits OLS E_total ~ sum(n_i * e_i) over atom types {C,H,N,O,S,F,Cl,Br,P,I}
(no intercept — physically motivated: atomic energies are additive).
Flags molecules where the residual |z| > threshold.

This approach handles all heavy-element composition variation simultaneously,
avoiding false positives from Cl/F-heavy molecules that inflate per-atom energy
within a naive S-bin z-score.

Writes:
    stats/qm40_energy_status.csv   — ID, energy_status, S_count, E_per_atom, z_residual
    logs/qm40_energy_flagged.csv   — flagged rows only, with canonical_SMILES for inspection

prepare_input.py reads stats/qm40_energy_status.csv and applies
energy_status != "flagged" filter if the file exists.

Usage:
    python3 energy_prefilter_qm40.py --full-data
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
        "stats_csv":   f"{SAMPLE_DIR}/qm40_stats.csv",
        "status_csv":  f"{SAMPLE_DIR}/stats/qm40_energy_status.csv",
        "flagged_csv": f"{SAMPLE_DIR}/logs/qm40_energy_flagged.csv",
    },
    "full": {
        "stats_csv":   f"{AIM4ML}/stats/qm40_stats.csv",
        "status_csv":  f"{AIM4ML}/stats/qm40_energy_status.csv",
        "flagged_csv": f"{AIM4ML}/logs/qm40_energy_flagged.csv",
    },
}

ATOM_SYMS = ["C", "H", "N", "O", "S", "F", "Cl", "Br", "P", "I"]
FEAT_COLS = [f"n_{s}" for s in ATOM_SYMS]

REPORT_COLS = ["ID", "canonical_SMILES", "NAT", "S_count",
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
                   help="Residual z-score threshold (default: 3.5).")
    p.add_argument("--nrows", type=int, default=None,
                   help="Read only first N rows (dev/testing).")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = CONFIGS[args.mode]

    print(f"Reading {cfg['stats_csv']} ...")
    df = pd.read_csv(cfg["stats_csv"], nrows=args.nrows)
    n_total = len(df)
    print(f"  Loaded: {n_total:,}")

    if "Internal_E(0K)" not in df.columns or "NAT" not in df.columns:
        print("  Internal_E(0K) or NAT absent in stats CSV — skipping energy prefilter.")
        return

    print("Counting atoms per molecule ...")
    atom_counts = df["canonical_SMILES"].map(count_atoms).apply(pd.Series)
    df = pd.concat([df, atom_counts], axis=1)
    df["S_count"]    = df["n_S"]
    df["E_per_atom"] = df["Internal_E(0K)"] / df["NAT"]

    print("\nAtom-type distribution:")
    for col in FEAT_COLS:
        n_nonzero = int((df[col] > 0).sum())
        print(f"  {col[2:]:3s}: present in {n_nonzero:,} molecules")

    print("\nFitting OLS: E_total ~ sum(n_i * e_i), no intercept ...")
    X = df[FEAT_COLS].values.astype(float)
    y = df["Internal_E(0K)"].values
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    print("  Fitted atomic energies (Ha):")
    for sym, e in zip(ATOM_SYMS, beta):
        if (df[f"n_{sym}"] > 0).any():
            print(f"    e_{sym:2s} = {e:12.6f}")

    residuals = y - X @ beta
    res_mean  = residuals.mean()
    res_std   = residuals.std()
    z         = (residuals - res_mean) / res_std
    df["z_residual"]   = z
    df["energy_status"] = "ok"
    df.loc[np.abs(z) > args.threshold, "energy_status"] = "flagged"

    n_flagged = int((df["energy_status"] == "flagged").sum())
    n_ok      = n_total - n_flagged

    print(f"\n  Residual stats: mean={res_mean:.4g} Ha  std={res_std:.4g} Ha")
    print(f"\n--- Energy outlier summary ---")
    print(f"  Total    : {n_total:,}")
    print(f"  Flagged  : {n_flagged:,}  ({100*n_flagged/n_total:.2f}%)")
    print(f"  OK       : {n_ok:,}  ({100*n_ok/n_total:.2f}%)")

    if n_flagged:
        df_flagged = df[df["energy_status"] == "flagged"].copy()
        print(f"\n--- Flagged sample (first {REPORT_N}) ---")
        show = [c for c in REPORT_COLS if c in df_flagged.columns]
        print(df_flagged[show].head(REPORT_N).to_string(index=False))

    status_cols = ["ID", "energy_status", "S_count", "E_per_atom", "z_residual"]
    Path(cfg["status_csv"]).parent.mkdir(parents=True, exist_ok=True)
    df[status_cols].to_csv(cfg["status_csv"], index=False)
    print(f"\n  Status CSV   → {cfg['status_csv']}  ({n_total:,} rows)")

    if n_flagged:
        df_flagged = df[df["energy_status"] == "flagged"]
        Path(cfg["flagged_csv"]).parent.mkdir(parents=True, exist_ok=True)
        df_flagged[
            status_cols + ["canonical_SMILES", "NAT", "Internal_E(0K)"]
        ].to_csv(cfg["flagged_csv"], index=False)
        print(f"  Flagged CSV  → {cfg['flagged_csv']}")


if __name__ == "__main__":
    main()
