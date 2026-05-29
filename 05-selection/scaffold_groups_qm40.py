#!/usr/bin/env python3
"""
scaffold_groups_qm40.py: Bemis-Murcko scaffold grouping for QM40 (informs D22).

Reads qm40_selection_input.csv, assigns Bemis-Murcko scaffold SMILES, groups
molecules by scaffold, and outputs summary stats + molecule-scaffold mapping.

Acyclic molecules (empty Murcko scaffold) are grouped as "[acyclic]".
Budget columns use k=10,000 as illustrative total for D22 supervisor meeting.

Usage:
    python3 scaffold_groups_qm40.py              # full dataset on cluster
    python3 scaffold_groups_qm40.py --nrows 500  # dev/testing
"""

import argparse
import math
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

IN_CSV      = "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/selection/qm40_selection_input.csv"
GROUPS_CSV  = "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/selection/scaffold_groups.csv"
MOL_MAP_CSV = "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/selection/mol_scaffold_map.csv"

ACYCLIC_LABEL  = "[acyclic]"
K_ILLUSTRATIVE = 10_000


def get_scaffold(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    scaf_smi = Chem.MolToSmiles(scaf) if scaf is not None else ""
    return scaf_smi if scaf_smi else ACYCLIC_LABEL


def parse_args():
    p = argparse.ArgumentParser(description="Bemis-Murcko scaffold grouping for QM40 (D22).")
    p.add_argument("--nrows", type=int, default=None,
                   help="Read only first N rows (dev/testing).")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Reading {IN_CSV} ...")
    df = pd.read_csv(IN_CSV, nrows=args.nrows)
    n_total = len(df)
    print(f"  Loaded: {n_total:,}")

    print("Computing Bemis-Murcko scaffolds ...")
    df["scaffold_SMILES"] = df["canonical_SMILES"].map(get_scaffold)

    n_failed = df["scaffold_SMILES"].isna().sum()
    if n_failed:
        print(f"  WARNING: {n_failed:,} SMILES failed RDKit parse — excluded from groups")
    df = df.dropna(subset=["scaffold_SMILES"])

    # --- scaffold groups ---
    groups = (
        df.groupby("scaffold_SMILES", sort=False)
        .size()
        .reset_index(name="n_mols")
        .sort_values("n_mols", ascending=False)
        .reset_index(drop=True)
    )
    groups["sqrt_n"] = groups["n_mols"].apply(math.sqrt)
    sqrt_sum = groups["sqrt_n"].sum()
    groups["budget_sqrt"] = (groups["sqrt_n"] / sqrt_sum * K_ILLUSTRATIVE).round().astype(int)
    groups["budget_prop"] = (groups["n_mols"] / len(df)  * K_ILLUSTRATIVE).round().astype(int)

    n_groups    = len(groups)
    budget_flat = round(K_ILLUSTRATIVE / n_groups)

    # --- summary ---
    n_acyclic   = int(groups.loc[groups["scaffold_SMILES"] == ACYCLIC_LABEL, "n_mols"].sum())
    n_singleton = int((groups["n_mols"] == 1).sum())

    print(f"\n--- Scaffold summary ({len(df):,} molecules) ---")
    print(f"  Unique scaffold groups : {n_groups:,}")
    print(f"  Acyclic molecules      : {n_acyclic:,}  ({100*n_acyclic/n_total:.1f}%)")
    print(f"  Singleton scaffolds    : {n_singleton:,}  ({100*n_singleton/n_groups:.1f}% of groups)")

    print(f"\n  Size distribution:")
    bands = [
        (1,  1,  "singleton (=1)  "),
        (2,  5,  "small     (2–5) "),
        (6,  50, "medium    (6–50)"),
        (51, None, "large     (>50) "),
    ]
    for lo, hi, label in bands:
        mask = (groups["n_mols"] >= lo) if hi is None else (groups["n_mols"] >= lo) & (groups["n_mols"] <= hi)
        n_g  = int(mask.sum())
        n_m  = int(groups.loc[mask, "n_mols"].sum())
        print(f"    {label} : {n_g:5,} groups  {n_m:7,} mols  ({100*n_m/n_total:.1f}%)")

    print(f"\n  Top 10 scaffolds by size:")
    for _, row in groups.head(10).iterrows():
        smi = row["scaffold_SMILES"]
        print(f"    n={int(row['n_mols']):5,}  {smi[:80]}")

    print(f"\n  Illustrative budget comparison (k={K_ILLUSTRATIVE:,}, {n_groups:,} groups):")
    print(f"    sqrt(n) — group budget range : {groups['budget_sqrt'].min()}–{groups['budget_sqrt'].max()}")
    print(f"    prop(n) — group budget range : {groups['budget_prop'].min()}–{groups['budget_prop'].max()}")
    print(f"    flat    — each group gets    : {budget_flat}")
    zero_sqrt = int((groups["budget_sqrt"] == 0).sum())
    if zero_sqrt:
        print(f"    WARNING: {zero_sqrt:,} groups get budget=0 under sqrt allocation (will be floored to 1 in selection_main.py)")

    # --- write outputs ---
    groups.to_csv(GROUPS_CSV, index=False)
    df[["ID", "scaffold_SMILES"]].to_csv(MOL_MAP_CSV, index=False)
    print(f"\n  Groups CSV  → {GROUPS_CSV}  ({len(groups):,} rows)")
    print(f"  Mol map CSV → {MOL_MAP_CSV}  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
