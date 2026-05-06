#!/usr/bin/env python3
"""
check_stereo_pairs.py — Investigate near-duplicate and stereoisomer pairs in QM40.

Context
-------
stats_qm40.py (full QM40 run) reported:
  - Mean nearest-neighbour Tanimoto: 0.624
  - Near-duplicates (Tanimoto > 0.85): 24 818 molecules (15.2%)
  - Max Tanimoto: 1.000

The max = 1.000 is expected: Morgan fingerprints are computed WITHOUT chirality
(useChirality=False default), so enantiomers and diastereomers produce identical
bit vectors. This script decomposes the 15.2% near-duplicate population into:

  1. Stereo pairs     — same 2D graph, different stereochemistry (Tanimoto = 1.0)
  2. Structural twins — genuinely different molecules with Tanimoto > 0.85

Also reports the Tanimoto = 1.0 group in full detail.

Usage:
    python3 check_stereo_pairs.py               # local sample
    python3 check_stereo_pairs.py --full-data   # full dataset on cluster
"""

import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

tqdm.pandas(desc="Stereo strip", unit="mol")

# --- Paths -------------------------------------------------------------------
PATHS = {
    "sample": {
        "stats_csv": "../../samples/qm40/qm40_stats.csv",
        "out_stereo": "../../samples/qm40/logs/stereo_pairs.tsv",
        "out_twins":  "../../samples/qm40/logs/structural_twins.tsv",
    },
    "full": {
        "stats_csv": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/stats/qm40_stats.csv",
        "out_stereo": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/logs/stereo_pairs.tsv",
        "out_twins":  "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/logs/structural_twins.tsv",
    },
}
# -----------------------------------------------------------------------------

NEAR_DUP_THRESHOLD = 0.85   # Tanimoto threshold used in stats_qm40.py


def parse_args():
    parser = argparse.ArgumentParser(description="Investigate near-duplicate pairs in QM40.")
    parser.add_argument("--full-data", action="store_true")
    return parser.parse_args()


def strip_stereo(smi):
    """Return canonical SMILES with all stereo information removed, or None on failure."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=False)
    except Exception:
        return None


def main():
    args  = parse_args()
    mode  = "full" if args.full_data else "sample"
    paths = PATHS[mode]

    print(f"Mode: {mode}")
    print(f"Reading {paths['stats_csv']} ...")
    df = pd.read_csv(paths["stats_csv"])
    n  = len(df)
    print(f"  {n} molecules loaded.")

    # -------------------------------------------------------------------------
    # Step 1 — strip stereo from canonical SMILES
    # -------------------------------------------------------------------------
    df["smi_no_stereo"] = df["canonical_SMILES"].progress_apply(strip_stereo)

    n_failed = df["smi_no_stereo"].isna().sum()
    if n_failed:
        print(f"  WARNING: {n_failed} molecules failed stereo stripping.")

    # -------------------------------------------------------------------------
    # Step 2 — find stereo groups (same 2D graph, different stereo)
    # -------------------------------------------------------------------------
    stereo_groups = df[df.duplicated("smi_no_stereo", keep=False)].copy()
    n_stereo_mols = len(stereo_groups)
    n_stereo_graphs = stereo_groups["smi_no_stereo"].nunique()

    print(f"\nStereo pair analysis:")
    print(f"  Molecules with at least one stereo partner: {n_stereo_mols} ({100*n_stereo_mols/n:.1f}%)")
    print(f"  Unique 2D scaffolds with multiple stereo forms: {n_stereo_graphs}")
    print(f"  → These account for the Tanimoto = 1.000 maximum.")

    # Size distribution of stereo groups
    group_sizes = stereo_groups.groupby("smi_no_stereo").size()
    print(f"\n  Stereo group size distribution:")
    for size, count in group_sizes.value_counts().sort_index().items():
        print(f"    {size} stereo forms: {count} groups ({count * size} molecules)")

    # -------------------------------------------------------------------------
    # Step 3 — near-duplicates that are NOT stereo pairs
    # -------------------------------------------------------------------------
    near_dup_df = df[df["max_tanimoto"] > NEAR_DUP_THRESHOLD].copy()
    n_near_dup  = len(near_dup_df)

    structural_twins = near_dup_df[~near_dup_df.duplicated("smi_no_stereo", keep=False)]
    n_twins = len(structural_twins)

    print(f"\nNear-duplicate decomposition (Tanimoto > {NEAR_DUP_THRESHOLD}):")
    print(f"  Total near-duplicates:  {n_near_dup} ({100*n_near_dup/n:.1f}%)")
    print(f"  Stereo pairs:           {n_stereo_mols} ({100*n_stereo_mols/n:.1f}%)")
    print(f"  Structural twins:       {n_twins} ({100*n_twins/n:.1f}%)")
    print(f"  (structural twins = near-duplicates that are NOT stereo partners)")

    # -------------------------------------------------------------------------
    # Step 4 — write output files
    # -------------------------------------------------------------------------
    import os
    os.makedirs(os.path.dirname(paths["out_stereo"]), exist_ok=True)

    cols_out = ["ID", "Zinc_id", "canonical_SMILES", "smi_no_stereo", "NAT", "max_tanimoto"]

    stereo_out = stereo_groups[cols_out].sort_values("smi_no_stereo")
    stereo_out.to_csv(paths["out_stereo"], sep="\t", index=False)
    print(f"\n  Stereo pairs written → {paths['out_stereo']}")

    twins_out = structural_twins[cols_out].sort_values("max_tanimoto", ascending=False)
    twins_out.to_csv(paths["out_twins"], sep="\t", index=False)
    print(f"  Structural twins written → {paths['out_twins']}")

    # -------------------------------------------------------------------------
    # Step 5 — implication for ML splitting
    # -------------------------------------------------------------------------
    print(f"\nML splitting implication:")
    print(f"  {n_stereo_mols} stereo-partner molecules should go to the SAME split.")
    print(f"  {n_twins} structural twins (Tanimoto > {NEAR_DUP_THRESHOLD}, not stereo) also risk leaking.")
    print(f"  → Use scaffold-aware splitting (Butina/Bemis-Murcko), not random split.")


if __name__ == "__main__":
    main()
