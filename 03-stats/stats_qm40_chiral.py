#!/usr/bin/env python3
"""
stats_qm40_chiral.py — stats_qm40.py with chirality-aware Morgan fingerprints.

Identical to stats_qm40.py except Morgan fingerprints use includeChirality=True.
This gives a more accurate Tanimoto distribution for stereoisomer-rich datasets:
enantiomers and diastereomers get distinct bit vectors instead of Tanimoto = 1.0.

Context: the non-chiral run (stats_qm40.py) on QM40 reported mean Tanimoto 0.624
and 15.2% near-duplicates (>0.85), but check_stereo_pairs.py showed 14.8% of those
were stereo pairs — a measurement artifact of non-chiral FP, not real redundancy.
This script corrects that. Outputs go to separate files (qm40_stats_chiral.csv,
plots/hist_tanimoto_chiral.pdf) so the original run is not overwritten.

Usage:
    python3 stats_qm40_chiral.py               # local sample (default)
    python3 stats_qm40_chiral.py --full-data   # full dataset on cluster
"""

import os
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required for cluster (no display)
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors, rdFingerprintGenerator

# --- Paths -------------------------------------------------------------------
PATHS = {
    "sample": {
        "mapping_csv": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/qm40_mapping.csv",
        "stats_csv":   "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/qm40_stats_chiral.csv",
        "plots_dir":   "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/plots",
    },
    "full": {
        "mapping_csv": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/qm40_mapping.csv",
        "stats_csv":   "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/stats/qm40_stats_chiral.csv",
        "plots_dir":   "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/stats/plots",
    },
}
# -----------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Compute descriptors and statistics for QM40.")
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Run on the full QM40 dataset on the cluster (default: local sample).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on sample dataset (cluster absolute paths, default).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for Tanimoto NN computation (default: 1).",
    )
    return parser.parse_args()


def compute_descriptors(smiles_list):
    """
    Compute MolWt and TPSA for each SMILES string.
    Returns two lists (molwt, tpsa), with None for molecules that fail to parse.
    """
    molwt_list = []
    tpsa_list  = []
    for smi in tqdm(smiles_list, desc="Descriptors", unit="mol"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            molwt_list.append(None)
            tpsa_list.append(None)
        else:
            molwt_list.append(Descriptors.MolWt(mol))
            tpsa_list.append(rdMolDescriptors.CalcTPSA(mol))
    return molwt_list, tpsa_list


_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


def compute_fingerprints(smiles_list):
    """
    Compute Morgan fingerprints (radius=2, nBits=2048) for all SMILES.
    Returns a list of RDKit ExplicitBitVect objects (None for parse failures).
    """
    fps = []
    for smi in tqdm(smiles_list, desc="Fingerprints", unit="mol"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fps.append(_MORGAN_GEN.GetFingerprint(mol))
    return fps


# Module-level globals so forked workers inherit fp_matrix/popcounts without pickling.
_FP_MATRIX  = None
_POPCOUNTS  = None


def _tanimoto_chunk(indices):
    """Worker: compute max Tanimoto for a chunk of row indices against the full fp matrix."""
    results = []
    for j in indices:
        intersections = _FP_MATRIX[j].dot(_FP_MATRIX.T)
        unions = _POPCOUNTS[j] + _POPCOUNTS - intersections
        sims = np.where(unions > 0, intersections / unions, 0.0)
        sims[j] = 0.0  # exclude self
        results.append((j, float(sims.max()) if len(sims) > 1 else 0.0))
    return results


def compute_nearest_neighbour_tanimoto(fps, n_total, n_workers=1):
    """
    For each molecule i, compute its maximum Tanimoto similarity to any other molecule.

    Converts fps to a numpy uint8 matrix. On Linux the matrix is inherited by forked
    workers via copy-on-write — not pickled per task — so IPC overhead is minimal.
    Self-similarity is excluded. Returns a list of float (None for missing fps).
    """
    global _FP_MATRIX, _POPCOUNTS

    valid = [(i, fp) for i, fp in enumerate(fps) if fp is not None]
    n_valid = len(valid)
    valid_indices = [i for i, _ in valid]

    # Build numpy fp matrix — uint8, shape (n_valid, fpSize)
    fpSize = len(valid[0][1].ToBitString()) if valid else 2048
    _FP_MATRIX = np.zeros((n_valid, fpSize), dtype=np.uint8)
    for k, (_, fp) in enumerate(valid):
        DataStructs.ConvertToNumpyArray(fp, _FP_MATRIX[k])
    _POPCOUNTS = _FP_MATRIX.sum(axis=1).astype(np.float32)

    row_indices = list(range(n_valid))
    max_tanimoto = [None] * len(fps)

    if n_workers == 1:
        results = _tanimoto_chunk(row_indices)
        for orig_j, val in tqdm(results, desc="Tanimoto NN", unit="mol", total=n_valid):
            max_tanimoto[valid_indices[orig_j]] = val
    else:
        # More chunks than workers → tqdm updates frequently as workers grab work
        n_chunks = n_workers * 20
        chunks = np.array_split(row_indices, n_chunks)
        chunks = [c.tolist() for c in chunks if len(c) > 0]
        print(f"  Tanimoto NN: {n_valid} mols, {n_workers} workers, {len(chunks)} chunks ...")
        # Pool is created AFTER globals are set so forked workers inherit _FP_MATRIX
        with mp.Pool(processes=n_workers) as pool:
            for chunk_results in tqdm(
                pool.imap_unordered(_tanimoto_chunk, chunks),
                desc="Tanimoto NN chunks",
                unit="chunk",
                total=len(chunks),
            ):
                for orig_j, val in chunk_results:
                    max_tanimoto[valid_indices[orig_j]] = val

    return max_tanimoto


def save_histogram(values, xlabel, title, path, bins=50, color="#4C72B0"):
    """Save a histogram to a PDF file. Skips None values silently."""
    data = [v for v in values if v is not None]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(data, bins=bins, color=color, edgecolor="white", linewidth=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    args      = parse_args()
    mode = "full" if args.full_data else "sample"
    n_workers = args.workers
    paths = PATHS[mode]

    MAPPING_CSV = paths["mapping_csv"]
    STATS_CSV   = paths["stats_csv"]
    PLOTS_DIR   = paths["plots_dir"]

    print(f"Mode: {mode}")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load mapping
    # -------------------------------------------------------------------------
    print(f"Reading {MAPPING_CSV} ...")
    mapping_df = pd.read_csv(MAPPING_CSV)
    n_total    = len(mapping_df)
    print(f"  {n_total} molecules loaded.")

    smiles_list = mapping_df["canonical_SMILES"].tolist()

    # -------------------------------------------------------------------------
    # Step 1 — per-molecule descriptors
    # -------------------------------------------------------------------------
    molwt_list, tpsa_list = compute_descriptors(smiles_list)

    n_failed = sum(1 for v in tpsa_list if v is None)
    if n_failed:
        print(f"  WARNING: {n_failed} molecules failed descriptor computation (unparseable SMILES).")

    # -------------------------------------------------------------------------
    # Step 2 — nearest-neighbour Tanimoto
    # -------------------------------------------------------------------------
    fps = compute_fingerprints(smiles_list)
    max_tanimoto = compute_nearest_neighbour_tanimoto(fps, n_total, n_workers=n_workers)

    # -------------------------------------------------------------------------
    # Step 3 — build and write master stats CSV
    # -------------------------------------------------------------------------
    stats_df = mapping_df.copy()
    # charge=0 for all QM40 molecules (neutral singlets); explicit for cross-dataset merge.
    stats_df["charge"]       = 0
    stats_df["MolWt"]        = molwt_list
    stats_df["TPSA"]         = tpsa_list
    stats_df["max_tanimoto"] = max_tanimoto

    # Reorder columns to match dashboard spec: ID, ICONF, SMILES, NAT, charge, TPSA, ...
    col_order = ["ID", "ICONF", "Zinc_id", "canonical_SMILES", "NAT",
                 "charge", "MolWt", "TPSA", "max_tanimoto"]
    stats_df = stats_df[col_order]
    stats_df.to_csv(STATS_CSV, index=False)
    print(f"\nStats CSV written: {STATS_CSV}")

    # -------------------------------------------------------------------------
    # Step 4 — histograms
    # -------------------------------------------------------------------------
    print(f"\nGenerating histograms ...")

    save_histogram(
        stats_df["NAT"].tolist(),
        xlabel="Number of atoms (NAT)",
        title=f"QM40 — Atom count distribution (N={n_total})",
        path=os.path.join(PLOTS_DIR, "hist_nat.pdf"),
        bins=range(int(stats_df["NAT"].min()), int(stats_df["NAT"].max()) + 2),
        color="#4C72B0",
    )

    save_histogram(
        stats_df["TPSA"].tolist(),
        xlabel="TPSA (Å²)",
        title=f"QM40 — Topological polar surface area (N={n_total})",
        path=os.path.join(PLOTS_DIR, "hist_tpsa.pdf"),
        bins=50,
        color="#55A868",
    )

    save_histogram(
        stats_df["max_tanimoto"].tolist(),
        xlabel="Nearest-neighbour Tanimoto (Morgan r=2, 2048 bits, chiral)",
        title=f"QM40 — Chemical diversity, chiral FP (N={n_total})",
        path=os.path.join(PLOTS_DIR, "hist_tanimoto_chiral.pdf"),
        bins=50,
        color="#C44E52",
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print(f"\nDone.")
    print(f"  Molecules:        {n_total}")
    print(f"  Descriptor fails: {n_failed}")
    print(f"  Stats CSV:        {STATS_CSV}")
    print(f"  Plots:            {PLOTS_DIR}/")

    # Quick diversity report
    tan = stats_df["max_tanimoto"].dropna()
    if len(tan) > 0:
        n_near_dupes = (tan > 0.85).sum()
        print(f"\n  Tanimoto summary:")
        print(f"    Mean:             {tan.mean():.3f}")
        print(f"    Median:           {tan.median():.3f}")
        print(f"    Max:              {tan.max():.3f}")
        print(f"    Near-duplicates (>0.85): {n_near_dupes} ({100*n_near_dupes/len(tan):.1f}%)")


if __name__ == "__main__":
    main()
