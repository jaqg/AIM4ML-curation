#!/usr/bin/env python3
"""
stats_qm40.py — Compute per-molecule descriptors and diversity statistics for QM40.

Reads qm40_mapping.csv + filtered_main.csv and produces:
  - qm40_stats.csv: master CSV with per-molecule descriptors and QM properties
  - plots/hist_nat.pdf, plots/hist_tpsa.pdf, plots/hist_tanimoto.pdf,
    plots/hist_energy.pdf

Descriptors computed from SMILES:
  - MolWt        : molecular weight (RDKit, includes H)
  - TPSA         : topological polar surface area (Å²)
  - logP         : Wildman-Crippen logP
  - nrot         : number of rotatable bonds (RDKit default definition)
  - charge       : formal molecular charge (0 for all QM40 — neutral singlets)
  - max_tanimoto       : nearest-neighbour Tanimoto similarity (Morgan FP, r=2, 2048 bits)
  - max_tanimoto_chiral: same with includeChirality=True (optional, --chiral flag)

QM properties joined from filtered_main.csv (join key: Zinc_id):
  - Internal_E(0K) : DFT internal energy at 0 K
  - HOMO           : HOMO energy
  - LUMO           : LUMO energy
  - HL_gap         : HOMO-LUMO gap

Curation status columns passed through from qm40_mapping.csv:
  - sdf_status, reorder_status, stereo_status

Usage:
    python3 stats_qm40.py               # local sample (default)
    python3 stats_qm40.py --full-data   # full dataset on cluster
    python3 stats_qm40.py --chiral      # also compute chirality-aware Tanimoto
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
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors, rdFingerprintGenerator

# --- Paths -------------------------------------------------------------------
QM_PROPS = ["Internal_E(0K)", "HOMO", "LUMO", "HL_gap"]

PATHS = {
    "sample": {
        "mapping_csv":       "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/qm40_mapping.csv",
        "filtered_main_csv": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/filtered_sample_main.csv",
        "stats_csv":         "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/qm40_stats.csv",
        "plots_dir":         "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/plots",
    },
    "full": {
        "mapping_csv":       "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/qm40_mapping.csv",
        "filtered_main_csv": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/filtered_main.csv",
        "stats_csv":         "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/stats/qm40_stats.csv",
        "plots_dir":         "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/stats/plots",
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
    parser.add_argument(
        "--chiral",
        action="store_true",
        help="Also compute chirality-aware Tanimoto NN (second FP pass, adds max_tanimoto_chiral column).",
    )
    return parser.parse_args()


def compute_descriptors(smiles_list):
    """
    Compute MolWt, TPSA, logP, and nrot for each SMILES string.
    Returns four lists, with None for molecules that fail to parse.
    """
    molwt_list = []
    tpsa_list  = []
    logp_list  = []
    nrot_list  = []
    for smi in tqdm(smiles_list, desc="Descriptors", unit="mol"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            molwt_list.append(None)
            tpsa_list.append(None)
            logp_list.append(None)
            nrot_list.append(None)
        else:
            molwt_list.append(Descriptors.MolWt(mol))
            tpsa_list.append(rdMolDescriptors.CalcTPSA(mol))
            logp_list.append(Crippen.MolLogP(mol))
            nrot_list.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
    return molwt_list, tpsa_list, logp_list, nrot_list


_MORGAN_GEN       = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_MORGAN_GEN_CHIRAL = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)


def compute_fingerprints(smiles_list, gen=_MORGAN_GEN, desc="Fingerprints"):
    """
    Compute Morgan fingerprints for all SMILES using the given generator.
    Returns a list of RDKit ExplicitBitVect objects (None for parse failures).
    """
    fps = []
    for smi in tqdm(smiles_list, desc=desc, unit="mol"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(None)
        else:
            fps.append(gen.GetFingerprint(mol))
    return fps


# Module-level globals so forked workers inherit fp_matrix/popcounts without pickling.
_FP_MATRIX  = None  # shape (n_valid, 2048), dtype uint8 --- each row = binary fingerprint of one mol
_POPCOUNTS  = None  # shape (n_valid,) --- popcount (population count: number of bits equal to 1 in a binary vector) per mol
#                                                   equiv to |A|


def _tanimoto_chunk(indices):
    """Worker: compute max Tanimoto for a chunk of row indices against the full fp matrix."""
    results = []
    for j in indices:
        intersections = _FP_MATRIX[j].dot(_FP_MATRIX.T)  # count of bits set in both j and i = |A ∩ B|
        unions = _POPCOUNTS[j] + _POPCOUNTS - intersections  # inclusion-exclusion: |A| + |B_i| − |A ∩ B_i| = |A ∪ B_i|
        sims = np.where(unions > 0, intersections / unions, 0.0)  # Tanimoto = |A ∩ B| / |A ∪ B|
        sims[j] = 0.0  # exclude self-similarity (= 1.0)
        results.append((j, float(sims.max()) if len(sims) > 1 else 0.0))  # Append nearest-neighbour Tanimoto (max) for mol j
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
    mode      = "full" if args.full_data else "sample"
    n_workers = args.workers
    do_chiral = args.chiral
    paths = PATHS[mode]

    MAPPING_CSV       = paths["mapping_csv"]
    FILTERED_MAIN_CSV = paths["filtered_main_csv"]
    STATS_CSV         = paths["stats_csv"]
    PLOTS_DIR         = paths["plots_dir"]

    print(f"Mode: {mode}")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Load mapping
    # -------------------------------------------------------------------------
    print(f"Reading {MAPPING_CSV} ...")
    mapping_df = pd.read_csv(MAPPING_CSV)
    n_total    = len(mapping_df)
    print(f"  {n_total} molecules loaded.")

    # -------------------------------------------------------------------------
    # Load QM properties from filtered_main.csv (join on Zinc_id)
    # -------------------------------------------------------------------------
    print(f"Reading QM properties from {FILTERED_MAIN_CSV} ...")
    qm_df = pd.read_csv(FILTERED_MAIN_CSV, usecols=["Zinc_id"] + QM_PROPS)
    mapping_df = mapping_df.merge(qm_df, on="Zinc_id", how="left")
    n_missing_qm = mapping_df[QM_PROPS[0]].isna().sum()
    if n_missing_qm:
        print(f"  WARNING: {n_missing_qm} molecules have no QM properties (not in filtered_main.csv).")

    smiles_list = mapping_df["canonical_SMILES"].tolist()

    # -------------------------------------------------------------------------
    # Step 1 — per-molecule descriptors
    # -------------------------------------------------------------------------
    molwt_list, tpsa_list, logp_list, nrot_list = compute_descriptors(smiles_list)

    n_failed = sum(1 for v in tpsa_list if v is None)
    if n_failed:
        print(f"  WARNING: {n_failed} molecules failed descriptor computation (unparseable SMILES).")

    # -------------------------------------------------------------------------
    # Step 2 — nearest-neighbour Tanimoto (non-chiral)
    # -------------------------------------------------------------------------
    fps = compute_fingerprints(smiles_list)
    max_tanimoto = compute_nearest_neighbour_tanimoto(fps, n_total, n_workers=n_workers)

    # Step 2b — chirality-aware Tanimoto (optional)
    if do_chiral:
        print("\nStep 2b: chirality-aware Tanimoto NN ...")
        fps_chiral = compute_fingerprints(smiles_list, gen=_MORGAN_GEN_CHIRAL, desc="Fingerprints (chiral)")
        max_tanimoto_chiral = compute_nearest_neighbour_tanimoto(fps_chiral, n_total, n_workers=n_workers)

    # -------------------------------------------------------------------------
    # Step 3 — build and write master stats CSV
    # -------------------------------------------------------------------------
    stats_df = mapping_df.copy()
    # charge=0 for all QM40 molecules (neutral singlets); explicit for cross-dataset merge.
    stats_df["charge"]       = 0
    stats_df["MolWt"]        = molwt_list
    stats_df["TPSA"]         = tpsa_list
    stats_df["logP"]         = logp_list
    stats_df["nrot"]         = nrot_list
    stats_df["max_tanimoto"] = max_tanimoto
    if do_chiral:
        stats_df["max_tanimoto_chiral"] = max_tanimoto_chiral

    col_order = [
        "ID", "ICONF", "Zinc_id", "canonical_SMILES", "NAT",
        "charge", "MolWt", "TPSA", "logP", "nrot",
        "Internal_E(0K)", "HOMO", "LUMO", "HL_gap",
        "max_tanimoto",
        *(["max_tanimoto_chiral"] if do_chiral else []),
        "sdf_status", "reorder_status", "stereo_status",
    ]
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
        xlabel="Nearest-neighbour Tanimoto (Morgan r=2, 2048 bits)",
        title=f"QM40 — Chemical diversity (N={n_total})",
        path=os.path.join(PLOTS_DIR, "hist_tanimoto.pdf"),
        bins=50,
        color="#C44E52",
    )

    if do_chiral:
        save_histogram(
            stats_df["max_tanimoto_chiral"].tolist(),
            xlabel="Nearest-neighbour Tanimoto (Morgan r=2, 2048 bits, chiral)",
            title=f"QM40 — Chemical diversity, chiral FP (N={n_total})",
            path=os.path.join(PLOTS_DIR, "hist_tanimoto_chiral.pdf"),
            bins=50,
            color="#DD8452",
        )

    save_histogram(
        stats_df["Internal_E(0K)"].tolist(),
        xlabel="Internal energy at 0 K (Ha)",
        title=f"QM40 — DFT internal energy distribution (N={n_total})",
        path=os.path.join(PLOTS_DIR, "hist_energy.pdf"),
        bins=50,
        color="#8172B2",
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
    for col, label in [
        ("max_tanimoto",       "non-chiral"),
        *([("max_tanimoto_chiral", "chiral")] if do_chiral else []),
    ]:
        tan = stats_df[col].dropna()
        if len(tan) > 0:
            n_near_dupes = (tan > 0.85).sum()
            print(f"\n  Tanimoto summary ({label}):")
            print(f"    Mean:                    {tan.mean():.3f}")
            print(f"    Median:                  {tan.median():.3f}")
            print(f"    Max:                     {tan.max():.3f}")
            print(f"    Near-duplicates (>0.85): {n_near_dupes} ({100*n_near_dupes/len(tan):.1f}%)")


if __name__ == "__main__":
    main()
