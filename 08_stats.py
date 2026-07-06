#!/usr/bin/env python3
"""
08_stats.py — Stage 8: Compute molecular descriptors and diversity statistics.

Reads curated Parquet batches and produces:
  - stats_summary.csv   — per-molecule descriptors (MolWt, TPSA, logP, nrot)
  - Histogram plots     — NAT, MolWt, TPSA, Energy_Ha
  - Tanimoto similarity — nearest-neighbour (optional, --tanimoto)

Uses CanonicalSMILES from dedup step for descriptor computation.

Usage:
    python3 08_stats.py -i reordered_batches/ -o stats/
    python3 08_stats.py -i reordered_batches/ -o stats/ --tanimoto --workers 8
"""

import os
import sys
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from lib.parquet_io import read_batch

# Shared fingerprint list for multiprocessing workers.
_GLOBAL_FP_LIST = None


# -- Descriptors ----------------------------------------------------------

def compute_descriptors(smiles):
    """Return (MolWt, TPSA, logP, nrot, num_atoms) from canonical SMILES.

    No sanitization — explicit-H SMILES from dedup may have N valence 4
    that triggers AtomValenceException. Descriptors compute correctly
    on unsanitized mols.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None, None, None, None, None
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)  # initialize ring info without sanitization
    try:
        return (
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumRotatableBonds(mol),
            mol.GetNumAtoms(),
        )
    except Exception:
        return None, None, None, None, None


# -- Fingerprints (Tanimoto) ----------------------------------------------

def compute_fingerprints(smiles_list, n_bits=2048, radius=2):
    """Morgan FP for a list of canonical SMILES. Returns list of fp objects."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is not None:
            mol.UpdatePropertyCache(strict=False)
            Chem.GetSymmSSSR(mol)
            fps.append((gen.GetFingerprint(mol), smi))
        else:
            fps.append((None, smi))
    return fps


def nearest_neighbour_tanimoto(fps, workers=1):
    """Compute max Tanimoto to nearest neighbour for each molecule.

    Uses multiprocessing when workers > 1.
    """
    valid = [(fp, smi) for fp, smi in fps if fp is not None]
    if len(valid) < 2:
        return {smi: None for _, smi in fps}

    from rdkit import DataStructs

    fp_list = [f for f, _ in valid]
    smi_list = [s for _, s in valid]

    if workers > 1:
        from multiprocessing import Pool
        chunk_size = max(1, len(smi_list) // workers)
        indices = list(range(len(fp_list)))
        with Pool(workers, initializer=_init_worker, initargs=(fp_list,)) as pool:
            sim_results = pool.map(_compute_one_tanimoto, indices,
                                   chunksize=chunk_size)
        result_map = {smi_list[i]: sim_results[i]
                      for i in range(len(smi_list))}
    else:
        result_map = {}
        for i, (fp_i, smi_i) in enumerate(valid):
            sims = DataStructs.BulkTanimotoSimilarity(fp_i, fp_list)
            sims[i] = -1
            result_map[smi_i] = max(sims)

    for fp, smi in fps:
        if fp is None and smi not in result_map:
            result_map[smi] = None
    return result_map


def _compute_one_tanimoto(i):
    """Worker: compute max Tanimoto for molecule i against all others."""
    from rdkit import DataStructs
    fp_list = _GLOBAL_FP_LIST
    sims = DataStructs.BulkTanimotoSimilarity(fp_list[i], fp_list)
    sims[i] = -1
    return max(sims)


def _init_worker(fp_list):
    global _GLOBAL_FP_LIST
    _GLOBAL_FP_LIST = fp_list


# -- Main ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AIM4ML Stage 8 — Descriptors and diversity statistics."
    )
    p.add_argument("-i", "--input-dir", type=str, default="reordered_batches",
                   help="Input Parquet batch directory (default: reordered_batches/).")
    p.add_argument("-o", "--output-dir", type=str, default="stats",
                   help="Output directory for stats and plots (default: stats/).")
    p.add_argument("--tanimoto", action="store_true",
                   help="Compute Tanimoto similarity (slower, O(n²)).")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers for Tanimoto (if --tanimoto).")
    return p.parse_args()


def main():
    args = parse_args()

    batch_files = sorted(
        f for f in os.listdir(args.input_dir) if f.endswith(".parquet")
    )
    if not batch_files:
        print(f"No .parquet files found in {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    plots_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    # -- Accumulate all molecules -----------------------------------------
    all_smiles = []
    all_energies = []
    all_compound_ids = []
    all_source_ids = []
    total = 0

    for fname in batch_files:
        path = os.path.join(args.input_dir, fname)
        batch = read_batch(path)
        for row in batch:
            smi = row.get("CanonicalSMILES")
            energy = row.get("Energy_Ha")
            cid = row.get("CompoundID")
            src = row.get("SourceID")
            all_smiles.append(smi if smi else "")
            all_energies.append(energy if energy else np.nan)
            all_compound_ids.append(cid)
            all_source_ids.append(src)
            total += 1

    print(f"  {total} molecules loaded")

    # -- Compute descriptors ----------------------------------------------
    print("Computing descriptors ...")
    mol_wt_list, tpsa_list, logp_list, nrot_list, nat_list = [], [], [], [], []
    failed = 0

    for smi in all_smiles:
        mw, tpsa, logp, nrot, nat = compute_descriptors(smi)
        if mw is None:
            failed += 1
        mol_wt_list.append(mw)
        tpsa_list.append(tpsa)
        logp_list.append(logp)
        nrot_list.append(nrot)
        nat_list.append(nat)

    if failed:
        print(f"  {failed} descriptor failures (SMILES unparseable)")

    # -- Compute Tanimoto (optional) --------------------------------------
    tanimoto_map = {}
    if args.tanimoto:
        print(f"Computing Tanimoto NN (Morgan r=2, 2048 bits) ...")
        fps = compute_fingerprints(all_smiles)
        tanimoto_map = nearest_neighbour_tanimoto(fps, workers=args.workers)
        tanimoto_vals = [tanimoto_map.get(smi) for smi in all_smiles]
    else:
        tanimoto_vals = [None] * total

    # -- Write CSV --------------------------------------------------------
    import csv
    csv_path = os.path.join(args.output_dir, "stats_summary.csv")
    headers = [
        "CompoundID", "SourceID", "CanonicalSMILES",
        "MolWt", "TPSA", "logP", "nrot", "num_atoms",
        "Energy_Ha", "max_tanimoto",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for i in range(total):
            writer.writerow([
                all_compound_ids[i], all_source_ids[i], all_smiles[i],
                mol_wt_list[i], tpsa_list[i], logp_list[i], nrot_list[i],
                nat_list[i], all_energies[i], tanimoto_vals[i],
            ])
    print(f"\n  Stats → {csv_path}")

    # -- Histograms -------------------------------------------------------
    _plot_hist(nat_list, "Number of atoms (NAT)", plots_dir, "hist_nat.pdf",
               valid_only=True)
    _plot_hist([mw for mw in mol_wt_list if mw is not None],
               "Molecular weight (Da)", plots_dir, "hist_molwt.pdf")
    _plot_hist([t for t in tpsa_list if t is not None],
               "TPSA (Å²)", plots_dir, "hist_tpsa.pdf")
    _plot_hist([e for e in all_energies if not np.isnan(e)],
               "Energy (Ha)", plots_dir, "hist_energy.pdf")
    if args.tanimoto:
        _plot_hist([t for t in tanimoto_vals if t is not None],
                   "Max Tanimoto (nearest neighbour)", plots_dir,
                   "hist_tanimoto.pdf")

    # -- Summary ----------------------------------------------------------
    print(f"\nReport")
    print(f"  Total molecules:       {total}")
    if mol_wt_list.count(None) < total:
        valid_mw = [mw for mw in mol_wt_list if mw is not None]
        print(f"  MolWt:  mean={np.mean(valid_mw):.1f}  median={np.median(valid_mw):.1f}  "
              f"min={np.min(valid_mw):.1f}  max={np.max(valid_mw):.1f}")
    if tpsa_list.count(None) < total:
        valid_t = [t for t in tpsa_list if t is not None]
        print(f"  TPSA:   mean={np.mean(valid_t):.1f}  median={np.median(valid_t):.1f}  "
              f"min={np.min(valid_t):.1f}  max={np.max(valid_t):.1f}")
    if args.tanimoto:
        valid_tani = [t for t in tanimoto_vals if t is not None]
        if valid_tani:
            print(f"  Tanimoto NN: mean={np.mean(valid_tani):.3f}  "
                  f"median={np.median(valid_tani):.3f}  "
                  f"max={np.max(valid_tani):.3f}")


def _plot_hist(data, xlabel, plots_dir, filename, valid_only=False):
    """Save a histogram to plots_dir/filename."""
    if not data:
        return
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    path = os.path.join(plots_dir, filename)
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot → {path}")


if __name__ == "__main__":
    main()
