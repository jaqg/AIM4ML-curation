#!/usr/bin/env python3
"""
selection_main.py: AIM4ML molecule selection — Layers 1+2 (D21, D22).

Layer 1 — Scaffold-aware tier partition (D22):
  Tier 1: scaffold groups with n_mols > TIER1_MIN → per-scaffold MaxMin, sqrt-prop budget
  Tier 2: remaining molecules (singletons + small/medium groups) → global MaxMin

Layer 2 — MaxMin diversity picking (D21):
  --metric morgan : ECFP4 (radius=2, 2048-bit) Tanimoto distance; fast, 2D; pipeline baseline
  --metric soap   : SOAP atomic environment distance, 3D (D21 first trial) [NOT YET IMPLEMENTED]

Budget allocation (D22-P2):
  --split-mode sqrt_prop (default):
    k_tier1 = round(k * sum(sqrt(n_i), tier1) / sum(sqrt(n_j), all groups))
  --split-mode fixed (D22-P4):
    k_tier1 = floor(tier1_fraction * k)   [e.g. fraction=0.30 → 3000 at k=10000]
  k_tier2 = k - k_tier1
  Within tier 1: budget_i = max(1, round(k_tier1 * sqrt(n_i) / sum(sqrt(n_j), tier1)))
  Tier 2:        global MaxMin on merged pool, budget = k_tier2

Sensitivity runs (D22-P3/P4):
    python3 selection_main.py --tier1-min 20
    python3 selection_main.py --tier1-min 50   (default)
    python3 selection_main.py --tier1-min 100
    python3 selection_main.py --split-mode fixed --tier1-fraction 0.30 --tier1-min 50

D21 trials:
    python3 selection_main.py --metric morgan  (default, fast)
    python3 selection_main.py --metric soap --workers 8  (D21 first trial, needs DScribe)

Dev/testing:
    python3 selection_main.py --nrows 5000
"""

import argparse
import math
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator

AIM4ML        = "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML"
SELECTION_CSV = f"{AIM4ML}/selection/qm40_selection_input.csv"
MOL_MAP_CSV   = f"{AIM4ML}/selection/mol_scaffold_map.csv"
GROUPS_CSV    = f"{AIM4ML}/selection/scaffold_groups.csv"
EXTXYZ_DIR    = f"{AIM4ML}/04-extxyz"
OUT_DIR       = f"{AIM4ML}/selection"


def parse_args():
    p = argparse.ArgumentParser(
        description="AIM4ML scaffold-aware MaxMin selection (D21/D22)."
    )
    p.add_argument("--metric",    choices=["morgan", "soap"], default="morgan",
                   help="Diversity metric for MaxMin (D21).")
    p.add_argument("--k",         type=int, default=10_000,
                   help="Total selection budget (D22-P1).")
    p.add_argument("--tier1-min", type=int, default=50,
                   help="Min scaffold family size for tier-1 MaxMin (D22-P3). Try 20 and 100.")
    p.add_argument("--seed",      type=int, default=42,
                   help="RNG seed for reproducibility.")
    p.add_argument("--split-mode", choices=["sqrt_prop", "fixed"], default="sqrt_prop",
                   help="Inter-tier budget split method (D22-P2/P4). "
                        "sqrt_prop: proportional to sqrt(n) globally; "
                        "fixed: tier1_fraction × k goes to tier 1.")
    p.add_argument("--tier1-fraction", type=float, default=0.30,
                   help="Fraction of k allocated to tier 1 when --split-mode fixed (D22-P4).")
    p.add_argument("--workers",   type=int, default=1,
                   help="Parallel workers (tier-1 scaffold groups; SOAP computation).")
    p.add_argument("--nrows",     type=int, default=None,
                   help="Load only first N rows (dev/testing).")
    return p.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(nrows):
    df     = pd.read_csv(SELECTION_CSV, nrows=nrows)
    mm     = pd.read_csv(MOL_MAP_CSV, usecols=["ID", "scaffold_SMILES"])
    groups = pd.read_csv(GROUPS_CSV)
    if nrows:
        mm = mm[mm["ID"].isin(df["ID"])]
    df = df.merge(mm, on="ID", how="left")
    df = df.reset_index(drop=True)  # positional index 0..N-1 must match desc array
    return df, groups


def assign_tiers(df, groups, tier1_min):
    t1_set = set(groups.loc[groups["n_mols"] > tier1_min, "scaffold_SMILES"])
    df = df.copy()
    df["tier"] = df["scaffold_SMILES"].map(
        lambda s: "tier1" if s in t1_set else "tier2"
    )
    return df, t1_set


# ── Descriptors ───────────────────────────────────────────────────────────────

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _morgan_worker(args):
    """ProcessPoolExecutor worker: computes Morgan FPs for one SMILES chunk."""
    smiles_list, nBits = args
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=nBits)
    local_mat = np.zeros((len(smiles_list), nBits), dtype=np.uint8)
    n_fail = 0
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_fail += 1
            continue
        fp = gen.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, local_mat[i])
    return local_mat, n_fail


def compute_morgan(df, nBits=2048, workers=1):
    smiles = df["canonical_SMILES"].tolist()
    n = len(smiles)
    print(f"  ECFP4 ({nBits} bits) for {n:,} molecules (workers={workers}) ...")
    chunk_size = max(1, math.ceil(n / workers))
    chunks = [
        (smiles[i:i + chunk_size], nBits)
        for i in range(0, n, chunk_size)
    ]
    matrix = np.zeros((n, nBits), dtype=np.uint8)
    n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, (local_mat, chunk_fails) in enumerate(ex.map(_morgan_worker, chunks)):
            start = k * chunk_size
            matrix[start:start + len(local_mat)] = local_mat
            n_fail += chunk_fails
    if n_fail:
        print(f"  WARNING: {n_fail} SMILES failed RDKit parse — zero-vector used.")
    print(f"  Matrix shape: {matrix.shape}, dtype: {matrix.dtype}, "
          f"size: {matrix.nbytes / 1e6:.0f} MB")
    return matrix


def compute_soap(df, extxyz_dir, workers):
    raise NotImplementedError(
        "SOAP metric not yet implemented.\n\n"
        "Implementation plan:\n"
        "  1. Parse extxyz files in EXTXYZ_DIR; index structures by 'idsml' field\n"
        "     in title line → dict {md5_id: (symbols, coords_Å)}\n"
        "  2. Build DScribe SOAP descriptor:\n"
        "       from dscribe.descriptors import SOAP\n"
        "       soap = SOAP(species=['C','H','N','O','S','F','Cl','Br','P','I'],\n"
        "                   r_cut=6.0, n_max=9, l_max=9, average='outer')\n"
        "  3. Build ase.Atoms objects from (symbols, coords); compute in parallel\n"
        "       with workers processes.\n"
        "  4. aggregate='outer' gives per-mol descriptor directly (sum over atoms).\n"
        "  5. L2-normalise rows → use cosine distance in make_soap_dist_fn.\n"
        "  6. Return (N, D) float32 array aligned to df row order.\n\n"
        "Use --metric morgan for now."
    )


# ── Distance factories ────────────────────────────────────────────────────────

def make_morgan_dist_fn(global_idx, fp_matrix, fp_counts):
    """Tanimoto distance closure over a subset of the full fingerprint matrix.

    sub_mat[i] = fp_matrix[global_idx[i]], so local index i maps to
    global position global_idx[i].  dist_fn(local_idx) returns distances from
    every member of the subset to the member at local_idx.
    """
    sub_mat    = fp_matrix[global_idx]  # already float32 (precomputed in main)
    sub_counts = fp_counts[global_idx]            # (M,) float32 precomputed bit counts

    def dist_fn(local_idx):
        inter = sub_mat.dot(sub_mat[local_idx])   # float32 BLAS, no cast needed
        union = sub_counts + float(sub_counts[local_idx]) - inter
        return 1.0 - np.where(union > 0.0, inter / union, 0.0).astype(np.float32)

    return dist_fn


def make_soap_dist_fn(global_idx, soap_matrix):
    """Cosine distance closure for L2-normalised SOAP descriptors."""
    sub_mat = soap_matrix[global_idx]             # (M, D) float32, L2-normalised

    def dist_fn(local_idx):
        dots    = sub_mat.dot(sub_mat[local_idx]) # (M,) cosine similarities
        dist_sq = np.maximum(2.0 - 2.0 * dots, 0.0)
        return np.sqrt(dist_sq).astype(np.float32)

    return dist_fn


# ── MaxMin algorithm ──────────────────────────────────────────────────────────

def maxmin_select(n, k, seed, dist_fn):
    """Greedy farthest-point (MaxMin) selection.  O(n × k) time, O(n) space.

    dist_fn(local_idx) → (n,) float32 distances from every pool member to
    the member at local_idx.

    Returns list of local indices (0..n-1), length min(k, n).
    Selection order is meaningful: index 0 = random seed, subsequent =
    farthest point from all previously selected.
    """
    if k >= n:
        return list(range(n))

    rng      = np.random.default_rng(seed)
    first    = int(rng.integers(n))
    selected = [first]
    min_d    = dist_fn(first)
    min_d[first] = 0.0

    for _ in range(k - 1):
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        new_d = dist_fn(nxt)
        np.minimum(min_d, new_d, out=min_d)
        min_d[nxt] = 0.0

    return selected


# ── Budget calculation ────────────────────────────────────────────────────────

def calc_k_tier1(groups, t1_set, k, split_mode="sqrt_prop", fraction=0.30):
    """Inter-tier budget split.

    sqrt_prop (D22-P2): k_tier1 = round(k × Σ√n_i(T1) / Σ√n_j(all))
    fixed     (D22-P4): k_tier1 = floor(fraction × k)
    """
    if split_mode == "fixed":
        return math.floor(fraction * k)
    all_sqrt = groups["n_mols"].apply(math.sqrt).sum()
    t1_sqrt  = groups.loc[
        groups["scaffold_SMILES"].isin(t1_set), "n_mols"
    ].apply(math.sqrt).sum()
    return round(k * t1_sqrt / all_sqrt)


def compute_scaffold_budgets(groups, t1_set, k_tier1):
    """Per-scaffold sqrt-prop budget within tier 1; floored to 1."""
    t1 = groups[groups["scaffold_SMILES"].isin(t1_set)].copy()
    t1["sqrt_n"] = t1["n_mols"].apply(math.sqrt)
    s = t1["sqrt_n"].sum()
    t1["budget"] = (t1["sqrt_n"] / s * k_tier1).round().clip(lower=1).astype(int)
    actual_sum = int(t1["budget"].sum())
    if actual_sum != k_tier1:
        print(f"  NOTE: after floor-to-1, tier1 budget sum = {actual_sum} "
              f"(target {k_tier1}; diff = {actual_sum - k_tier1})")
    return dict(zip(t1["scaffold_SMILES"], t1["budget"]))


# ── Tier runners ──────────────────────────────────────────────────────────────

def run_tier1(df, desc_matrix, fp_counts, groups, t1_set, k_tier1, seed, metric, workers):
    t1_df   = df[df["tier"] == "tier1"]
    budgets = compute_scaffold_budgets(groups, t1_set, k_tier1)
    items   = list(enumerate(t1_df.groupby("scaffold_SMILES")))

    def _process(args):
        i, (scaffold, grp) = args
        budget  = budgets.get(scaffold, 1)
        g_idx   = grp.index.to_numpy()
        if metric == "morgan":
            dist_fn = make_morgan_dist_fn(g_idx, desc_matrix, fp_counts)
        else:
            dist_fn = make_soap_dist_fn(g_idx, desc_matrix)
        local_sel = maxmin_select(len(g_idx), budget, seed + i, dist_fn)
        return g_idx[local_sel]

    selected = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for g_sel in ex.map(_process, items):
            selected.extend(g_sel)
    return selected


def run_tier2(df, desc_matrix, fp_counts, k_tier2, seed, metric):
    t2_df = df[df["tier"] == "tier2"]
    g_idx = t2_df.index.to_numpy()
    if metric == "morgan":
        dist_fn = make_morgan_dist_fn(g_idx, desc_matrix, fp_counts)
    else:
        dist_fn = make_soap_dist_fn(g_idx, desc_matrix)
    local_sel = maxmin_select(len(g_idx), k_tier2, seed, dist_fn)
    return g_idx[local_sel].tolist()


# ── Output ────────────────────────────────────────────────────────────────────

def write_output(df, t1_selected, t2_selected, metric, k, tier1_min,
                 split_mode="sqrt_prop", fraction=0.30):
    """Write selection CSV.  Tier-1 rows first (scaffold-grouped), then tier-2.
    selection_order within each tier reflects MaxMin greedy rank (1 = seed or
    first selected; higher = farther from previous picks).
    """
    parts = []
    for order, idx in enumerate(t1_selected, start=1):
        row = df.iloc[idx][["ID", "canonical_SMILES", "NAT", "MolWt",
                             "scaffold_SMILES", "tier", "source_dataset"]].to_dict()
        row["selection_order"] = order
        parts.append(row)
    for order, idx in enumerate(t2_selected, start=1):
        row = df.iloc[idx][["ID", "canonical_SMILES", "NAT", "MolWt",
                             "scaffold_SMILES", "tier", "source_dataset"]].to_dict()
        row["selection_order"] = order
        parts.append(row)

    out = pd.DataFrame(parts, columns=[
        "ID", "canonical_SMILES", "NAT", "MolWt",
        "scaffold_SMILES", "tier", "selection_order", "source_dataset",
    ])
    split_tag = f"_ff{round(fraction * 100)}" if split_mode == "fixed" else ""
    tag  = f"{metric}_k{k}_t{tier1_min}{split_tag}"
    path = Path(OUT_DIR) / f"qm40_selected_{tag}.csv"
    out.to_csv(path, index=False)
    return path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(f"\n=== AIM4ML selection  metric={args.metric}  k={args.k}  "
          f"tier1_min={args.tier1_min}  split={args.split_mode}"
          + (f"  fraction={args.tier1_fraction}" if args.split_mode == "fixed" else "")
          + " ===")

    print("\n[1/5] Loading data ...")
    df, groups = load_data(args.nrows)
    df, t1_set = assign_tiers(df, groups, args.tier1_min)

    n_t1_groups = df[df["tier"] == "tier1"]["scaffold_SMILES"].nunique()
    n_t1_mols   = (df["tier"] == "tier1").sum()
    n_t2_mols   = (df["tier"] == "tier2").sum()
    k_tier1     = calc_k_tier1(groups, t1_set, args.k,
                               args.split_mode, args.tier1_fraction)
    k_tier2     = args.k - k_tier1

    print(f"  Pool:   {len(df):,} molecules, {len(groups):,} scaffold groups")
    print(f"  Tier 1: {n_t1_groups} groups, {n_t1_mols:,} mols  →  budget {k_tier1}")
    print(f"  Tier 2: {n_t2_mols:,} mols  →  budget {k_tier2}")

    print(f"\n[2/5] Computing {args.metric} descriptors ...")
    if args.metric == "morgan":
        desc_matrix  = compute_morgan(df, workers=args.workers)
        fp_counts    = desc_matrix.sum(axis=1).astype(np.float32)
        desc_matrix  = desc_matrix.astype(np.float32)  # precompute once; BLAS dot in all dist_fn calls
        print(f"  Float32 matrix: {desc_matrix.nbytes / 1e6:.0f} MB")
    else:
        desc_matrix = compute_soap(df, EXTXYZ_DIR, args.workers)
        fp_counts   = None

    print(f"\n[3/5] Tier 1 MaxMin ({n_t1_groups} groups, budget={k_tier1}) ...")
    t1_sel = run_tier1(df, desc_matrix, fp_counts, groups, t1_set,
                       k_tier1, args.seed, args.metric, args.workers)
    print(f"  Selected: {len(t1_sel):,}")

    print(f"\n[4/5] Tier 2 MaxMin (pool={n_t2_mols:,}, budget={k_tier2}) ...")
    t2_sel = run_tier2(df, desc_matrix, fp_counts, k_tier2, args.seed, args.metric)
    print(f"  Selected: {len(t2_sel):,}")

    total = len(t1_sel) + len(t2_sel)
    print(f"\n[5/5] Writing output ({total:,} molecules) ...")
    path = write_output(df, t1_sel, t2_sel, args.metric, args.k, args.tier1_min,
                        args.split_mode, args.tier1_fraction)
    print(f"  → {path}")

    print(f"\n=== Done.  Total selected: {total:,} / {args.k:,} requested ===\n")


if __name__ == "__main__":
    main()
