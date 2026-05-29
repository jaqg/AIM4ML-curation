#!/usr/bin/env python3
"""
analyze_sensitivity.py  —  AIM4ML selection sensitivity comparison.

Reads all qm40_selected_*.csv from OUT_DIR.
Prints: (1) per-run stats table, (2) pairwise Jaccard overlap matrix.
With --nn: also computes within-set NN Tanimoto diversity metric per run.

Usage:
    python analyze_sensitivity.py
    python analyze_sensitivity.py --out-dir /custom/path/
    python analyze_sensitivity.py --nn --workers 40 --chunk-size 250
"""

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/selection"
POOL_N  = 149_752  # QM40 pool after Phase 1 filters
NBITS   = 2048


def parse_args():
    p = argparse.ArgumentParser(
        description="Compare AIM4ML selection sensitivity runs."
    )
    p.add_argument("--out-dir",    default=OUT_DIR,
                   help="Directory with qm40_selected_*.csv files.")
    p.add_argument("--nn",         action="store_true",
                   help="Compute within-set NN Tanimoto diversity (slow; needs RDKit).")
    p.add_argument("--workers",    type=int, default=4,
                   help="Parallel workers for NN computation (--nn only).")
    p.add_argument("--chunk-size", type=int, default=250,
                   help="Rows per worker chunk for NN computation (--nn only).")
    return p.parse_args()


def parse_tag(fname):
    """Extract run metadata from filename tag.

    Patterns:
      qm40_selected_morgan_k10000_t50.csv        → split=sqrt
      qm40_selected_morgan_k10000_t50_ff30.csv   → split=ff30
    """
    m = re.match(
        r"qm40_selected_(?P<metric>\w+)_k(?P<k>\d+)_t(?P<t1min>\d+)"
        r"(?:_ff(?P<ff>\d+))?\.csv",
        fname,
    )
    if not m:
        return None
    return {
        "metric": m.group("metric"),
        "k":      int(m.group("k")),
        "t1_min": int(m.group("t1min")),
        "split":  f"ff{m.group('ff')}" if m.group("ff") else "sqrt",
    }


# ── CSV stats ─────────────────────────────────────────────────────────────────

def compute_stats(df, meta):
    t1 = df[df["tier"] == "tier1"]
    t2 = df[df["tier"] == "tier2"]
    n_t1_groups  = t1["scaffold_SMILES"].nunique() if len(t1) else 0
    scaf_counts  = df["scaffold_SMILES"].value_counts()
    n_singletons = int((scaf_counts == 1).sum())
    avg_t1_bgt   = round(len(t1) / n_t1_groups, 1) if n_t1_groups else 0.0

    return {
        "run":         f"k{meta['k']}_t{meta['t1_min']}_{meta['split']}",
        "k":           meta["k"],
        "t1_min":      meta["t1_min"],
        "split":       meta["split"],
        "total":       len(df),
        "t1_n":        len(t1),
        "t2_n":        len(t2),
        "t1_%":        round(100 * len(t1) / len(df), 1),
        "t1_groups":   n_t1_groups,
        "avg_t1_bgt":  avg_t1_bgt,
        "uniq_scaf":   df["scaffold_SMILES"].nunique(),
        "singleton_n": n_singletons,
        "singletons%": round(100 * n_singletons / len(df), 1),
        "MW_mean":     round(df["MolWt"].mean(), 1),
        "MW_std":      round(df["MolWt"].std(), 1),
        "NAT_mean":    round(df["NAT"].mean(), 2),
        "NAT_std":     round(df["NAT"].std(), 2),
    }


def jaccard(a, b):
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _short_label(run):
    """k10000_t50_ff30 → t50_ff30;  k5000_t50_sqrt → 5k_t50"""
    return (run
            .replace("k10000_", "")
            .replace("k5000_",  "5k_")
            .replace("k20000_", "20k_")
            .replace("_sqrt",   "")
           )


# ── NN Tanimoto (--nn only) ───────────────────────────────────────────────────

def _build_fp_matrix(smiles_list):
    from rdkit import Chem
    from rdkit.Chem import DataStructs, rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=NBITS)
    mat = np.zeros((len(smiles_list), NBITS), dtype=np.float32)
    n_fail = 0
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            n_fail += 1
            continue
        fp  = gen.GetFingerprint(mol)
        row = np.zeros(NBITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, row)
        mat[i] = row
    if n_fail:
        print(f"  WARNING: {n_fail} SMILES failed parse — zero-vector used.")
    return mat


def _nn_worker(args):
    """Compute NN Tanimoto for rows [chunk_start, chunk_end).

    fp_matrix lives in shared memory; fp_counts passed as bytes (small).
    """
    shm_name, shape, chunk_start, chunk_end, counts_bytes = args

    counts = np.frombuffer(counts_bytes, dtype=np.float32)
    shm    = shared_memory.SharedMemory(name=shm_name)
    fp_mat = np.ndarray(shape, dtype=np.float32, buffer=shm.buf)

    chunk        = fp_mat[chunk_start:chunk_end]           # view, no copy
    inter        = chunk @ fp_mat.T                        # (C, N) BLAS SGEMM
    chunk_counts = counts[chunk_start:chunk_end]
    union        = chunk_counts[:, None] + counts[None, :] - inter
    tan          = np.where(union > 0.0, inter / union, 0.0)

    for local_i in range(chunk_end - chunk_start):
        tan[local_i, chunk_start + local_i] = 0.0         # exclude self

    nn_sim = tan.max(axis=1).astype(np.float32)
    shm.close()
    return chunk_start, nn_sim


def compute_nn_tanimoto(fp_matrix, fp_counts, workers, chunk_size):
    N   = len(fp_matrix)
    shm = shared_memory.SharedMemory(create=True, size=fp_matrix.nbytes)
    try:
        np.ndarray(fp_matrix.shape, dtype=np.float32, buffer=shm.buf)[:] = fp_matrix
        counts_bytes = fp_counts.tobytes()
        chunks = [
            (shm.name, fp_matrix.shape, i, min(i + chunk_size, N), counts_bytes)
            for i in range(0, N, chunk_size)
        ]
        nn_sim = np.zeros(N, dtype=np.float32)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for start, result in ex.map(_nn_worker, chunks):
                nn_sim[start:start + len(result)] = result
    finally:
        shm.close()
        shm.unlink()
    return nn_sim


def _nn_stats(arr):
    return {
        "nn_mean":   round(float(np.mean(arr)),           4),
        "nn_median": round(float(np.median(arr)),         4),
        "nn_p75":    round(float(np.percentile(arr, 75)), 4),
        "nn_p95":    round(float(np.percentile(arr, 95)), 4),
        "t1_nn_mean":    None,
        "t1_nn_median":  None,
        "t2_nn_mean":    None,
        "t2_nn_median":  None,
    }


def add_nn_stats(stats, df, nn_sim):
    s = _nn_stats(nn_sim)
    for tier, key_mean, key_med in [
        ("tier1", "t1_nn_mean", "t1_nn_median"),
        ("tier2", "t2_nn_mean", "t2_nn_median"),
    ]:
        idx = df.index[df["tier"] == tier].to_numpy()
        if len(idx):
            s[key_mean]   = round(float(np.mean(nn_sim[idx])),   4)
            s[key_med]    = round(float(np.median(nn_sim[idx])), 4)
    stats.update(s)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    files = sorted(Path(args.out_dir).glob("qm40_selected_*.csv"))
    if not files:
        print(f"No qm40_selected_*.csv found in {args.out_dir}")
        return

    runs    = []
    id_sets = {}

    for f in files:
        meta = parse_tag(f.name)
        if meta is None:
            print(f"  skip (unrecognized): {f.name}")
            continue
        df    = pd.read_csv(f)
        stats = compute_stats(df, meta)

        if args.nn:
            print(f"  {f.name}: building FP matrix ({len(df):,} mols) ...")
            fp_mat    = _build_fp_matrix(df["canonical_SMILES"].tolist())
            fp_counts = fp_mat.sum(axis=1)
            print(f"  {f.name}: computing NN Tanimoto "
                  f"(workers={args.workers}, chunk={args.chunk_size}) ...")
            nn_sim = compute_nn_tanimoto(fp_mat, fp_counts,
                                         args.workers, args.chunk_size)
            add_nn_stats(stats, df, nn_sim)
            print(f"    nn_mean={stats['nn_mean']:.4f}  "
                  f"nn_median={stats['nn_median']:.4f}  "
                  f"nn_p95={stats['nn_p95']:.4f}")
        else:
            print(f"  loaded  {f.name}  ({len(df):,} rows)")

        runs.append(stats)
        id_sets[stats["run"]] = set(df["ID"])

    if not runs:
        print("Nothing to analyze.")
        return

    runs.sort(key=lambda r: (r["k"], r["t1_min"], r["split"]))

    # ── Per-run stats ──────────────────────────────────────────────────────────
    tier_cols = ["k", "t1_min", "split", "total", "t1_n", "t2_n", "t1_%",
                 "t1_groups", "avg_t1_bgt"]
    prop_cols = ["uniq_scaf", "singleton_n", "singletons%",
                 "MW_mean", "MW_std", "NAT_mean", "NAT_std"]
    nn_cols   = ["nn_mean", "nn_median", "nn_p75", "nn_p95",
                 "t1_nn_mean", "t1_nn_median", "t2_nn_mean", "t2_nn_median"]

    all_cols  = tier_cols + prop_cols + (nn_cols if args.nn else [])
    df_all    = pd.DataFrame(runs)
    present   = ["run"] + [c for c in all_cols if c in df_all.columns]
    summary   = df_all[present].set_index("run")

    SEP = "─" * 78

    def _section(title, cols):
        present_cols = [c for c in cols if c in summary.columns]
        if not present_cols:
            return
        print(f"\n{SEP}")
        print(f"  {title}  (pool N={POOL_N:,})" if "split" in present_cols
              else f"  {title}")
        print(SEP)
        print(summary[present_cols].to_string())

    _section("Tier split", tier_cols)
    _section("Scaffold / property stats", prop_cols)
    if args.nn:
        _section("Within-set NN Tanimoto  (lower = more diverse)", nn_cols)

    # ── Pairwise Jaccard ───────────────────────────────────────────────────────
    labels = [r["run"] for r in runs]
    short  = [_short_label(lbl) for lbl in labels]
    n      = len(labels)
    jmat   = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            jmat[i, j] = jaccard(id_sets[labels[i]], id_sets[labels[j]])

    jdf = pd.DataFrame(jmat, index=short, columns=short)
    print(f"\n{SEP}")
    print("  Pairwise Jaccard overlap (ID sets)")
    print(SEP)
    if any(s != l for s, l in zip(short, labels)):
        print("  " + "  ".join(f"{s}={l}" for s, l in zip(short, labels)))
        print()
    print(jdf.round(3).to_string())
    print()


if __name__ == "__main__":
    main()
