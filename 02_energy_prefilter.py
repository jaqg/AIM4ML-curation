#!/usr/bin/env python3
"""
02_energy_prefilter.py — Stage 2: Energy outlier detection via atom-type OLS.

Fits OLS regression  E_total ~ sum(n_i * e_i)  over atom types (no intercept —
physically motivated: atomic energies are additive).  Flags molecules where the
MAD-based robust z-score exceeds a configurable threshold.

Reads all Parquet batches from a directory, fits a global model, then writes
each batch to an output directory with an `energy_status` column ("ok" / "flagged").
Flagged molecules are also written as human-readable SDF to rejects/02_energy_prefilter/.

Usage:
    python3 02_energy_prefilter.py -i batches/ -o filtered_batches/ --threshold 3.5
"""

import os
import sys
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
from rdkit import Chem

from lib.parquet_io import read_batch, write_batch
from lib.schema import PARQUET_COLUMNS
from lib.sdf_io import write_reject_sdf

# Default atom types for the OLS regression.
ATOM_TYPES = ["C", "H", "N", "O", "S", "F", "Cl", "Br", "P", "I"]


# -- Atom counting --------------------------------------------------------

def count_atoms(mol):
    """Count heavy atoms and explicit hydrogens in an RDKit Mol.

    The molecule is reconstructed from mol_block, which includes all atoms
    (explicit H from the original xyz coords).  No AddHs call.
    """
    counts = {sym: 0 for sym in ATOM_TYPES}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym in counts:
            counts[sym] += 1
    return counts


# -- OLS fitting ----------------------------------------------------------

def fit_ols(rows, atom_types):
    """Fit OLS  E_total ~ sum(n_i * e_i)  (no intercept).

    Parameters
    ----------
    rows : list of dict
        Each row must contain 'Energy_Ha' (float) and 'mol' (RDKit Mol).
    atom_types : list of str
        Element symbols used as regressors.

    Returns
    -------
    beta : np.ndarray
        Fitted atomic energies, indexed by atom_types order.
    residuals : np.ndarray
        Residuals (E_actual - E_fitted) for each row.
    z : np.ndarray
        MAD-based robust z-score for each row.
    """
    n = len(rows)
    k = len(atom_types)
    X = np.zeros((n, k))
    y = np.zeros(n)

    for i, row in enumerate(rows):
        mol = row["_mol"]
        counts = count_atoms(mol)
        for j, sym in enumerate(atom_types):
            X[i, j] = counts[sym]
        y[i] = row["Energy_Ha"]

    # Fit
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Residuals and MAD-based robust z-score (k=0.6745)
    residuals = y - X @ beta
    res_median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - res_median)))
    z = 0.6745 * (residuals - res_median) / mad

    return beta, residuals, z


# -- Main ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AIM4ML Stage 2 — Energy outlier detection (OLS atom-type)."
    )
    p.add_argument("-i", "--input-dir", type=str, default="batches",
                   help="Directory of Parquet batch files (default: batches/).")
    p.add_argument("-o", "--output-dir", type=str, default="filtered_batches",
                   help="Output directory for Parquet batches (default: filtered_batches/).")
    p.add_argument("--rejects-dir", type=str, default="rejects/02_energy_prefilter",
                   help="Rejected molecules SDF directory (default: rejects/02_energy_prefilter/).")
    p.add_argument("--threshold", type=float, default=3.5,
                   help="MAD-based robust z-score threshold (default: 3.5).")
    p.add_argument("--atom-types", type=str, nargs="*",
                   default=ATOM_TYPES,
                   help="Atom types for OLS regression (default: C H N O S F Cl Br P I).")
    return p.parse_args()


def main():
    args = parse_args()
    atom_types = args.atom_types

    # -- Discover batches -------------------------------------------------
    batch_files = sorted(
        f for f in os.listdir(args.input_dir) if f.endswith(".parquet")
    )
    if not batch_files:
        print(f"No .parquet files found in {args.input_dir}")
        sys.exit(1)

    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    # -- Load all rows + reconstruct mols ---------------------------------
    all_rows = []
    total = 0
    for fname in batch_files:
        path = os.path.join(args.input_dir, fname)
        batch = read_batch(path)
        for row in batch:
            mol = Chem.MolFromMolBlock(row["mol_block"], sanitize=False,
                                       removeHs=False)
            row["mol_block"] = row["mol_block"]  # keep original
            if mol is None:
                row["_mol"] = None
            else:
                row["_mol"] = mol
            all_rows.append(row)
        total += len(batch)

    print(f"  {total} molecules loaded")

    # -- Filter rows with valid mols for fitting --------------------------
    fit_rows = [r for r in all_rows if r["_mol"] is not None]
    n_corrupt = total - len(fit_rows)
    if n_corrupt:
        print(f"  {n_corrupt} corrupt mols excluded from fit (will be marked 'mol_corrupt')")

    if len(fit_rows) == 0:
        print("Nothing to fit — aborting.")
        sys.exit(1)

    # -- Fit --------------------------------------------------------------
    print(f"\nFitting OLS  E~sum(n_i*e_i)  ({len(atom_types)} atom types: {', '.join(atom_types)}) ...")
    beta, residuals, z = fit_ols(fit_rows, atom_types)

    print("  Fitted atomic energies (Ha):")
    for sym, e in zip(atom_types, beta):
        print(f"    e_{sym:2s} = {e:12.6f}")

    res_median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - res_median)))
    print(f"  Residual median: {res_median:.4g} Ha  MAD: {mad:.4g} Ha")

    # Map fit_row index → z score, then propagate to all_rows
    fit_row_to_z = {id(r): z[i] for i, r in enumerate(fit_rows)}

    # -- Flag and write ---------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    n_ok = 0
    n_flagged = 0
    n_corrupt_flag = 0
    flagged_rows = []

    batch_offset = 0
    for fname in batch_files:
        in_path  = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        batch = read_batch(in_path)
        out_rows = []

        for pos, row in enumerate(batch):
            global_row = all_rows[batch_offset + pos]
            mol = global_row.get("_mol")
            if mol is None:
                row["energy_status"] = "mol_corrupt"
                n_corrupt_flag += 1
            else:
                z_val = fit_row_to_z.get(id(global_row))
                if z_val is not None and abs(z_val) > args.threshold:
                    row["energy_status"] = "flagged"
                    n_flagged += 1
                    flagged_rows.append(global_row)
                else:
                    row["energy_status"] = "ok"
                    n_ok += 1

            out_rows.append(row)

        # Drop internal _mol objects before writing (not serializable)
        for row in out_rows:
            row.pop("_mol", None)

        write_batch(out_path, out_rows)
        batch_offset += len(batch)

    # -- Reject SDF -------------------------------------------------------
    if flagged_rows:
        reject_path = os.path.join(args.rejects_dir, "energy_flagged.sdf")
        write_reject_sdf(reject_path, flagged_rows,
                         reject_reason=f"energy_z>{args.threshold}")
        print(f"\n  {len(flagged_rows)} flagged → {reject_path}")

    # -- Report ------------------------------------------------------------
    print(f"\nReport")
    print(f"  Total:        {total}")
    print(f"  OK:           {n_ok}")
    print(f"  Flagged:      {n_flagged}")
    print(f"  Mol corrupt:  {n_corrupt_flag}")

    if n_flagged:
        pct = 100 * n_flagged / total
        print(f"  (% flagged:   {pct:.2f}%)")

    # Sentinel for Makefile
    if n_flagged:
        sentinel = os.path.join(args.rejects_dir, ".FLAGGED")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(sentinel, "w") as f:
            f.write(f"{n_flagged} molecules flagged\n")


if __name__ == "__main__":
    main()
