#!/usr/bin/env python3
"""
06_stereo_filter.py — Stage 6: Enantiomer removal (SMILES-based).

Groups molecules by flat SMILES (stereo markers stripped).  Within each
multi-member group, checks whether any pair are enantiomers by inverting
the tetrahedral stereocenters of one molecule and comparing to the other.
Enantiomeric duplicates are flagged (first occurrence kept, rest removed).

Operates on the CanonicalSMILES column (computed by dedup).  No geometry
loading required.

Usage:
    python3 06_stereo_filter.py -i deduped_batches/ -o stereo_batches/
"""

import os
import sys
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from lib.parquet_io import read_batch, write_batch
from lib.sdf_io import write_reject_sdf


# -- Stereo helpers -------------------------------------------------------

def flat_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return smiles
    Chem.SanitizeMol(mol, catchErrors=True)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def has_tetrahedral_stereo(mol):
    return any(
        atom.GetChiralTag() in (
            ChiralType.CHI_TETRAHEDRAL_CW,
            ChiralType.CHI_TETRAHEDRAL_CCW,
        )
        for atom in mol.GetAtoms()
    )


def invert_tetrahedral_stereo(mol):
    rw = Chem.RWMol(Chem.Mol(mol))
    for atom in rw.GetAtoms():
        chi = atom.GetChiralTag()
        if chi == ChiralType.CHI_TETRAHEDRAL_CW:
            atom.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CCW)
        elif chi == ChiralType.CHI_TETRAHEDRAL_CCW:
            atom.SetChiralTag(ChiralType.CHI_TETRAHEDRAL_CW)
    return rw.GetMol()


def are_enantiomers(mol_a, mol_b):
    if not has_tetrahedral_stereo(mol_a) or not has_tetrahedral_stereo(mol_b):
        return False
    inverted_a = invert_tetrahedral_stereo(mol_a)
    stereo_a_inv = Chem.MolToSmiles(inverted_a, isomericSmiles=True)
    stereo_b = Chem.MolToSmiles(mol_b, isomericSmiles=True)
    return stereo_a_inv == stereo_b


# -- Main ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AIM4ML Stage 6 — Enantiomer filter (SMILES-based)."
    )
    p.add_argument("-i", "--input-dir", type=str, default="deduped_batches",
                   help="Input Parquet batch directory (default: deduped_batches/).")
    p.add_argument("-o", "--output-dir", type=str, default="stereo_batches",
                   help="Output directory (default: stereo_batches/).")
    p.add_argument("--rejects-dir", type=str, default="rejects/06_stereo_filter",
                   help="Rejected molecules SDF directory (default: rejects/06_stereo_filter/).")
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
    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    # -- Pass 1: accumulate rows, build flat-SMILES groups ---------------
    all_rows = []
    total_mols = 0
    mol_parse_failures = 0

    for fname in batch_files:
        path = os.path.join(args.input_dir, fname)
        batch = read_batch(path)
        for row in batch:
            all_rows.append(row)
            row["_batch_file"] = fname
        total_mols += len(batch)

    print(f"  Loaded {total_mols} molecules")

    # Build mol dict: index → mol (from CanonicalSMILES)
    flat_groups = {}  # flat_smiles → list of (row_idx, CompoundID)
    idx_to_mol = {}

    for idx, row in enumerate(all_rows):
        cid = row.get("CompoundID")
        can_smi = row.get("CanonicalSMILES")
        if not can_smi:
            row["stereo_status"] = "skipped_no_smiles"
            continue
        mol = Chem.MolFromSmiles(can_smi, sanitize=False)
        if mol is None:
            row["stereo_status"] = "smiles_unparseable"
            mol_parse_failures += 1
            continue
        # No sanitize — chirality tags are set by MolFromSmiles even without
        # sanitization. SanitizeMol would throw AtomValenceException on mols
        # with explicit-H representation (N valence 4 from dedup).
        idx_to_mol[idx] = mol
        flat = flat_smiles(can_smi)
        flat_groups.setdefault(flat, []).append((idx, cid))

    # -- Pass 2: detect enantiomer pairs within each group ----------------
    n_removed = 0
    n_kept = 0
    rejected_rows = []

    multi_groups = {k: v for k, v in flat_groups.items() if len(v) > 1}
    print(f"  Stereo groups (>1 member): {len(multi_groups)}")

    for flat_smi, members in multi_groups.items():
        n = len(members)
        keep = [True] * n

        for i in range(n):
            if not keep[i]:
                continue
            idx_i, cid_i = members[i]
            mol_i = idx_to_mol.get(idx_i)
            if mol_i is None:
                continue
            for j in range(i + 1, n):
                if not keep[j]:
                    continue
                idx_j, cid_j = members[j]
                mol_j = idx_to_mol.get(idx_j)
                if mol_j is None:
                    continue

                if are_enantiomers(mol_i, mol_j):
                    keep[j] = False
                    all_rows[idx_j]["stereo_status"] = "removed_enantiomer"
                    n_removed += 1
                    rejected_rows.append(all_rows[idx_j])

    # Mark kept molecules
    for idx, mol in idx_to_mol.items():
        if all_rows[idx].get("stereo_status") is None:
            all_rows[idx]["stereo_status"] = "kept"
            n_kept += 1

    # -- Write output batches ---------------------------------------------
    # Group rows back by original batch file, preserving order
    batch_groups = {}
    for row in all_rows:
        fname = row.pop("_batch_file")
        batch_groups.setdefault(fname, []).append(row)

    for fname, rows in batch_groups.items():
        out_path = os.path.join(args.output_dir, fname)
        write_batch(out_path, rows)

    # -- Reject SDF -------------------------------------------------------
    if rejected_rows:
        reject_path = os.path.join(args.rejects_dir, "stereo_removed.sdf")
        write_reject_sdf(reject_path, rejected_rows,
                         reject_reason="removed_enantiomer")
        print(f"  {len(rejected_rows)} enantiomers → {reject_path}")

    # -- Report ------------------------------------------------------------
    total = total_mols
    n_other = total - n_kept - n_removed - mol_parse_failures
    print(f"\nReport")
    print(f"  Total:               {total}")
    print(f"  Kept:                {n_kept}")
    print(f"  Removed enantiomers: {n_removed}")
    print(f"  SMILES unparseable:  {mol_parse_failures}")
    if n_other:
        print(f"  Other (skipped):     {n_other}")

    if n_removed:
        sentinel = os.path.join(args.rejects_dir, ".REMOVED")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(sentinel, "w") as f:
            f.write(f"{n_removed} enantiomers removed\n")


if __name__ == "__main__":
    main()
