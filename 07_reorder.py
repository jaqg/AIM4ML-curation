#!/usr/bin/env python3
"""
07_reorder.py — Stage 7: Canonical atom reordering.

Reorders atoms in mol_block via one of two backends:
  amber  — AMBER antechamber (MACE/NequIP/SchNetPack standard, cluster)
  rdkit  — RDKit CanonicalRankAtoms (pure Python, portable)

Updates mol_block in-place and adds a 'reorder_status' column ("ok" / "failed").

Usage:
    python3 07_reorder.py -i stereo_batches/ -o reordered_batches/ --backend rdkit
    python3 07_reorder.py -i stereo_batches/ -o reordered_batches/ --backend amber
"""

import os
import sys
import argparse
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from lib.parquet_io import read_batch, write_batch
from lib.sdf_io import write_reject_sdf


# -- RDKit backend --------------------------------------------------------

def reorder_rdkit(mol_block):
    """Canonical rank atoms via RDKit, renumber, return new mol_block."""
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    if mol is None:
        return None
    # Chem.CanonicalRankAtoms: include isotopes, break ties with coords
    ranking = list(Chem.CanonicalRankAtoms(
        mol, breakTies=True, includeChirality=True, includeIsotopes=True,
    ))
    mol_reordered = Chem.RenumberAtoms(mol, ranking)
    block = Chem.MolToMolBlock(mol_reordered)
    m_end = block.index("M  END") + len("M  END")
    return block[:m_end]


# -- AMBER backend --------------------------------------------------------

def _mol_block_to_xyz(mol_block, path):
    """Write a temporary XYZ file from mol_block."""
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    if mol is None:
        return False
    conf = mol.GetConformer()
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    with open(path, "w") as f:
        f.write(f"{len(atoms)}\n")
        f.write("temporary\n")
        for i, sym in enumerate(atoms):
            pos = conf.GetAtomPosition(i)
            f.write(f"{sym:2s}  {pos.x:.6f}  {pos.y:.6f}  {pos.z:.6f}\n")
    return True


def _xyz_to_mol_block(xyz_path, ref_mol):
    """Read reordered XYZ, rebuild mol_block with correct ordering."""
    with open(xyz_path) as f:
        lines = f.readlines()
    nat = int(lines[0].strip())
    new_order = []
    for line in lines[2:2 + nat]:
        parts = line.split()
        new_order.append(parts[0])  # element symbol

    # Map old atom idx → new atom idx based on element ordering
    ref_atoms = [atom.GetSymbol() for atom in ref_mol.GetAtoms()]
    # Find permutation: for each position in new_order, find which old atom matches
    # This is fragile — antechamber may change atom order in complex ways.
    # Simple approach: if element lists match as multisets, renumber sequentially.
    from collections import Counter
    if Counter(new_order) != Counter(ref_atoms):
        return None  # reordering changed elements — can't reconstruct

    # Greedy mapping: for each element type, pair old→new in order
    old_by_elem = {}  # element → list of old indices
    for i, sym in enumerate(ref_atoms):
        old_by_elem.setdefault(sym, []).append(i)

    ranking = [0] * nat
    seen = {}
    for new_i, sym in enumerate(new_order):
        old_list = old_by_elem[sym]
        pos = seen.get(sym, 0)
        if pos >= len(old_list):
            return None
        ranking[old_list[pos]] = new_i
        seen[sym] = pos + 1

    mol_reordered = Chem.RenumberAtoms(ref_mol, ranking)
    block = Chem.MolToMolBlock(mol_reordered)
    m_end = block.index("M  END") + len("M  END")
    return block[:m_end]


# -- AMBER backend --------------------------------------------------------

_SCRIPT_LIB = os.path.join(_SCRIPT_DIR, "lib")
_AMBER_SCRIPT = os.path.join(_SCRIPT_LIB, "antechamber_xyz_reord.sh")


def reorder_amber(mol_block, timeout=120):
    """Reorder via antechamber + obabel (bundled shell script).

    Returns new mol_block or None on failure.
    """
    if not os.path.exists(_AMBER_SCRIPT):
        print(f"  ERROR: antechamber script not found at {_AMBER_SCRIPT}",
              file=sys.stderr)
        return None

    ref_mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    if ref_mol is None:
        return None

    with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w",
                                     delete=False) as f:
        _mol_block_to_xyz(mol_block, f.name)
        xyz_path = f.name

    try:
        result = subprocess.run(
            ["bash", _AMBER_SCRIPT, xyz_path],
            capture_output=True, timeout=timeout,
        )
        # Shell script returns 0 on success, may rename xyz on failure
        if result.returncode != 0 or not os.path.exists(xyz_path):
            return None
        new_block = _xyz_to_mol_block(xyz_path, ref_mol)
        return new_block
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    finally:
        for suffix in ("", "_ORIG"):
            p = xyz_path + suffix
            if os.path.exists(p):
                os.unlink(p)


# -- Main ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AIM4ML Stage 7 — Canonical atom reordering."
    )
    p.add_argument("-i", "--input-dir", type=str, default="stereo_batches",
                   help="Input Parquet batch directory (default: stereo_batches/).")
    p.add_argument("-o", "--output-dir", type=str, default="reordered_batches",
                   help="Output directory (default: reordered_batches/).")
    p.add_argument("--rejects-dir", type=str, default="rejects/07_reorder",
                   help="Rejected molecules SDF (default: rejects/07_reorder/).")
    p.add_argument("--backend", type=str, default="rdkit",
                   choices=["rdkit", "amber"],
                   help="Reordering backend: rdkit (default) or amber.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (default: 1, for amber backend).")
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
    print(f"Backend: {args.backend}")
    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    reorder_fn = reorder_rdkit if args.backend == "rdkit" else reorder_amber
    total_ok = 0
    total_failed = 0
    failed_rows = []

    # Worker function for parallel reordering
    def _reorder_row(row):
        new_block = reorder_fn(row["mol_block"])
        if new_block is None:
            row["reorder_status"] = "failed"
        else:
            row["mol_block"] = new_block
            row["reorder_status"] = "ok"
        return row

    for fname in batch_files:
        in_path  = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        batch = read_batch(in_path)

        if args.workers > 1:
            out_rows = []
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(_reorder_row, row) for row in batch]
                for future in as_completed(futures):
                    out_rows.append(future.result())
        else:
            out_rows = [_reorder_row(row) for row in batch]

        for row in out_rows:
            if row["reorder_status"] == "ok":
                total_ok += 1
            else:
                total_failed += 1
                failed_rows.append(row)

        write_batch(out_path, out_rows)

    # -- Reject SDF -------------------------------------------------------
    if failed_rows:
        reject_path = os.path.join(args.rejects_dir, "reorder_failed.sdf")
        write_reject_sdf(reject_path, failed_rows,
                         reject_reason="reorder_failed")
        print(f"  {len(failed_rows)} failed → {reject_path}")

    # -- Report ------------------------------------------------------------
    total = total_ok + total_failed
    print(f"\nReport")
    print(f"  Total:   {total}")
    print(f"  OK:      {total_ok}")
    print(f"  Failed:  {total_failed}")

    if total_failed:
        sentinel = os.path.join(args.rejects_dir, ".FAILED")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(sentinel, "w") as f:
            f.write(f"{total_failed} molecules failed\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
