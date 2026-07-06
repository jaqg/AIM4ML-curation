#!/usr/bin/env python3
"""
04_dedup.py — Stage 4: Compute canonical SMILES, assign CompoundID,
             and remove conformer duplicates.

For each molecule in the input Parquet batches:
  1. Reconstructs mol from mol_block (bonds already set by filter stage).
  2. Computes canonical SMILES via RDKit (kekulization → SMILES).
  3. If kekulization fails and a SMILES tag is present, runs template
     fallback via AssignBondOrdersFromTemplate.
  4. Computes CompoundID = MD5(canonical SMILES).
  5. Detects conformer duplicates: molecules sharing the same CompoundID
     (keep the first occurrence, flag the rest).
  6. Updates mol_block with corrected bond representation on success.

Usage:
    python3 04_dedup.py -i curated_batches/ -o deduped_batches/
"""

import os
import sys
import argparse
import hashlib
import math

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

from lib.parquet_io import read_batch, write_batch
from lib.sdf_io import write_reject_sdf


# -- Core logic -----------------------------------------------------------

def canonicalize_and_assign(mol_block, smiles_tag=None):
    """
    Compute canonical SMILES from mol_block.  Falls back to SMILES template
    if kekulization fails.

    Returns (canonical_smiles, updated_mol_block, status, reason).
      canonical_smiles  — canonical SMILES string, or None on failure
      updated_mol_block — mol block with corrected bonds (or original on failure)
      status            — "ok" | "bond_assignment_failed" | "mol_corrupt"
      reason            — human-readable detail
    """
    mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
    if mol is None:
        return None, mol_block, "mol_corrupt", "could not parse mol_block"

    # -- Primary path: canonicalize from existing bonds --------------------
    try:
        can_smi = Chem.MolToSmiles(mol)
        # Successful — update mol_block with canonicalized mol
        new_block = _mol_to_block(mol)
        return can_smi, new_block, "ok", ""
    except Exception:
        pass  # kekulization failed → try template fallback

    # -- Template fallback (requires SMILES tag) --------------------------
    if smiles_tag is None or (isinstance(smiles_tag, float) and math.isnan(smiles_tag)) or smiles_tag == "":
        return None, mol_block, "bond_assignment_failed", \
               "kekulize failed, no SMILES for template"

    template = Chem.MolFromSmiles(smiles_tag)
    if template is None:
        return None, mol_block, "bond_assignment_failed", \
               "SMILES tag unparseable"

    try:
        # AssignBondOrdersFromTemplate may produce warnings for symmetric mols;
        # SanitizeMol catches bad results.
        mol_assigned = AllChem.AssignBondOrdersFromTemplate(template, mol)
        Chem.SanitizeMol(mol_assigned)
        can_smi = Chem.MolToSmiles(mol_assigned)
        new_block = _mol_to_block(mol_assigned)
        return can_smi, new_block, "ok", ""
    except Exception:
        return None, mol_block, "bond_assignment_failed", \
               "template fallback failed"


def _mol_to_block(mol):
    """Extract V2000 mol block from an RDKit Mol."""
    block = Chem.MolToMolBlock(mol)
    m_end = block.index("M  END") + len("M  END")
    return block[:m_end]


# -- Main ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AIM4ML Stage 4 — Dedup: canonical SMILES + CompoundID + conformer removal."
    )
    p.add_argument("-i", "--input-dir", type=str, default="curated_batches",
                   help="Input Parquet batch directory (default: curated_batches/).")
    p.add_argument("-o", "--output-dir", type=str, default="deduped_batches",
                   help="Output directory (default: deduped_batches/).")
    p.add_argument("--rejects-dir", type=str, default="rejects/04_dedup",
                   help="Rejected molecules SDF directory (default: rejects/04_dedup/).")
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

    seen_ids = set()
    total_ok = 0
    total_dup = 0
    total_fail = 0
    total_corrupt = 0
    rejected_rows = []

    for fname in batch_files:
        in_path  = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        batch = read_batch(in_path)
        out_rows = []

        for row in batch:
            smiles_raw = row.get("SMILES")
            can_smi, new_block, status, reason = canonicalize_and_assign(
                row["mol_block"], smiles_raw,
            )

            if status == "ok":
                cid = hashlib.md5(can_smi.encode()).hexdigest()
                row["CanonicalSMILES"] = can_smi
                row["CompoundID"] = cid
                row["mol_block"] = new_block

                if cid in seen_ids:
                    row["dedup_status"] = "conformer_duplicate"
                    total_dup += 1
                    rejected_rows.append(row)
                else:
                    row["dedup_status"] = "ok"
                    row["ICONF"] = 1
                    seen_ids.add(cid)
                    total_ok += 1
            elif status == "mol_corrupt":
                row["dedup_status"] = "mol_corrupt"
                total_corrupt += 1
                rejected_rows.append(row)
            else:  # bond_assignment_failed
                row["dedup_status"] = status
                row["dedup_reason"] = reason
                total_fail += 1
                rejected_rows.append(row)

            out_rows.append(row)

        write_batch(out_path, out_rows)

    # -- Reject SDF -------------------------------------------------------
    if rejected_rows:
        reject_path = os.path.join(args.rejects_dir, "dedup_rejected.sdf")
        write_reject_sdf(reject_path, rejected_rows,
                         reject_reason="dedup_rejected")
        print(f"  {len(rejected_rows)} rejected → {reject_path}")

    # -- Report ------------------------------------------------------------
    total = total_ok + total_dup + total_fail + total_corrupt
    print(f"\nReport")
    print(f"  Total:                 {total}")
    print(f"  Unique (ok):           {total_ok}")
    print(f"  Conformer duplicates:  {total_dup}")
    print(f"  Bond assignment fail:  {total_fail}")
    print(f"  Mol corrupt:           {total_corrupt}")

    if total_dup or total_fail:
        sentinel = os.path.join(args.rejects_dir, ".REJECTED")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(sentinel, "w") as f:
            f.write(f"{total_dup + total_fail} molecules rejected\n")


if __name__ == "__main__":
    main()
