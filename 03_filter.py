#!/usr/bin/env python3
"""
03_filter.py — Stage 3: Chemical filter (neutral / non-zwitterion / closed-shell).

For every molecule in the input Parquet batches:
  1. Reconstructs the RDKit Mol from mol_block (unsanitized).
  2. Assigns bond orders from 3D geometry via DetermineBondOrders.
  3. Sanitizes the molecule to obtain per-atom formal charges and radical counts.
  4. Checks three criteria:
      - Neutral:         net formal charge == 0
      - Non-zwitterion:  no atom carries a non-zero formal charge
      - Closed-shell:    radical electrons == 0
  5. Writes a filter_status column ("ok" / "rejected") and updates the
     mol_block with the sanitized representation (correct bond types).

Usage:
    python3 03_filter.py -i filtered_batches/ -o curated_batches/
"""

import os
import sys
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from lib.parquet_io import read_batch, write_batch
from lib.sdf_io import write_reject_sdf


# -- Presets --------------------------------------------------------------

PRESETS = {
    "neutral_closed_shell": {
        "charge":       True,   # net formal charge == 0
        "zwitterion":   True,   # no per-atom formal charges
        "radicals":     True,   # radical electrons == 0
    },
    "neutral": {
        "charge":       True,
        "zwitterion":   False,
        "radicals":     False,
    },
    "none": {
        "charge":       False,
        "zwitterion":   False,
        "radicals":     False,
    },
}


# -- Filter logic ---------------------------------------------------------

def process_molecule(row, checks):
    """
    Process a single molecule row: reconstruct mol, determine bonds,
    sanitize, check enabled criteria.

    Parameters
    ----------
    row : dict
        Parquet row with mol_block, FormalCharge.
    checks : dict
        Boolean flags for 'charge', 'zwitterion', 'radicals'.

    Returns
    -------
    (mol, filter_status, reason)
    """
    mol = Chem.MolFromMolBlock(row["mol_block"], sanitize=False,
                               removeHs=False)
    if mol is None:
        return None, "mol_corrupt", "could not parse mol_block"

    # Step 1: determine bond orders from 3D geometry
    try:
        rdDetermineBonds.DetermineBonds(mol, charge=int(row["FormalCharge"]))
    except Exception:
        # If bond determination fails, we still try to sanitize and check
        pass

    # Step 2: sanitize to compute formal charges, radicals, aromaticity
    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:
        return mol, "mol_corrupt", "sanitization failed"

    # Step 3: check enabled criteria
    atoms = list(mol.GetAtoms())

    if checks.get("charge"):
        net_charge = sum(atom.GetFormalCharge() for atom in atoms)
        if net_charge != 0:
            return mol, "rejected", f"charge={net_charge:+d}"

    if checks.get("zwitterion"):
        if any(atom.GetFormalCharge() != 0 for atom in atoms):
            return mol, "rejected", "zwitterion"

    if checks.get("radicals"):
        radicals = sum(atom.GetNumRadicalElectrons() for atom in atoms)
        if radicals > 0:
            return mol, "rejected", f"radicals={radicals}"

    return mol, "ok", ""


# -- Main ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="AIM4ML Stage 3 — Chemical filter (neutral / non-zwitterion / closed-shell)."
    )
    p.add_argument("-i", "--input-dir", type=str, default="filtered_batches",
                   help="Input Parquet batch directory (default: filtered_batches/).")
    p.add_argument("-o", "--output-dir", type=str, default="curated_batches",
                   help="Output directory (default: curated_batches/).")
    p.add_argument("--rejects-dir", type=str, default="rejects/03_filter",
                   help="Rejected molecules SDF directory (default: rejects/03_filter/).")
    p.add_argument("--preset", type=str, default="neutral_closed_shell",
                   choices=list(PRESETS.keys()),
                   help="Filter preset: neutral_closed_shell (default), neutral, none.")
    return p.parse_args()


def main():
    args = parse_args()
    checks = PRESETS[args.preset]

    batch_files = sorted(
        f for f in os.listdir(args.input_dir) if f.endswith(".parquet")
    )
    if not batch_files:
        print(f"No .parquet files found in {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Preset: {args.preset}")
    print(f"Input batches: {len(batch_files)} files in {args.input_dir}")

    total_ok = 0
    total_rejected = 0
    total_corrupt = 0
    rejected_rows = []
    reason_counts = {}

    for fname in batch_files:
        in_path  = os.path.join(args.input_dir, fname)
        out_path = os.path.join(args.output_dir, fname)
        batch = read_batch(in_path)
        out_rows = []

        for row in batch:
            mol, status, reason = process_molecule(row, checks)
            row["filter_status"] = status
            if reason:
                row["filter_reason"] = reason

            if status == "ok":
                # Update mol_block with sanitized representation
                if mol is not None:
                    try:
                        block = Chem.MolToMolBlock(mol)
                        m_end = block.index("M  END") + len("M  END")
                        row["mol_block"] = block[:m_end]
                    except Exception:
                        pass  # keep original mol_block if serialization fails
                row["num_atoms"] = mol.GetNumAtoms() if mol else row["num_atoms"]
                row["num_bonds"] = mol.GetNumBonds() if mol else row["num_bonds"]
                total_ok += 1
            elif status == "rejected":
                total_rejected += 1
                rejected_rows.append(row)
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            else:
                total_corrupt += 1

            out_rows.append(row)

        write_batch(out_path, out_rows)

    # -- Reject SDF -------------------------------------------------------
    if rejected_rows:
        reject_path = os.path.join(args.rejects_dir, "filter_rejected.sdf")
        write_reject_sdf(reject_path, rejected_rows,
                         reject_reason="filter_rejected")
        print(f"  {len(rejected_rows)} rejected → {reject_path}")

    # -- Report ------------------------------------------------------------
    total = total_ok + total_rejected + total_corrupt
    print(f"\nReport")
    print(f"  Total:        {total}")
    print(f"  OK:           {total_ok} ({100*total_ok/total:.2f}%)" if total else "  OK: 0")
    print(f"  Rejected:     {total_rejected} ({100*total_rejected/total:.2f}%)" if total else "  Rejected: 0")
    print(f"  Mol corrupt:  {total_corrupt}")

    if reason_counts:
        print(f"\nRejection breakdown:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    if total_rejected:
        sentinel = os.path.join(args.rejects_dir, ".REJECTED")
        os.makedirs(args.rejects_dir, exist_ok=True)
        with open(sentinel, "w") as f:
            f.write(f"{total_rejected} molecules rejected\n")


if __name__ == "__main__":
    main()
