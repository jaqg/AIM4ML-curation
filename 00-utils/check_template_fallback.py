#!/usr/bin/env python3
"""
check_template_fallback.py — QA check for molecules processed via the SMILES template fallback.

One-time validation script. Reads logs/topology_fixed.txt and logs/sdf_recovered.txt,
looks up their SMILES in qm40_mapping.csv, and classifies each molecule as:

  - SAFE:    matches aromatic-S pattern only — the expected failure class for
             DetermineBondOrders (thiophene/thiazole/thiadiazole/isothiazole).
             Template fix is reliable: no tautomerism involved.

  - FLAGGED: contains a tautomer-prone functional group (amide, enol, iminol, amidine).
             Silent wrong bond orders are possible if the DFT-optimised structure is a
             different tautomeric form than the QM40 SMILES. Requires manual inspection.
             See qm40-pipeline-plan.md §4.5 for the full explanation.

  - OTHER:   matches neither pattern — inspect manually.

Usage:
    python3 check_template_fallback.py               # local sample logs
    python3 check_template_fallback.py --full-data   # full-data cluster logs
"""

import argparse
import os
import pandas as pd
from rdkit import Chem

# --- Paths -------------------------------------------------------------------
PATHS = {
    "sample": {
        "mapping_csv": "../../samples/qm40/qm40_mapping.csv",
        "log_dir":     "../../samples/qm40/logs",
        "out_dir":     "../../samples/qm40/logs",
    },
    "full": {
        "mapping_csv": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/qm40_mapping.csv",
        "log_dir":     "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/logs",
        "out_dir":     "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/logs",
    },
}
# -----------------------------------------------------------------------------

# Molecules matching ONLY these patterns are the expected failure class —
# DetermineBondOrders fails on 5-membered aromatic S heterocycles.
# Template assignment is reliable for these (no tautomerism).
SAFE_PATTERNS = {
    "aromatic_S_ring": Chem.MolFromSmarts("[sX2,sX3]"),
}

# Molecules matching any of these patterns have functional groups that exist in
# multiple tautomeric forms with the same heavy-atom connectivity graph.
# If the DFT structure is in a different tautomeric form than the SMILES, the
# template will silently impose the wrong bond orders.
TAUTOMER_PATTERNS = {
    "amide":     Chem.MolFromSmarts("[CX3](=[OX1])[NX3]"),
    "thioamide": Chem.MolFromSmarts("[CX3](=[SX1])[NX3]"),
    "enol":      Chem.MolFromSmarts("[OX2H][CX3]=[CX3]"),
    "iminol":    Chem.MolFromSmarts("[OX2H][CX3]=[NX2]"),
    "amidine":   Chem.MolFromSmarts("[NX3][CX3]=[NX2]"),
}


def classify(smiles):
    """
    Classify a molecule by functional group.

    Returns:
        safe  (list[str]): names of matching SAFE patterns
        flags (list[str]): names of matching TAUTOMER patterns
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [], ["PARSE_ERROR"]
    safe  = [name for name, pat in SAFE_PATTERNS.items()   if mol.HasSubstructMatch(pat)]
    flags = [name for name, pat in TAUTOMER_PATTERNS.items() if mol.HasSubstructMatch(pat)]
    return safe, flags


def read_zinc_ids(path):
    """Read Zinc_ids from a log file (one per line, skipping # comment lines)."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def main():
    parser = argparse.ArgumentParser(description="QA check for SMILES-template fallback molecules.")
    parser.add_argument("--full-data", action="store_true",
                        help="Run on full-data cluster logs (default: local sample).")
    args = parser.parse_args()

    mode  = "full" if args.full_data else "sample"
    paths = PATHS[mode]

    print(f"Mode: {mode}")

    # Load SMILES lookup from mapping CSV
    if not os.path.exists(paths["mapping_csv"]):
        print(f"ERROR: mapping CSV not found: {paths['mapping_csv']}")
        return
    mapping_df = pd.read_csv(paths["mapping_csv"])
    smiles_map = dict(zip(mapping_df["Zinc_id"], mapping_df["canonical_SMILES"]))

    # Read both fallback log files
    log_files = {
        "topology_fixed.txt": os.path.join(paths["log_dir"], "topology_fixed.txt"),
        "sdf_recovered.txt":  os.path.join(paths["log_dir"], "sdf_recovered.txt"),
    }
    all_entries = []   # list of (log_name, zinc_id)
    for log_name, log_path in log_files.items():
        ids = read_zinc_ids(log_path)
        print(f"  {log_name}: {len(ids)} molecules")
        all_entries.extend((log_name, zid) for zid in ids)

    if not all_entries:
        print("\nNo template-fallback molecules found in logs. Nothing to check.")
        return

    total = len(all_entries)

    # Classify each molecule
    safe_only = []
    flagged   = []
    other     = []

    for log_name, zinc_id in all_entries:
        smi = smiles_map.get(zinc_id)
        if smi is None:
            other.append((log_name, zinc_id, "N/A", [], ["NOT_IN_MAPPING"]))
            continue
        safe, flags = classify(smi)
        entry = (log_name, zinc_id, smi, safe, flags)
        if flags:
            flagged.append(entry)
        elif safe:
            safe_only.append(entry)
        else:
            other.append(entry)

    # --- Report ---------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"Template fallback QA — {mode} mode")
    print(f"{'='*55}")
    print(f"Total template-fallback molecules:  {total}")
    print()
    print(f"  SAFE (aromatic-S only):    {len(safe_only):>6}  ({100*len(safe_only)/total:.1f}%)")
    print(f"  FLAGGED (tautomer-prone):  {len(flagged):>6}  ({100*len(flagged)/total:.1f}%)")
    print(f"  OTHER (no pattern match):  {len(other):>6}  ({100*len(other)/total:.1f}%)")

    if flagged:
        print(f"\n{'─'*55}")
        print(f"FLAGGED molecules — tautomer group breakdown:")
        group_counts = {}
        for _, _, _, _, flags in flagged:
            for g in flags:
                group_counts[g] = group_counts.get(g, 0) + 1
        for group, count in sorted(group_counts.items(), key=lambda x: -x[1]):
            print(f"  {group:<14} {count:>6}  ({100*count/total:.1f}%)")

        out_path = os.path.join(paths["out_dir"], "template_fallback_tautomer_flags.tsv")
        with open(out_path, "w") as f:
            f.write("log_file\tZinc_id\tcanonical_SMILES\ttautomer_groups\tsafe_groups\n")
            for log_name, zinc_id, smi, safe, flags in flagged:
                f.write(f"{log_name}\t{zinc_id}\t{smi}\t{','.join(flags)}\t{','.join(safe)}\n")
        print(f"\nFlagged molecules written to: {out_path}")
        print("Inspect these manually: if the DFT structure is the non-SMILES tautomeric form,")
        print("the SDF bond orders for that molecule are wrong (silent error).")
    else:
        print(f"\nNo tautomer-prone molecules in fallback group — risk is negligible.")

    if other:
        print(f"\n{'─'*55}")
        print(f"OTHER — no pattern match (first 10 shown):")
        for log_name, zinc_id, smi, _, flags in other[:10]:
            print(f"  [{log_name}] {zinc_id}  {smi}  flags={flags}")
        if len(other) > 10:
            print(f"  ... and {len(other) - 10} more")


if __name__ == "__main__":
    main()
