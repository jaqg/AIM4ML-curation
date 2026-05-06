#!/usr/bin/env python3
"""
filter_qm40.py — Phase 1 curation filter for QM40.

Criteria:
    - Neutral:      formal charge == 0       (RDKit, from canonical SMILES)
    - Closed-shell: radical electrons == 0   (RDKit, from canonical SMILES)

Writes:
    filtered_main.csv         — molecules passing all criteria
    logs/filter_rejected.csv  — rejected molecules with reason code

Usage:
    python3 filter_qm40.py               # local sample (default)
    python3 filter_qm40.py --full-data   # full dataset on cluster
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

PATHS = {
    "sample": {
        "main_csv":      "../../samples/qm40/sample_main.csv",
        "filtered_csv":  "../../samples/qm40/filtered_sample_main.csv",
        "log_dir":       "../../samples/qm40/logs",
    },
    "full": {
        "main_csv":      "/datos_pool/mldata1/QMdatasets/QM40/main.csv",
        "filtered_csv":  "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/filtered_main.csv",
        "log_dir":       "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/logs",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1 curation filter for QM40.")
    parser.add_argument("--full-data", action="store_true",
                        help="Run on the full QM40 dataset on the cluster (default: local sample).")
    return parser.parse_args()


def check_molecule(smiles):
    """
    Apply filter criteria to a SMILES string.

    Returns (passes: bool, reason: str).
    reason is '' when passes=True, or a short code when passes=False:
      'smiles_unparseable' — RDKit could not parse the SMILES
      'charge=+N'          — molecule carries a net formal charge
      'radicals=N'         — molecule has unpaired electrons (open-shell)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "smiles_unparseable"

    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    if charge != 0:
        return False, f"charge={charge:+d}"

    radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    if radicals > 0:
        return False, f"radicals={radicals}"

    return True, ""


def main():
    args = parse_args()
    mode = "full" if args.full_data else "sample"
    paths = PATHS[mode]

    MAIN_CSV     = paths["main_csv"]
    FILTERED_CSV = paths["filtered_csv"]
    LOG_DIR      = paths["log_dir"]

    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"Mode: {mode}")
    print(f"Reading {MAIN_CSV} ...")
    main_df = pd.read_csv(MAIN_CSV)
    n_total = len(main_df)
    print(f"  {n_total} molecules")

    passed   = []
    rejected = []

    print(f"\nApplying filter (neutral + closed-shell) ...")
    for _, row in tqdm(main_df.iterrows(), total=n_total, desc="Filtering", unit="mol", file=sys.stdout):
        ok, reason = check_molecule(row["smile"])
        if ok:
            passed.append(row)
        else:
            rejected.append({"Zinc_id": row["Zinc_id"], "smile": row["smile"], "reason": reason})

    passed_df   = pd.DataFrame(passed, columns=main_df.columns)
    rejected_df = pd.DataFrame(rejected, columns=["Zinc_id", "smile", "reason"])

    passed_df.to_csv(FILTERED_CSV, index=False)

    n_pass   = len(passed_df)
    n_reject = len(rejected_df)

    print(f"\nDone.")
    print(f"  Passed:   {n_pass} ({100 * n_pass / n_total:.2f}%)")
    print(f"  Rejected: {n_reject} ({100 * n_reject / n_total:.2f}%)")
    print(f"  Filtered → {FILTERED_CSV}")

    if n_reject:
        rejected_path = os.path.join(LOG_DIR, "filter_rejected.csv")
        rejected_df.to_csv(rejected_path, index=False)
        print(f"  Rejected → {rejected_path}")
        for reason, count in rejected_df["reason"].value_counts().items():
            print(f"    {reason}: {count}")
    else:
        print(f"  All molecules passed — no rejected log written.")


if __name__ == "__main__":
    main()
