#!/usr/bin/env python3
"""
convert_qm40.py — Convert raw QM40 data to pipeline-standard input SDF.

Reads QM40's three raw CSVs (main, xyz, bond) and writes a single SDF file
with standardised property tags per the AIM4ML curation pipeline contract.

Input (raw QM40):
    main.csv  — Zinc_id, smile, Internal_E(0K), HOMO, LUMO, HL_gap, ...
    xyz.csv   — Zinc_id, atom, final_x/y/z, charge (Mulliken)
    bond.csv  — Zinc_id, atom1, atom2 (1-based connectivity)

Output:
    A single SDF with every molecule. Properties written as > <TAG> data fields:
        Energy_Ha, FormalCharge, Multiplicity   (required)
        SMILES, SourceID                         (recommended)
        HOMO_Ha, LUMO_Ha, HL_Gap_Ha,            (optional pass-through)
        PartialCharges                           (optional pass-through)

Molecule construction:
    Atoms + 3D geometry from xyz.csv (final_x/y/z, DFT-optimised).
    Connectivity from bond.csv written as SINGLE bonds (type 1).
    Bond orders are NOT resolved here — the pipeline's dedup stage handles
    DetermineBondOrders + SMILES-template fallback later.

Usage:
    python3 convert_qm40.py                          # sample (200 mols)
    python3 convert_qm40.py --sample                 # same
    python3 convert_qm40.py --full-data              # full QM40
    python3 convert_qm40.py --local                  # local sample copy
    python3 convert_qm40.py --full-data --output /path/to/output.sdf
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import SDWriter
from rdkit.Geometry import rdGeometry

# -- Paths ---------------------------------------------------------------

_BASE_SAMPLE = "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples"

_LOCAL_SAMPLE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "samples"
)

PATHS = {
    "sample": {
        "main_csv": os.path.join(_BASE_SAMPLE, "filtered_sample_main.csv"),
        "xyz_csv":  os.path.join(_BASE_SAMPLE, "sample_xyz.csv"),
        "bond_csv": os.path.join(_BASE_SAMPLE, "sample_bond.csv"),
    },
    "full": {
        "main_csv": "/datos_pool/mldata1/QMdatasets/QM40/main.csv",
        "xyz_csv":  "/datos_pool/mldata1/QMdatasets/QM40/xyz.csv",
        "bond_csv": "/datos_pool/mldata1/QMdatasets/QM40/bond.csv",
    },
    "local": {
        "main_csv": os.path.join(_LOCAL_SAMPLE, "sample_main.csv"),
        "xyz_csv":  os.path.join(_LOCAL_SAMPLE, "sample_xyz.csv"),
        "bond_csv": os.path.join(_LOCAL_SAMPLE, "sample_bond.csv"),
    },
}

# -- Helpers -------------------------------------------------------------

def build_mol(atoms, coords, bond_pairs):
    """
    Build RDKit Mol from atom elements, 3D coords, and 1-based bond pairs.

    Bond pairs are written as SINGLE bonds (type 1). Bond orders are not
    resolved — the pipeline's dedup stage handles that later.

    Returns mol or None on failure.
    """
    try:
        mol = Chem.RWMol()
        for el in atoms:
            mol.AddAtom(Chem.Atom(el))

        conf = Chem.Conformer(len(atoms))
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, rdGeometry.Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

        if bond_pairs is not None:
            for a1, a2 in bond_pairs:
                mol.AddBond(a1 - 1, a2 - 1, Chem.BondType.SINGLE)

        return mol.GetMol()
    except Exception:
        return None


def compute_formal_charge(charges):
    """Sum per-atom Mulliken charges, round to nearest integer."""
    return int(round(sum(charges)))


# -- Main ----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert raw QM40 CSVs to pipeline-standard input SDF."
    )
    p.add_argument("--full-data", action="store_true",
                   help="Run on full QM40 dataset (default: sample).")
    p.add_argument("--sample", action="store_true",
                   help="Run on sample dataset (default).")
    p.add_argument("--local", action="store_true",
                   help="Run on local sample files in ../samples/.")
    p.add_argument("--output", type=str, default=None,
                   help="Output SDF path (default: <mode>/qm40_input.sdf).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.local:
        mode = "local"
    elif args.full_data:
        mode = "full"
    else:
        mode = "sample"
    paths = PATHS[mode]

    # Resolve output path
    if args.output:
        out_sdf = args.output
    else:
        out_dir = os.path.dirname(paths["main_csv"])
        out_sdf = os.path.join(out_dir, "qm40_input.sdf")

    out_dir = os.path.dirname(out_sdf)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # -- Load CSVs -------------------------------------------------------
    print(f"Mode: {mode}")
    print(f"Reading {paths['main_csv']} ...")
    main_df = pd.read_csv(paths["main_csv"])

    # Honour energy prefilter (already run on full dataset)
    if "energy_status" in main_df.columns:
        n_before = len(main_df)
        main_df = main_df[main_df["energy_status"] == "ok"].copy()
        print(f"  Energy prefilter: {n_before - len(main_df)} removed "
              f"({len(main_df)} kept)")

    n_rows = len(main_df)
    print(f"  {n_rows} molecules to convert")

    print(f"Reading {paths['xyz_csv']} ...")
    xyz_df = pd.read_csv(paths["xyz_csv"])
    xyz_groups = {zid: grp for zid, grp in xyz_df.groupby("Zinc_id", sort=False)}

    print(f"Reading {paths['bond_csv']} ...")
    bond_df = pd.read_csv(paths["bond_csv"])
    bond_groups = {
        zid: list(zip(
            grp["atom1"].astype(int),
            grp["atom2"].astype(int),
        ))
        for zid, grp in bond_df.groupby("Zinc_id", sort=False)
    }

    # -- Convert ---------------------------------------------------------
    print(f"\nWriting {out_sdf} ...")
    writer = SDWriter(out_sdf)
    written = 0
    skipped = 0

    for _, row in tqdm(main_df.iterrows(), total=n_rows,
                       desc="Converting", unit="mol", file=sys.stdout):
        zinc_id = row["Zinc_id"]
        raw_smi = row["smile"]

        if zinc_id not in xyz_groups:
            skipped += 1
            continue

        grp = xyz_groups[zinc_id]
        atoms   = grp["atom"].tolist()
        coords  = list(zip(
            grp["final_x"].astype(float),
            grp["final_y"].astype(float),
            grp["final_z"].astype(float),
        ))
        charges = grp["charge"].astype(float).tolist()
        bond_pairs = bond_groups.get(zinc_id, None)

        mol = build_mol(atoms, coords, bond_pairs)
        if mol is None:
            skipped += 1
            continue

        # Set standard SDF tags
        mol.SetProp("_Name", zinc_id)
        mol.SetProp("Energy_Ha",     f"{row['Internal_E(0K)']:.8f}")
        mol.SetProp("FormalCharge",  str(compute_formal_charge(charges)))
        mol.SetProp("Multiplicity",  "1")
        mol.SetProp("SMILES",        raw_smi)
        mol.SetProp("SourceID",      zinc_id)
        mol.SetProp("HOMO_Ha",       f"{row['HOMO']:.6f}")
        mol.SetProp("LUMO_Ha",       f"{row['LUMO']:.6f}")
        mol.SetProp("HL_Gap_Ha",     f"{row['HL_gap']:.6f}")
        mol.SetProp("PartialCharges", " ".join(f"{q:.6f}" for q in charges))

        writer.write(mol)
        written += 1

    writer.close()

    # -- Report ----------------------------------------------------------
    print(f"\nDone.")
    print(f"  Written:    {written}")
    print(f"  Skipped:    {skipped}")
    print(f"  Output:     {out_sdf}")


if __name__ == "__main__":
    main()
