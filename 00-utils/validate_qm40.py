#!/usr/bin/env python3
"""
validate_qm40.py — Data integrity validator for the QM40 pipeline output.

Reads the raw source files (xyz.csv, bond.csv) and the pipeline output
(mol_files/, qm40_mapping.csv) and asserts they agree atom-by-atom.

Checks performed:
    1.  File existence    — every row in the mapping has an XYZ and SDF on disk
    2.  NAT consistency   — mapping NAT matches line 1 of the XYZ file
    3.  Atom ordering     — elements in output XYZ match xyz.csv order exactly
    4.  Coordinates       — output XYZ coords match xyz.csv final_x/y/z (±COORD_TOL)
    5.  Bond count        — SDF bond count matches bond.csv bond count
    6.  Mulliken charges  — SDF MullikenCharges field matches xyz.csv charges (±CHARGE_TOL)
    7.  Charge sum        — sum of Mulliken charges ≈ 0 for neutral molecules (±QSUM_TOL)
    8.  SMILES            — SMILES field in SDF matches canonical_SMILES in mapping
    9.  ID uniqueness     — no MD5 hash collisions in mapping

Usage:
    python3 validate_qm40.py               # validate sample output (default)
    python3 validate_qm40.py --full-data   # validate full QM40 output on cluster
"""

import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

COORD_TOL  = 1e-4   # Å  — tolerance for coordinate comparison
CHARGE_TOL = 1e-4   # e  — tolerance for per-atom Mulliken charge comparison
QSUM_TOL   = 0.05   # e  — tolerance for total Mulliken charge sum (neutral mol)

PATHS = {
    "sample": {
        "xyz_csv":     "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/sample_xyz.csv",
        "bond_csv":    "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/sample_bond.csv",
        "mapping_csv": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/qm40_mapping.csv",
        "mol_dir":     "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/mol_files",
    },
    "full": {
        "xyz_csv":     "/datos_pool/mldata1/QMdatasets/QM40/xyz.csv",
        "bond_csv":    "/datos_pool/mldata1/QMdatasets/QM40/bond.csv",
        "mapping_csv": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/qm40_mapping.csv",
        "mol_dir":     "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/mol_files",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate QM40 pipeline output.")
    parser.add_argument(
        "--full-data", action="store_true",
        help="Validate the full QM40 output on the cluster (default: sample).",
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Validate sample dataset output (cluster absolute paths, default).",
    )
    return parser.parse_args()


def read_xyz(path):
    """Return (nat, atoms, coords) from an XYZ file."""
    with open(path) as f:
        lines = f.readlines()
    nat = int(lines[0].strip())
    atoms, coords = [], []
    for line in lines[2:2 + nat]:
        parts = line.split()
        atoms.append(parts[0])
        coords.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return nat, atoms, coords


def read_sdf(path):
    """Return RDKit mol from an SDF file (keeps explicit H for charge check)."""
    supplier = Chem.SDMolSupplier(path, removeHs=False)
    return next(iter(supplier), None)


def check(condition, msg, errors):
    """Append msg to errors if condition is False."""
    if not condition:
        errors.append(msg)


def main():
    args  = parse_args()
    mode = "full" if args.full_data else "sample"
    paths = PATHS[mode]

    print(f"Mode: {mode}")

    xyz_df  = pd.read_csv(paths["xyz_csv"])
    bond_df = pd.read_csv(paths["bond_csv"])
    map_df  = pd.read_csv(paths["mapping_csv"])

    # Pre-group for O(1) lookup
    xyz_groups  = {zid: grp for zid, grp in xyz_df.groupby("Zinc_id",  sort=False)}
    bond_counts = bond_df.groupby("Zinc_id").size().to_dict()

    n_total      = len(map_df)
    n_sdf_failed = (map_df.get("sdf_status", pd.Series(["ok"] * n_total)) == "sdf_failed").sum()
    failures     = []   # list of (zinc_id, [error messages])

    for _, row in tqdm(map_df.iterrows(), total=n_total, desc="Validating", unit="mol"):
        zinc_id    = row["Zinc_id"]
        cid        = row["ID"]
        iconf      = int(row["ICONF"])
        nat_map    = int(row["NAT"])
        can_smi    = row["canonical_SMILES"]
        sdf_status = row.get("sdf_status", "ok")

        xyz_path = os.path.join(paths["mol_dir"], f"mol_{cid}_{iconf}.xyz")
        sdf_path = os.path.join(paths["mol_dir"], f"mol_{cid}_{iconf}.sdf")
        errs = []

        # ------------------------------------------------------------------
        # Check 1 — file existence
        # ------------------------------------------------------------------
        xyz_exists = os.path.exists(xyz_path)
        sdf_exists = os.path.exists(sdf_path)
        check(xyz_exists, "XYZ file missing", errs)
        # SDF may be legitimately absent when dedup marked it sdf_failed
        if sdf_status != "sdf_failed":
            check(sdf_exists, "SDF file missing", errs)

        if not xyz_exists:
            failures.append((zinc_id, errs))
            continue

        # ------------------------------------------------------------------
        # Read output XYZ
        # ------------------------------------------------------------------
        nat_xyz, atoms_xyz, coords_xyz = read_xyz(xyz_path)

        # Check 2 — NAT consistency
        check(nat_xyz == nat_map,
              f"NAT mismatch: mapping={nat_map}, XYZ line 1={nat_xyz}", errs)

        # ------------------------------------------------------------------
        # Cross-check against xyz.csv
        # ------------------------------------------------------------------
        if zinc_id not in xyz_groups:
            errs.append("Zinc_id not found in xyz.csv — cannot validate atoms/coords/charges")
            failures.append((zinc_id, errs))
            continue

        grp         = xyz_groups[zinc_id]
        ref_atoms   = grp["atom"].tolist()
        ref_coords  = list(zip(
            grp["final_x"].astype(float),
            grp["final_y"].astype(float),
            grp["final_z"].astype(float),
        ))
        ref_charges = grp["charge"].astype(float).tolist()

        # Check 3 — atom ordering (elements must match xyz.csv in order)
        check(atoms_xyz == ref_atoms,
              f"Atom ordering mismatch: output={atoms_xyz[:5]}... ref={ref_atoms[:5]}...",
              errs)

        # Check 4 — coordinate accuracy (per-atom, stop at first mismatch)
        for i, ((x, y, z), (rx, ry, rz)) in enumerate(zip(coords_xyz, ref_coords)):
            if abs(x - rx) > COORD_TOL or abs(y - ry) > COORD_TOL or abs(z - rz) > COORD_TOL:
                errs.append(
                    f"Coordinate mismatch at atom {i+1}: "
                    f"({x:.6f},{y:.6f},{z:.6f}) vs ({rx:.6f},{ry:.6f},{rz:.6f})"
                )
                break

        # ------------------------------------------------------------------
        # Cross-check SDF
        # ------------------------------------------------------------------
        if sdf_status == "sdf_failed":
            if errs:
                failures.append((zinc_id, errs))
            continue

        if not sdf_exists:
            failures.append((zinc_id, errs))
            continue

        mol = read_sdf(sdf_path)
        if mol is None:
            errs.append("SDF could not be parsed by RDKit")
            failures.append((zinc_id, errs))
            continue

        # Check 5 — bond count matches bond.csv
        n_bonds_sdf = mol.GetNumBonds()
        n_bonds_csv = bond_counts.get(zinc_id)
        if n_bonds_csv is not None:
            check(n_bonds_sdf == n_bonds_csv,
                  f"Bond count mismatch: SDF={n_bonds_sdf}, bond.csv={n_bonds_csv}", errs)

        # Check 6 — Mulliken charges match xyz.csv, atom by atom
        try:
            sdf_charges = [float(q) for q in mol.GetProp("MullikenCharges").split()]
            for i, (q_out, q_ref) in enumerate(zip(sdf_charges, ref_charges)):
                if abs(q_out - q_ref) > CHARGE_TOL:
                    errs.append(
                        f"Mulliken charge mismatch at atom {i+1}: "
                        f"output={q_out:.6f}, xyz.csv={q_ref:.6f}"
                    )
                    break
        except KeyError:
            errs.append("MullikenCharges property missing from SDF")

        # Check 7 — total charge sum ≈ 0 (neutral molecule)
        try:
            q_sum = sum(float(q) for q in mol.GetProp("MullikenCharges").split())
            check(abs(q_sum) < QSUM_TOL,
                  f"Mulliken charge sum = {q_sum:.4f} (expected ≈ 0 for neutral mol)", errs)
        except KeyError:
            pass  # already flagged above

        # Check 8 — SMILES field in SDF matches mapping
        try:
            sdf_smi = mol.GetProp("SMILES")
            check(sdf_smi == can_smi,
                  f"SMILES mismatch: SDF='{sdf_smi}' | mapping='{can_smi}'", errs)
        except KeyError:
            errs.append("SMILES property missing from SDF")

        if errs:
            failures.append((zinc_id, errs))

    # ------------------------------------------------------------------
    # Check 9 — ID uniqueness
    # ------------------------------------------------------------------
    n_ids = map_df["ID"].nunique()
    id_ok = (n_ids == n_total)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_pass = n_total - len(failures)
    print(f"Results: {n_pass}/{n_total} molecules passed all checks")
    if n_sdf_failed:
        print(f"SDF skipped (sdf_failed in mapping): {n_sdf_failed} — XYZ-only checks applied")
    print(f"ID uniqueness: {'OK' if id_ok else f'FAIL — {n_total - n_ids} collision(s)'}\n")

    if failures:
        print(f"FAILURES ({len(failures)}):")
        for zinc_id, errs in failures:
            print(f"  {zinc_id}:")
            for e in errs:
                print(f"    - {e}")
        sys.exit(1)   # non-zero exit → make stops the pipeline
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
