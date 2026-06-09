#!/usr/bin/env python3
"""
dedup_qm40.py — Assign compound IDs and rename XYZ files for QM40.

Steps:
    1. Verify no conformers (each Zinc_id appears exactly once in main.csv)
        - If there are no conformers, assign ICONF=1 to all molecules
        - If conformers are found, stop (not yet implemented)
    2. Canonicalise SMILES with RDKit
    3. Compute MD5 hash of canonical SMILES → compound ID
    4. Copy/rename {Zinc_id}.xyz → mol_{ID}_{ICONF}.xyz
    5. Build SDF from 3D geometry (rdDetermineBonds) + Mulliken charges
    6. Write a mapping CSV: Zinc_id, canonical_SMILES, ID, ICONF, NAT

Usage:
    python3 dedup_qm40.py               # local sample (default)
    python3 dedup_qm40.py --full-data   # full dataset on cluster
"""

import os
import sys
import warnings
import argparse
import hashlib
import shutil
import pandas as pd
from tqdm import tqdm
from rdkit import Chem, RDLogger
from rdkit.Chem import SDWriter, rdDetermineBonds, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Geometry import rdGeometry

warnings.filterwarnings('ignore')         # suppress Python-level warnings (e.g. RDKit valence warnings)
RDLogger.DisableLog('rdApp.*')            # suppress RDKit C++ logger messages
_rdlog = RDLogger.logger()                # kept for the CRITICAL/WARNING toggle around AssignBondOrdersFromTemplate

# Normalizes charge-separated representations before SMILES comparison.
# Key transform applied: [S+][O-] → S=O  (and analogues for N, P, etc.)
# This avoids false-positive topology warnings when DetermineBondOrders
# perceives a sulfoxide S-O bond as a coordinate/dative bond ([S+][O-])
# rather than the conventional double-bond form (S=O).
_NORMALIZER = rdMolStandardize.Normalizer()

HASH_LENGTH = 32   # full MD5 hex (D19/D20: 32-char for dataset scale safety)

# --- Paths -------------------------------------------------------------------
PATHS = {
    "sample": {
        "main_csv":    "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/filtered_sample_main.csv",
        "xyz_csv":     "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/sample_xyz.csv",
        "bond_csv":    "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/sample_bond.csv",
        "xyz_dir":     "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/xyz_files",
        "out_dir":     "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/mol_files",
        "mapping_csv": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/qm40_mapping.csv",
        "file_mode":   "copy",
    },
    "full": {
        "main_csv":    "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/filtered_main.csv",
        "xyz_csv":     "/datos_pool/mldata1/QMdatasets/QM40/xyz.csv",
        "bond_csv":    "/datos_pool/mldata1/QMdatasets/QM40/bond.csv",
        "xyz_dir":     "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/xyz_files",
        "out_dir":     "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/mol_files",
        "mapping_csv": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/qm40_mapping.csv",
        "file_mode":   "copy",
    },
}
# -----------------------------------------------------------------------------


def canonical_smiles(smiles):
    """Return RDKit canonical SMILES, or None if the molecule cannot be parsed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def compound_id(canonical_smi):
    """Return the first HASH_LENGTH characters of the MD5 hash of the canonical SMILES."""
    return hashlib.md5(canonical_smi.encode()).hexdigest()[:HASH_LENGTH]


def parse_args():
    parser = argparse.ArgumentParser(description="Assign compound IDs and rename XYZ files for QM40.")
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Run on the full QM40 dataset on the cluster (default: local sample).",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Run on sample dataset (cluster absolute paths, default).",
    )
    return parser.parse_args()


def transfer_file(src, dst, file_mode):
    """Copy or move a file depending on file_mode."""
    if file_mode == "move":
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)


def smiles_topology_ok(mol, can_smi):
    """
    Return True if the perceived bond topology of `mol` matches `can_smi`.

    Three-level comparison to suppress known false positives:
      1. Normalize charge-separated forms ([S+][O-] → S=O, etc.) then compare
         with stereo. Equal → clean match.
      2. Compare without stereo (isomericSmiles=False). Equal → the graph is
         correct; 3D geometry added stereocenters absent in the flat SMILES.
      3. Still not equal → genuine bond-order / connectivity mismatch. Return False.

    Returns False only for real topology errors worth logging.
    """
    try:
        mol_norm = _NORMALIZER.normalize(Chem.RemoveAllHs(mol))
        ref_norm = _NORMALIZER.normalize(Chem.MolFromSmiles(can_smi))
        if Chem.MolToSmiles(mol_norm) == Chem.MolToSmiles(ref_norm):
            return True   # exact match after normalisation (handles [S+][O-] vs S=O)
        perceived_ns = Chem.MolToSmiles(mol_norm, isomericSmiles=False)
        ref_ns       = Chem.MolToSmiles(ref_norm,  isomericSmiles=False)
        return perceived_ns == ref_ns   # True → stereo-only difference, not a real error
    except Exception:
        return False   # valence violation or other RDKit error — mol is not topology-OK


def build_mol_from_xyz(atoms, coords, charges, bond_pairs=None, total_charge=0):
    """
    Build an RDKit Mol from atom elements, 3D coordinates, and Mulliken charges.

    total_charge=0 default: QM40 contains only neutral singlet molecules.

    Hybrid strategy:
    - If bond_pairs is provided (from bond.csv), connectivity is read directly from
      the source data. Only bond orders are then inferred by DetermineBondOrders.
    - If bond_pairs is None (fallback), both connectivity and bond orders are inferred
      by DetermineBonds from 3D geometry alone.

    In both cases the XYZ atom ordering is preserved (consistent with bond.csv indices).
    Returns None if bond perception fails.
    """
    try:
        # Create an empty editable molecule (RWMol = read/write mol).
        # Atoms are added in XYZ order, which is critical: this ordering must
        # match bond.csv (atom1/atom2 indices are 1-based positions in xyz.csv).
        mol = Chem.RWMol()
        for element in atoms:
            mol.AddAtom(Chem.Atom(element))

        # A Conformer is RDKit's container for 3D coordinates (one (x,y,z) per atom).
        # We fill it with the DFT-optimised coordinates from xyz.csv (final_x/y/z).
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, rdGeometry.Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)

        if bond_pairs is not None:
            # Hybrid approach: connectivity from bond.csv, bond orders from geometry.
            #
            # bond.csv atom1/atom2 are 1-based indices into the XYZ atom ordering.
            # We add them as single bonds; DetermineBondOrders then upgrades them
            # (single → double/aromatic) using 3D geometry + Hückel aromaticity.
            # This is strictly more faithful than DetermineBonds: the graph topology
            # is guaranteed by the QM40 source data, not inferred from distances.
            for a1, a2 in bond_pairs:
                mol.AddBond(a1, a2, Chem.BondType.SINGLE)
            try:
                # Clone before Hückel: if it fails mid-run, it may have partially
                # upgraded bonds. The fallback must start from a clean SINGLE-bond mol,
                # not a partially-modified one that could produce silently wrong orders.
                mol_attempt = Chem.RWMol(mol)
                rdDetermineBonds.DetermineBondOrders(mol_attempt, charge=total_charge, useHueckel=True)
                mol = mol_attempt   # Hückel succeeded — use the result
            except Exception:
                rdDetermineBonds.DetermineBondOrders(mol, charge=total_charge)
        else:
            # Fallback: infer both connectivity and bond orders from 3D geometry.
            # Used only if the molecule is absent from bond.csv.
            #
            # Two-attempt strategy:
            # 1. useHueckel=True: uses the Hückel aromaticity model for bond order
            #    assignment — handles aromatic/fused rings correctly, avoids valence errors.
            # 2. Fallback to standard algorithm (useHueckel=False) if Hückel fails.
            # Clone before Hückel for the same reason as above (clean fallback).
            try:
                mol_attempt = Chem.RWMol(mol)
                rdDetermineBonds.DetermineBonds(mol_attempt, charge=total_charge, useHueckel=True)
                mol = mol_attempt
            except Exception:
                rdDetermineBonds.DetermineBonds(mol, charge=total_charge)

        # Attach the Mulliken charge to each atom as a named float property.
        # GetAtomWithIdx(i) retrieves the atom object at position i (0-based).
        # SetDoubleProp("MullikenCharge", q) works like a per-atom dictionary entry:
        #   atom.properties["MullikenCharge"] = q
        # Retrieved later with GetDoubleProp("MullikenCharge").
        for i, q in enumerate(charges):
            mol.GetAtomWithIdx(i).SetDoubleProp("MullikenCharge", q)

        # Convert the editable RWMol back to a read-only Mol for writing.
        return mol.GetMol()

    except Exception:
        return None


def build_mol_from_smiles_template(atoms, coords, charges, bond_pairs, can_smi):
    """
    Template-based SDF generation: assign bond orders from canonical SMILES.

    Why this exists
    ---------------
    DetermineBondOrders systematically mis-assigns bond orders for 5-membered
    aromatic sulfur heterocycles (thiophene, thiazole, thiadiazole, isothiazole).
    Instead of perceiving aromaticity, it produces a charge-separated Kekulé form
    with [S+] on the ring sulfur and [O-] on the nearest exocyclic oxygen. Analysis
    of the full QM40 run (162 954 molecules) showed 1 507 topology warnings and
    2 295 complete SDF failures, 87.7% of warnings from this single cause.

    The fix: use AllChem.AssignBondOrdersFromTemplate, which takes the canonical
    SMILES (known-correct aromaticity and bond orders from main.csv) as a reference
    and maps its bond orders onto the heavy-atom mol via substructure matching.
    This bypasses 3D-geometry bond inference entirely for these problem classes.

    Strategy
    --------
    1. Parse can_smi → template mol (correct bond orders, implicit H).
    2. Build heavy-atom RWMol from xyz with bond.csv connectivity (all SINGLE bonds).
    3. AssignBondOrdersFromTemplate: match template to heavy mol → fix bond orders.
    4. Add H atoms back with xyz positions and single bonds from bond.csv.
    5. Set Mulliken charges on all atoms.

    Returns None if bond_pairs is absent (no connectivity to match against)
    or if any step raises an exception.
    """
    if bond_pairs is None:
        return None   # cannot use template without explicit connectivity

    try:
        template = Chem.MolFromSmiles(can_smi)
        if template is None:
            return None

        # --- Separate H and heavy atoms; preserve original xyz index ordering ---
        heavy_orig = [i for i, el in enumerate(atoms) if el != 'H']
        h_orig     = [i for i, el in enumerate(atoms) if el == 'H']
        orig_to_heavy = {orig: new for new, orig in enumerate(heavy_orig)}

        # --- Build heavy-atom RWMol with xyz connectivity (SINGLE bonds) ---
        mol_h = Chem.RWMol()
        for i in heavy_orig:
            mol_h.AddAtom(Chem.Atom(atoms[i]))

        conf = Chem.Conformer(len(heavy_orig))
        for new_i, orig_i in enumerate(heavy_orig):
            x, y, z = coords[orig_i]
            conf.SetAtomPosition(new_i, rdGeometry.Point3D(x, y, z))
        mol_h.AddConformer(conf, assignId=True)

        for a1, a2 in bond_pairs:
            if a1 in orig_to_heavy and a2 in orig_to_heavy:
                mol_h.AddBond(orig_to_heavy[a1], orig_to_heavy[a2], Chem.BondType.SINGLE)

        # --- Assign bond orders from SMILES template via substructure matching ---
        # AssignBondOrdersFromTemplate picks one match arbitrarily when the molecule
        # is symmetric, emitting a "More than one matching pattern found" warning to
        # stderr. The chosen mapping can produce valence violations (e.g. N valence 5).
        # Suppress RDKit stderr around this call; SanitizeMol catches bad results.
        _rdlog.setLevel(RDLogger.CRITICAL)
        try:
            mol_assigned = AllChem.AssignBondOrdersFromTemplate(template, mol_h.GetMol())
        finally:
            _rdlog.setLevel(RDLogger.WARNING)
        Chem.SanitizeMol(mol_assigned)   # raises AtomValenceException if match was wrong

        # --- Rebuild full mol: add H atoms back with xyz positions ---
        mol_full = Chem.RWMol(mol_assigned)
        orig_to_full = {orig: new for new, orig in enumerate(heavy_orig)}
        for orig_h in h_orig:
            new_idx = mol_full.AddAtom(Chem.Atom('H'))
            orig_to_full[orig_h] = new_idx

        n_full = len(heavy_orig) + len(h_orig)
        conf_full = Chem.Conformer(n_full)
        for new_i in range(len(heavy_orig)):
            conf_full.SetAtomPosition(new_i, mol_assigned.GetConformer().GetAtomPosition(new_i))
        for orig_h in h_orig:
            x, y, z = coords[orig_h]
            conf_full.SetAtomPosition(orig_to_full[orig_h], rdGeometry.Point3D(x, y, z))
        mol_full.RemoveAllConformers()
        mol_full.AddConformer(conf_full, assignId=True)

        # Add H–heavy bonds from bond.csv (always SINGLE)
        h_set = set(h_orig)
        for a1, a2 in bond_pairs:
            if a1 in h_set or a2 in h_set:
                new_a1 = orig_to_full.get(a1)
                new_a2 = orig_to_full.get(a2)
                if new_a1 is not None and new_a2 is not None:
                    mol_full.AddBond(new_a1, new_a2, Chem.BondType.SINGLE)

        # --- Set Mulliken charges on all atoms ---
        for orig_i, new_i in orig_to_full.items():
            mol_full.GetAtomWithIdx(new_i).SetDoubleProp("MullikenCharge", charges[orig_i])

        return mol_full.GetMol()

    except Exception:
        return None


def write_sdf(path, mol, zinc_id, can_smi):
    """Write a single molecule to an SDF file with metadata fields."""
    # _Name is a special RDKit property: it becomes line 1 of the SDF header.
    mol.SetProp("_Name", zinc_id)
    # SMILES becomes a "> <SMILES>" data field in the SDF file.
    mol.SetProp("SMILES", can_smi)

    # SDF V2000 has no standard way to store per-atom floating-point values
    # in the atom block, so Mulliken charges are written as a single data field:
    # "> <MullikenCharges>" with space-separated values, one per atom.
    charges = [mol.GetAtomWithIdx(i).GetDoubleProp("MullikenCharge")
               for i in range(mol.GetNumAtoms())]
    mol.SetProp("MullikenCharges", " ".join(f"{q:.6f}" for q in charges))

    # SDWriter writes the full SDF record: atom block, bond block, data fields, $$$$
    writer = SDWriter(path)
    writer.write(mol)
    writer.close()


def main():
    args  = parse_args()
    mode = "full" if args.full_data else "sample"
    paths = PATHS[mode]

    MAIN_CSV    = paths["main_csv"]
    XYZ_CSV     = paths["xyz_csv"]
    BOND_CSV    = paths["bond_csv"]
    XYZ_DIR     = paths["xyz_dir"]
    OUT_DIR     = paths["out_dir"]
    MAPPING_CSV = paths["mapping_csv"]
    FILE_MODE   = paths["file_mode"]

    print(f"Mode: {mode}")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Reading {MAIN_CSV} ...")
    main_df = pd.read_csv(MAIN_CSV)

    if "energy_status" in main_df.columns:
        n_before = len(main_df)
        main_df = main_df[main_df["energy_status"] == "ok"].copy()
        print(f"  Energy filter: {n_before - len(main_df)} flagged removed ({len(main_df)} remain)")

    # -------------------------------------------------------------------------
    # Step 1 — conformer check
    # -------------------------------------------------------------------------

    # First check if there are conformers in QM40 by checking if there are
    # multiple rows for the same Zinc_id (duplicates)
    # If there are no duplicates, assign ICONF=1 to all rows

    # count total number of lines
    n_rows    = len(main_df)
    # count number of unique Zinc_ids 
    n_unique  = main_df["Zinc_id"].nunique()
    # check if there is any Zinc_id duplicated (boolean)
    has_dupes = main_df["Zinc_id"].duplicated().any()

    print(f"\nConformer check:")
    print(f"  Total rows:       {n_rows}")
    print(f"  Unique Zinc_ids:  {n_unique}")
    print(f"  Duplicates found: {has_dupes}")

    if has_dupes:
        # Get the list of Zinc_ids that appear more than once
        # If duplicates are found, warns the user and stops
        dupes = main_df[main_df["Zinc_id"].duplicated(keep=False)]["Zinc_id"].unique()
        print(f"  WARNING: {len(dupes)} Zinc_ids appear more than once.")
        print(f"  Conformer handling is not yet implemented — stopping.")
        return
    else:
        print(f"  OK — no conformers. Assigning ICONF=1 to all molecules.")
        iconf = 1

    # -------------------------------------------------------------------------
    # Load xyz.csv and bond.csv; group by Zinc_id for fast lookup during SDF generation
    # -------------------------------------------------------------------------
    print(f"\nReading {XYZ_CSV} ...")
    xyz_df = pd.read_csv(XYZ_CSV)
    # Build a dict {Zinc_id → sub-DataFrame} for O(1) lookup inside the loop.
    # Avoids searching the full xyz_df on every iteration (162K molecules × 162K search = slow).
    xyz_groups = {zid: grp for zid, grp in xyz_df.groupby("Zinc_id", sort=False)}

    print(f"Reading {BOND_CSV} ...")
    bond_df = pd.read_csv(BOND_CSV)
    # Build a dict {Zinc_id → list of (a1, a2) 0-based index pairs} for O(1) lookup.
    # bond.csv atom1/atom2 are 1-based → subtract 1 to get 0-based RDKit indices.
    bond_groups = {
        zid: list(zip(grp["atom1"].astype(int) - 1, grp["atom2"].astype(int) - 1))
        for zid, grp in bond_df.groupby("Zinc_id", sort=False)
    }

    # -------------------------------------------------------------------------
    # Steps 2–5 — canonicalise, hash, rename, write SDF
    # -------------------------------------------------------------------------
    LOG_DIR = os.path.join(os.path.dirname(MAPPING_CSV), "logs")
    os.makedirs(LOG_DIR, exist_ok=True)

    records           = []
    skipped_smiles    = []   # RDKit could not parse SMILES
    skipped_xyz       = []   # XYZ file not found in XYZ_DIR
    skipped_sdf       = []   # SDF generation failed — both DetermineBondOrders and template failed
    topology_warnings = []   # perceived SMILES != canonical SMILES — both approaches failed
    topology_fixed    = []   # DetermineBondOrders gave wrong topology, fixed by template
    sdf_recovered     = []   # DetermineBondOrders failed completely, recovered by template

    print(f"\nProcessing {n_rows} molecules ...")
    for _, row in tqdm(main_df.iterrows(), total=n_rows, desc="Processing", unit="mol", file=sys.stdout):
        zinc_id = row["Zinc_id"]
        smiles  = row["smile"]

        # Step 2 — canonicalise
        can_smi = canonical_smiles(smiles)
        if can_smi is None:
            skipped_smiles.append(zinc_id)
            continue

        # Step 3 — compute compound ID
        cid   = compound_id(can_smi)

        # Step 4 — transfer XYZ file
        src = os.path.join(XYZ_DIR, f"{zinc_id}.xyz")
        dst = os.path.join(OUT_DIR, f"mol_{cid}_{iconf}.xyz")

        if not os.path.exists(src):
            skipped_xyz.append(zinc_id)
            continue

        # Read NAT from line 1 before transferring (src may be moved away)
        with open(src) as f:
            nat = int(f.readline().strip())

        transfer_file(src, dst, FILE_MODE)

        # Step 5 — build and write SDF
        sdf_status = "sdf_failed"   # updated below as each path succeeds
        if zinc_id in xyz_groups:
            grp     = xyz_groups[zinc_id]
            atoms   = grp["atom"].tolist()
            coords  = list(zip(
                grp["final_x"].astype(float),
                grp["final_y"].astype(float),
                grp["final_z"].astype(float),
            ))
            charges = grp["charge"].astype(float).tolist()

            bond_pairs = bond_groups.get(zinc_id, None)
            mol = build_mol_from_xyz(atoms, coords, charges, bond_pairs=bond_pairs)

            if mol is not None:
                perceived_smi = Chem.MolToSmiles(Chem.RemoveAllHs(mol))
                if not smiles_topology_ok(mol, can_smi):
                    # DetermineBondOrders produced wrong topology — typically aromatic
                    # sulfur heterocycles (thiophene, thiazole, thiadiazole) where
                    # the algorithm generates a charge-separated [S+]...[O-] form
                    # instead of the correct aromatic representation.
                    # Fall back to SMILES-template bond order assignment.
                    mol_fixed = build_mol_from_smiles_template(
                        atoms, coords, charges, bond_pairs, can_smi
                    )
                    if mol_fixed is not None and smiles_topology_ok(mol_fixed, can_smi):
                        mol = mol_fixed
                        sdf_status = "topology_fixed"
                        topology_fixed.append(zinc_id)
                    else:
                        sdf_status = "topology_warning"
                        topology_warnings.append((zinc_id, can_smi, perceived_smi))
                else:
                    sdf_status = "ok"
            else:
                # DetermineBondOrders failed completely — try template approach.
                mol = build_mol_from_smiles_template(
                    atoms, coords, charges, bond_pairs, can_smi
                )
                if mol is not None:
                    sdf_status = "sdf_recovered"
                    sdf_recovered.append(zinc_id)
                else:
                    skipped_sdf.append(zinc_id)
                    # sdf_status remains "sdf_failed"

            if mol is not None:
                dst_sdf = os.path.join(OUT_DIR, f"mol_{cid}_{iconf}.sdf")
                try:
                    write_sdf(dst_sdf, mol, zinc_id, can_smi)
                    # Verify RDKit can read it back with full sanitization —
                    # SDWriter can silently write mols that fail on read (e.g. valence violations).
                    readback = next(iter(Chem.SDMolSupplier(dst_sdf, removeHs=False)), None)
                    if readback is None:
                        raise ValueError("SDF read-back failed sanitization")
                except Exception:
                    skipped_sdf.append(zinc_id)
                    sdf_status = "sdf_failed"
                    if os.path.exists(dst_sdf):
                        os.remove(dst_sdf)
        else:
            skipped_sdf.append(zinc_id)
            # sdf_status remains "sdf_failed"

        # Unconditional: the molecule is tracked even when SDF generation failed.
        # SDF failures are in skipped_sdf; the mapping will have more rows than SDFs.
        records.append({
            "Zinc_id":          zinc_id,
            "canonical_SMILES": can_smi,
            "ID":               cid,
            "ICONF":            iconf,
            "NAT":              nat,
            "sdf_status":       sdf_status,
        })

    # -------------------------------------------------------------------------
    # Step 6 — write mapping CSV
    # -------------------------------------------------------------------------
    mapping_df = pd.DataFrame(records, columns=["Zinc_id", "canonical_SMILES", "ID", "ICONF", "NAT", "sdf_status"])
    mapping_df.to_csv(MAPPING_CSV, index=False)

    print(f"\nDone.")
    print(f"  Processed:              {len(records)}")
    print(f"  Topology OK:            {len(records) - len(topology_warnings) - len(topology_fixed)}")
    print(f"  Topology fixed (tmpl):  {len(topology_fixed)}")
    print(f"  SDF recovered (tmpl):   {len(sdf_recovered)}")
    print(f"  Topology warnings:      {len(topology_warnings)}")
    print(f"  Skipped (SMILES):       {len(skipped_smiles)}")
    print(f"  Skipped (XYZ):          {len(skipped_xyz)}")
    print(f"  SDF failed:             {len(skipped_sdf)}")
    print(f"  Mapping:                {MAPPING_CSV}")
    print(f"  Files:                  {OUT_DIR}/")
    print(f"  Logs:                   {LOG_DIR}/")

    def _write_log(path, header, rows):
        with open(path, "w") as f:
            f.write(header + "\n")
            for row in rows:
                f.write(row + "\n")

    if topology_fixed:
        p = os.path.join(LOG_DIR, "topology_fixed.txt")
        _write_log(p, f"# Topology fixed by SMILES template ({len(topology_fixed)})", topology_fixed)
        print(f"\n  Topology fixed list → {p}")

    if sdf_recovered:
        p = os.path.join(LOG_DIR, "sdf_recovered.txt")
        _write_log(p, f"# SDF recovered by SMILES template ({len(sdf_recovered)})", sdf_recovered)
        print(f"  SDF recovered list  → {p}")

    if skipped_smiles:
        p = os.path.join(LOG_DIR, "skipped_smiles.txt")
        _write_log(p, f"# Skipped — RDKit could not parse SMILES ({len(skipped_smiles)})", skipped_smiles)
        print(f"\n  Skipped SMILES list → {p}")

    if skipped_xyz:
        p = os.path.join(LOG_DIR, "skipped_xyz.txt")
        _write_log(p, f"# Skipped — XYZ file not found ({len(skipped_xyz)})", skipped_xyz)
        print(f"  Skipped XYZ list   → {p}")

    if skipped_sdf:
        p = os.path.join(LOG_DIR, "skipped_sdf.txt")
        _write_log(p, f"# SDF failed — rdDetermineBonds error ({len(skipped_sdf)})", skipped_sdf)
        print(f"  SDF failed list    → {p}")

    if topology_warnings:
        p = os.path.join(LOG_DIR, "topology_warnings.txt")
        with open(p, "w") as f:
            f.write(f"# Topology mismatches — perceived bond orders differ from main.csv SMILES ({len(topology_warnings)})\n")
            f.write("# Zinc_id\tmain.csv_SMILES\tperceived_SMILES\n")
            for zinc_id, can_smi, perceived_smi in topology_warnings:
                f.write(f"{zinc_id}\t{can_smi}\t{perceived_smi}\n")
        print(f"  Topology warnings  → {p}")

    # -------------------------------------------------------------------------
    # Sanity checks
    # -------------------------------------------------------------------------
    if mapping_df.empty:
        print(f"\n  WARNING: no molecules processed — mapping CSV is empty.")
        print(f"  Check that parse_qm40.py has been run and XYZ_DIR exists: {XYZ_DIR}")
        return

    # Check if there are any ID collisions
    n_ids = mapping_df["ID"].nunique()
    if n_ids < len(mapping_df):
        n_collisions = len(mapping_df) - n_ids
        print(f"\n  WARNING: {n_collisions} ID collision(s) detected.")
        print(f"  Consider increasing HASH_LENGTH (currently {HASH_LENGTH}).")
    else:
        print(f"\n  ID collision check: OK — all {n_ids} IDs are unique.")


if __name__ == "__main__":
    main()
