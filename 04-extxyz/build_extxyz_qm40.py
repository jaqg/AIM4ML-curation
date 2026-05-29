#!/usr/bin/env python3
"""
Stage 6 (D18): Build extxyz trajectory files for QM40.

Reads reordered mol_*.xyz + supporting CSVs; writes batched extxyz files
(BATCH_SIZE structures each) with comma-separated key,value metadata in
the title line.

Title line format:
  ID,{Zinc_id},Formula,{formula},nat,{nat},CNSO,{cnso},chrg,{chrg},mult,1,
  e,{e:.5f},fmax,,Family,QM40,idsml,{md5},smiles,{smiles},
  tpsa,{tpsa:.2f},logp,{logp:.2f},nrot,{nrot},nfrag,1,iconf,0,

Key notes:
  - ID    : original Zinc_id (source provenance)
  - idsml : full MD5 hex of canonical SMILES (D19/D20; cross-dataset lookup key)
  - CNSO  : total count of C+N+S+O atoms
  - mult  : hardcoded 1 (QM40 = neutral singlets, enforced by filter_qm40.py)
  - fmax  : empty (pre-optimised geometry, no forces stored)
  - iconf : 0-indexed; always last key

Usage:
    python3 build_extxyz_qm40.py               # local sample
    python3 build_extxyz_qm40.py --full-data   # full dataset on cluster
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

BATCH_SIZE = 5000
CNSO_ELEMENTS = {"C", "N", "S", "O"}

PATHS = {
    "sample": {
        "mapping_csv":       "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/qm40_mapping.csv",
        "filtered_main_csv": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/filtered_sample_main.csv",
        "stats_csv":         "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/qm40_stats.csv",
        "mol_dir":           "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/mol_files",
        "out_dir":           "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples/extxyz",
    },
    "full": {
        "mapping_csv":       "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/qm40_mapping.csv",
        "filtered_main_csv": "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/filtered_main.csv",
        "stats_csv":         "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/stats/qm40_stats.csv",
        "mol_dir":           "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/mol_files",
        "out_dir":           "/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/extxyz",
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Build extxyz batches for QM40 (D18).")
    p.add_argument("--full-data", action="store_true")
    p.add_argument("--sample", action="store_true",
                            help="Run on sample dataset (cluster absolute paths).")
    return p.parse_args()


def hill_formula(atom_counts):
    result = ""
    for elem in ["C", "H"] + sorted(e for e in atom_counts if e not in ("C", "H")):
        n = atom_counts.get(elem, 0)
        if n:
            result += elem + (str(n) if n > 1 else "")
    return result


def parse_xyz(path):
    lines = Path(path).read_text().splitlines()
    nat = int(lines[0].strip())
    atoms = []
    for line in lines[2 : 2 + nat]:
        parts = line.split()
        atoms.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return nat, atoms

def build_title_line(zinc_id, formula, nat, cnso, chrg, e, idsml, smiles, tpsa, logp, nrot):
    return (
        f"ID,{zinc_id},"
        f"Formula,{formula},"
        f"nat,{nat},"
        f"CNSO,{cnso},"
        f"chrg,{chrg},"
        f"mult,1,"
        f"e,{e:.5f},"
        f"fmax,,"
        f"Family,QM40,"
        f"idsml,{idsml},"
        f"smiles,{smiles},"
        f"tpsa,{tpsa:.2f},"
        f"logp,{logp:.2f},"
        f"nrot,{nrot},"
        f"nfrag,1,"
        f"iconf,0,"
    )


def write_batch(out_dir, batch_idx, blocks):
    path = out_dir / f"qm40_single_batch{batch_idx:04d}.xyz"
    with open(path, "w") as f:
        for block in blocks:
            f.write("\n".join(block) + "\n")


def main():
    args = parse_args()
    p = PATHS["full" if args.full_data else "sample"]

    mol_dir  = Path(p["mol_dir"])
    out_dir  = Path(p["out_dir"])
    logs_dir = out_dir.parent / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(p["mapping_csv"])
    for col in ("reorder_status", "stereo_status"):
        if col not in mapping.columns:
            sys.exit(f"ERROR: '{col}' column missing from mapping CSV — run reorder and stereo_filter first.")

    main_df = pd.read_csv(p["filtered_main_csv"])
    stats_df = pd.read_csv(p["stats_csv"])

    energy = main_df.set_index("Zinc_id")["Internal_E(0K)"].to_dict()
    stats = stats_df.set_index("ID")[["TPSA", "charge", "logP", "nrot"]].to_dict("index")

    kept = mapping[
        (mapping["reorder_status"] == "success") &
        (mapping["stereo_status"]  == "kept") &
        (mapping["sdf_status"]     != "sdf_failed")
    ].copy()

    assert not any("." in s for s in kept["canonical_SMILES"]), \
        "Unexpected complex/salt in QM40 SMILES — check filtering pipeline."

    n_written = n_skipped = 0
    batch_idx = 1
    batch_buf = []
    skip_log = []

    for _, row in tqdm(kept.iterrows(), total=len(kept), desc="Building extxyz"):
        mol_id  = row["ID"]
        zinc_id = row["Zinc_id"]
        smiles  = row["canonical_SMILES"]

        xyz_path = mol_dir / f"mol_{mol_id}_1.xyz"

        reason = None
        if not xyz_path.exists():
            reason = "xyz_missing"
        elif zinc_id not in energy:
            reason = "energy_missing"
        elif mol_id not in stats:
            reason = "stats_missing"

        if reason:
            skip_log.append((zinc_id, mol_id, reason))
            n_skipped += 1
            continue

        nat, atoms = parse_xyz(xyz_path)
        atom_counts = Counter(a[0] for a in atoms)
        formula = hill_formula(atom_counts)
        cnso = sum(atom_counts.get(e, 0) for e in CNSO_ELEMENTS)
        idsml = row["ID"]

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skip_log.append((zinc_id, mol_id, "smiles_parse_failed"))
            n_skipped += 1
            continue

        tpsa = stats[mol_id]["TPSA"]
        logp = stats[mol_id]["logP"]
        nrot = int(stats[mol_id]["nrot"])
        chrg = int(stats[mol_id]["charge"])
        e    = energy[zinc_id]

        title = build_title_line(zinc_id, formula, nat, cnso, chrg, e, idsml, smiles, tpsa, logp, nrot)
        coord_lines = [f"{elem}  {x:.6f}  {y:.6f}  {z:.6f}" for elem, x, y, z in atoms]
        batch_buf.append([str(nat), title] + coord_lines)
        n_written += 1

        if len(batch_buf) == BATCH_SIZE:
            write_batch(out_dir, batch_idx, batch_buf)
            batch_idx += 1
            batch_buf = []

    if batch_buf:
        write_batch(out_dir, batch_idx, batch_buf)

    if skip_log:
        skip_path = logs_dir / "extxyz_skipped.csv"
        pd.DataFrame(skip_log, columns=["Zinc_id", "ID", "reason"]).to_csv(skip_path, index=False)

    n_batches = batch_idx - 1 if not batch_buf else batch_idx
    print(f"Written : {n_written}")
    print(f"Skipped : {n_skipped}" + (f" → {skip_path}" if skip_log else ""))
    print(f"Batches : {n_batches}")


if __name__ == "__main__":
    main()
