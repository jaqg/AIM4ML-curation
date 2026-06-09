#!/usr/bin/env python3
"""
Stage 5: Enantiomer filter for QM40 pipeline (SMILES-based).

Groups molecules by flat SMILES (stereo markers stripped), then compares
canonical stereo SMILES after inverting all tetrahedral stereocenters of
molecule i against molecule j. Match → true enantiomers → keep i, remove j.

Method replaces Kabsch RMSD (archived in RMSD_stereo_filter_qm40.py.bak).
Reason: RMSD is conformation-dependent; independently optimised enantiomers
can land in different local minima, inflating mirror RMSD above any fixed
threshold. SMILES inversion is exact with no tunable parameters.

Works entirely from the canonical_SMILES column — no xyz files loaded.

Adds 'stereo_status' column to qm40_mapping.csv:
  kept               — molecule passes, enters extxyz stage
  removed_enantiomer — true enantiomer of a kept molecule, skipped at extxyz
  not_processed      — sdf_status == sdf_failed, not considered

Logs removed pairs to logs/stereo_removed.csv.
Does NOT delete xyz/sdf files (policy decision: valid structures, reversible).
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType


def flat_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def stereo_smiles(mol):
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def has_tetrahedral_stereo(mol):
    return any(
        atom.GetChiralTag() in (ChiralType.CHI_TETRAHEDRAL_CW, ChiralType.CHI_TETRAHEDRAL_CCW)
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
    return stereo_smiles(inverted_a) == stereo_smiles(mol_b)


def main():
    parser = argparse.ArgumentParser(description="QM40 enantiomer filter (SMILES-based)")
    parser.add_argument('--full-data', action='store_true',
                        help="Run on full dataset.")
    parser.add_argument('--sample', action='store_true',
                        help="Run on sample dataset (cluster absolute paths, default).")
    args = parser.parse_args()

    BASE = Path("/datos_pool/mldata1/QMdatasets/QM40/AIM4ML") if args.full_data \
        else Path("/datos_pool/mldata1/QMdatasets/QM40/AIM4ML/samples")
    MAPPING_CSV = BASE / "qm40_mapping.csv"
    LOG_DIR     = BASE / "logs"

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(MAPPING_CSV)

    active = mapping[mapping['sdf_status'] != 'sdf_failed'].copy()
    print(f"Active molecules (sdf ok): {len(active)}")

    assert not any('.' in smi for smi in active['canonical_SMILES']), \
        "Unexpected complex/salt in QM40 SMILES"

    active['_mol'] = active['canonical_SMILES'].apply(Chem.MolFromSmiles)
    n_failed = active['_mol'].isna().sum()
    if n_failed:
        print(f"WARNING: {n_failed} SMILES failed to parse — will be skipped", file=sys.stderr)

    active['_flat'] = active['canonical_SMILES'].apply(flat_smiles)
    groups = active.groupby('_flat')['ID'].apply(list)

    n_groups_multi = sum(1 for g in groups if len(g) > 1)
    print(f"Stereo groups (>1 member): {n_groups_multi}")

    stereo_status = {mol_id: 'kept' for mol_id in active['ID']}
    removed_pairs = []

    id_to_mol = dict(zip(active['ID'], active['_mol']))
    id_to_smi = dict(zip(active['ID'], active['canonical_SMILES']))

    multi_groups = [(k, v) for k, v in groups.items() if len(v) > 1]
    for flat_smi, id_list in tqdm(multi_groups, desc="Stereo filter (SMILES)", unit=" groups"):
        n = len(id_list)
        keep = [True] * n

        for i in range(n):
            if not keep[i] or id_to_mol[id_list[i]] is None:
                continue
            mol_i = id_to_mol[id_list[i]]
            for j in range(i + 1, n):
                if not keep[j] or id_to_mol[id_list[j]] is None:
                    continue
                mol_j = id_to_mol[id_list[j]]

                if are_enantiomers(mol_i, mol_j):
                    keep[j] = False
                    stereo_status[id_list[j]] = 'removed_enantiomer'
                    removed_pairs.append({
                        'kept_id':               id_list[i],
                        'removed_id':            id_list[j],
                        'flat_smiles':           flat_smi,
                        'kept_stereo_smiles':    id_to_smi[id_list[i]],
                        'removed_stereo_smiles': id_to_smi[id_list[j]],
                    })

    failed_ids = set(mapping.loc[mapping['sdf_status'] == 'sdf_failed', 'ID'])
    for mol_id in failed_ids:
        stereo_status[mol_id] = 'not_processed'

    mapping['stereo_status'] = mapping['ID'].map(stereo_status)
    mapping.to_csv(MAPPING_CSV, index=False)

    if removed_pairs:
        pd.DataFrame(removed_pairs).to_csv(LOG_DIR / 'stereo_removed.csv', index=False)

    n_kept    = sum(1 for v in stereo_status.values() if v == 'kept')
    n_removed = sum(1 for v in stereo_status.values() if v == 'removed_enantiomer')
    n_skip    = sum(1 for v in stereo_status.values() if v == 'not_processed')

    print(f"\nStereo filter complete:")
    print(f"  Kept               : {n_kept}")
    print(f"  Removed enantiomers: {n_removed}")
    print(f"  Not processed      : {n_skip}")
    print(f"  Mapping  → {MAPPING_CSV}")
    if removed_pairs:
        print(f"  Log      → {LOG_DIR / 'stereo_removed.csv'}")


if __name__ == '__main__':
    main()
