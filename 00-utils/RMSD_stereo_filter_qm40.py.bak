#!/usr/bin/env python3
"""
Stage 5: Enantiomer filter for QM40 pipeline.

Groups molecules by flat SMILES (stereo markers stripped), then uses Kabsch
RMSD on heavy atoms + mirror reflection to identify true enantiomers.
Keeps one enantiomer per pair.

Adds 'stereo_status' column to qm40_mapping.csv:
  kept               — molecule passes, enters extxyz stage
  removed_enantiomer — true enantiomer of a kept molecule, skipped at extxyz
  not_processed      — reorder_status != success, not considered

Logs removed pairs to logs/stereo_removed.csv.
Does NOT delete xyz/sdf files (policy decision: valid structures, reversible).
"""

import re
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


RMSD_TOL        = 0.5   # min RMSD(i,j) to rule out same structure (Å)
RMSD_MIRROR_TOL = 0.10  # max RMSD(mirror_j → i) to confirm enantiomer (Å)


def kabsch_rmsd(P, Q):
    """Heavy-atom RMSD after Kabsch alignment of P onto Q (no return of rotation)."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    C = Pc.T @ Qc
    V, S, Wt = np.linalg.svd(C)
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1.0, 1.0, d]) @ Wt
    P_rot = Pc @ U + Q.mean(axis=0)
    return float(np.sqrt(((P_rot - Q) ** 2).mean()))


def strip_stereo(smiles):
    return re.sub(r'@{1,2}', '', smiles)


def read_xyz_coords(path):
    """Returns (symbols list, coords ndarray shape (nat,3))."""
    lines = Path(path).read_text().splitlines()
    nat = int(lines[0].strip())
    symbols, coords = [], []
    for line in lines[2:2 + nat]:
        parts = line.split()
        symbols.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return symbols, np.array(coords)


def heavy_coords(symbols, coords):
    idx = [i for i, s in enumerate(symbols) if s != 'H']
    return coords[idx]


def main():
    parser = argparse.ArgumentParser(description="QM40 enantiomer filter")
    parser.add_argument('--full-data', action='store_true',
                        help="Use cluster paths instead of sample paths")
    args = parser.parse_args()

    if args.full_data:
        BASE        = Path("/datos_pool/mldata1/QMdatasets/QM40/AIM4ML")
        XYZ_DIR     = BASE / "mol_files"
        MAPPING_CSV = BASE / "qm40_mapping.csv"
        LOG_DIR     = BASE / "logs"
    else:
        BASE        = Path("../../samples/qm40")
        XYZ_DIR     = BASE / "mol_files"
        MAPPING_CSV = BASE / "qm40_mapping.csv"
        LOG_DIR     = BASE / "logs"

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(MAPPING_CSV)

    active = mapping[mapping['reorder_status'] == 'success'].copy()
    print(f"Active molecules (reorder success): {len(active)}")

    # Safeguard: all must be single molecules
    assert not any('.' in smi for smi in active['canonical_SMILES']), \
        "Unexpected complex/salt in QM40 SMILES"

    # Group by flat SMILES
    active['_flat'] = active['canonical_SMILES'].apply(strip_stereo)
    groups = active.groupby('_flat')['ID'].apply(list)

    n_groups_multi = sum(1 for g in groups if len(g) > 1)
    print(f"Stereo groups (>1 member): {n_groups_multi}")

    stereo_status = {}
    for mol_id in active['ID']:
        stereo_status[mol_id] = 'kept'

    removed_pairs = []

    multi_groups = [(k, v) for k, v in groups.items() if len(v) > 1]
    for flat_smi, id_list in tqdm(multi_groups, desc="Stereo filter (RMSD)", unit=" groups"):


        # Load heavy-atom coords for all members in group
        loaded = []
        for mol_id in id_list:
            xyz_path = XYZ_DIR / f"mol_{mol_id}_1.xyz"
            if not xyz_path.exists():
                # reorder_status should have caught this, but be safe
                stereo_status[mol_id] = 'not_processed'
                loaded.append(None)
                continue
            symbols, coords = read_xyz_coords(xyz_path)
            loaded.append(heavy_coords(symbols, coords))

        n = len(id_list)
        keep = [True] * n

        for i in range(n):
            if not keep[i] or loaded[i] is None:
                continue
            Q = loaded[i]
            for j in range(i + 1, n):
                if not keep[j] or loaded[j] is None:
                    continue
                P = loaded[j]

                if len(P) != len(Q):
                    # Atom count mismatch — different connectivity, not enantiomers
                    continue

                rmsd_ij     = kabsch_rmsd(P, Q)
                P_mirror    = P * np.array([-1.0, 1.0, 1.0])
                rmsd_mirror = kabsch_rmsd(P_mirror, Q)

                if rmsd_ij >= RMSD_TOL and rmsd_mirror <= RMSD_MIRROR_TOL:
                    keep[j] = False
                    stereo_status[id_list[j]] = 'removed_enantiomer'
                    removed_pairs.append({
                        'kept_id':     id_list[i],
                        'removed_id':  id_list[j],
                        'flat_smiles': flat_smi,
                        'rmsd_ij':     round(rmsd_ij, 4),
                        'rmsd_mirror': round(rmsd_mirror, 4),
                    })

    # Molecules not processed (reorder failed) get not_processed
    failed_ids = set(mapping.loc[mapping['reorder_status'] != 'success', 'ID'])
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
