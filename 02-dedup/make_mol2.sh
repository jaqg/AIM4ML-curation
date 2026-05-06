#!/usr/bin/env bash
# make_mol2.sh — Convert SDF files in mol_files/ to MOL2 via OpenBabel.
#
# Converts every mol_*.sdf found in the output directory. Molecules that have
# no SDF (sdf_status=sdf_failed in qm40_mapping.csv) are naturally skipped
# because no .sdf file exists for them.
#
# Charges: Gasteiger (OpenBabel default). Mulliken charges remain in the SDF
# data field and are NOT transferred to the MOL2 charge column.
#
# Usage:
#   bash make_mol2.sh               # local sample (default)
#   bash make_mol2.sh --full-data   # full dataset on cluster

set -euo pipefail

FULL_DATA=0
for arg in "$@"; do
    [[ "$arg" == "--full-data" ]] && FULL_DATA=1
done

if [[ $FULL_DATA -eq 1 ]]; then
    MOL_DIR="/home/joseq/AIM4ML/QM40/mol_files"
    MODE="full"
else
    MOL_DIR="../../samples/qm40/mol_files"
    MODE="sample"
fi

echo "Mode: $MODE"
echo "Converting SDF → MOL2 in: $MOL_DIR"

if ! command -v obabel &>/dev/null; then
    echo "ERROR: obabel not found. Load the OpenBabel module first." >&2
    exit 1
fi

sdf_files=("$MOL_DIR"/mol_*.sdf)
total=${#sdf_files[@]}

if [[ $total -eq 0 || ! -f "${sdf_files[0]}" ]]; then
    echo "No SDF files found in $MOL_DIR — run dedup_qm40.py first."
    exit 1
fi

echo "Found $total SDF files."

done=0
failed=0
LOG_EVERY=1000

for sdf in "${sdf_files[@]}"; do
    mol2="${sdf%.sdf}.mol2"
    if obabel "$sdf" -O "$mol2" 2>/dev/null; then
        (( done++ )) || true
    else
        (( failed++ )) || true
    fi
    if (( (done + failed) % LOG_EVERY == 0 )); then
        echo "  $((done + failed)) / $total ..."
    fi
done

echo ""
echo "Done."
echo "  Converted: $done"
echo "  Failed:    $failed"
echo "  Output:    $MOL_DIR/"
