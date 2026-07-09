# AIM4ML Curation Pipeline

Reproducible curation pipeline for quantum-chemistry molecular datasets. Converts raw SDF input through 11 validation, filtering, deduplication, and formatting stages into extended XYZ trajectory files ready for machine-learned interatomic potential (MLIP) training with MACE, NequIP, SchNetPack, and similar frameworks.

## Pipeline Architecture

```
input.sdf
  │
  ▼
[0] validate       ── contract check (required tags, types)
[1] split          ── SDF → Parquet batches
[2] energy_prefilter ── OLS atom-type outlier detection (MAD z‑score)
[3] filter         ── neutral / non‑zwitterion / closed‑shell
[4] dedup          ── canonical SMILES + CompoundID + conformer dedup
[5] validate       ── integrity cross‑checks
[6] stereo_filter  ── enantiomer removal
[7] reorder        ── canonical atom ordering (RDKit or AMBER backend)
[8] conformer_filter ── conformer RMSD pruning
[9] stats          ── descriptors + histograms + Tanimoto diversity
[10] extxyz        ── extended XYZ trajectory files (MLIP‑ready)
  │
  ▼
extxyz/*.xyz   +   stats/stats_summary.csv   +   stats/plots/
```

## Dependencies

- Python ≥ 3.10
- [RDKit](https://www.rdkit.org/) ≥ 2024
- NumPy, pandas, pyarrow (Parquet I/O)
- Matplotlib (stats plots)
- OpenBabel (optional, for legacy QM40 scripts)
- AMBER antechamber (optional, for AMBER reorder backend)

Install with conda:

```bash
conda create -n aim4ml python=3.11 rdkit numpy pandas pyarrow matplotlib -c conda-forge
conda activate aim4ml
```

## Quick Start

```bash
# Place your input SDF in samples/ (or symlink it)
cd scripts/

# Run full pipeline on sample data
make MODE=sample WORKERS=8

# Run full pipeline on complete dataset
make MODE=full WORKERS=40

# Individual stages
make stage0    # validate input
make stage3    # chemical filter only
make extxyz    # generate extXYZ files

# Run test suite
make test
```

## Data Contract

### Input SDF

Each molecule must carry these SDF property tags:

| Tag | Type | Status |
|-----|------|--------|
| `Energy_Ha` | float | required |
| `FormalCharge` | int | required |
| `Multiplicity` | int | required |
| `SMILES` | str | recommended |
| `SourceID` | str | recommended |
| `HOMO_Ha`, `LUMO_Ha`, `HL_Gap_Ha` | float | optional |
| `PartialCharges` | str | optional |

Molecules are stored as V2000 mol blocks with 3D coordinates and single bonds (bond orders resolved by dedup stage via `DetermineBondOrders` + SMILES template fallback).

### Output: Extended XYZ

```
natoms
SourceID=...,CompoundID=...,Formula=...,nat=...,CNSO=...,chrg=...,mult=...,e=...,smiles=...
C   0.123456  0.234567  0.345678
H   1.234567  0.345678  0.456789
...
```

Metadata keys: `SourceID`, `CompoundID`, `Formula`, `nat`, `CNSO`, `chrg`, `mult`, `e` (Energy_Ha), `smiles` (canonical).

## QM40-Specific Scripts

Three standalone scripts handle the QM40 dataset as a worked example:

- `convert_qm40.py` — converts raw QM40 CSVs (main, xyz, bond) to pipeline-standard SDF
- `filter_qm40.py` — Phase‑1 curation filter (neutral, non‑zwitterion, closed‑shell)
- `energy_prefilter_qm40.py` — atom‑type OLS energy outlier detection

## Data Preservation

The pipeline is fully reproducible from `input.sdf` alone. All intermediate Parquet batches can be regenerated. See [`DATA_PRESERVATION.md`](DATA_PRESERVATION.md) for details on what to archive and what can be safely deleted.

## Repository Structure

```
scripts/
├── 00_validate.py           # Stage 0: input contract check
├── 01_split.py              # Stage 1: SDF → Parquet
├── 02_energy_prefilter.py   # Stage 2: energy outlier detection
├── 03_filter.py             # Stage 3: chemical filter
├── 04_dedup.py               # Stage 4: canonicalize + dedup
├── 05_validate.py            # Stage 5: integrity checks
├── 06_stereo_filter.py       # Stage 6: enantiomer removal
├── 07_reorder.py             # Stage 7: atom reordering
├── 08_conformer_filter.py    # Stage 8: conformer pruning
├── 09_stats.py               # Stage 9: descriptors + plots
├── 10_extxyz.py              # Stage 10: extXYZ generation
├── convert_qm40.py           # QM40 CSV → SDF converter
├── filter_qm40.py            # QM40 curation filter
├── energy_prefilter_qm40.py  # QM40 energy outlier script
├── lib/                      # Shared library
│   ├── schema.py             #   tag definitions
│   ├── sdf_io.py             #   SDF read/write helpers
│   ├── parquet_io.py         #   Parquet batch I/O
│   └── antechamber_xyz_reord.sh  # AMBER backend helper
├── 00-utils/                 # Legacy helper scripts (reference)
├── Makefile                  # Full pipeline orchestration
├── DATA_PRESERVATION.md      # Archival policy
└── README.md                 # This file
```

## Citation

If you use this pipeline in your research, please cite the accompanying manuscript:

> Quiñonero, J. A. et al. *AIM4ML: Automated Curation Pipeline for Quantum Chemistry Datasets*. (in preparation)

## License

[Specify your license here]
