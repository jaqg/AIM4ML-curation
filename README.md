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

## Modular Usage

Each stage is a standalone script — run independently, swap backends, tune thresholds, or repurpose for different datasets without touching the Makefile.

### Input/Output conventions

All stages 2–10 use Parquet batches internally (zstd‑compressed). Stages 0–1 convert SDF ↔ Parquet. Each script accepts at minimum:

| Flag | Meaning |
|------|---------|
| `-i` / `--input-dir` | Directory of Parquet batches (default varies by stage) |
| `-o` / `--output-dir` | Directory for output Parquet batches |
| `--rejects-dir` | Directory for rejected‑molecule SDFs |
| `--force-keep-rejected` | Keep rejected rows in Parquet output (default: drop) |

### Stage-by-stage flags

#### Stage 0 — `00_validate.py`
```bash
python3 00_validate.py input.sdf [-o output.sdf] [--rejects-dir rejects/00_validate] [--lenient]
```
| Flag | Effect |
|------|--------|
| `input.sdf` | Positional: path to raw SDF |
| `-o` | Clean output SDF (default: `<input>_valid.sdf`) |
| `--lenient` | Warn instead of rejecting on missing required tags |

#### Stage 1 — `01_split.py`
```bash
python3 01_split.py input.sdf [-o batches/] [-b 5000]
```
| Flag | Effect |
|------|--------|
| `-b` / `--batch-size` | Molecules per Parquet batch (default: 5000) |

#### Stage 2 — `02_energy_prefilter.py`
```bash
python3 02_energy_prefilter.py -i batches/ -o filtered_batches/ \
    [--threshold 3.5] [--atom-types H C N O F S Cl Br] [--skip] [--force-keep-rejected]
```
| Flag | Effect |
|------|--------|
| `--threshold` | MAD z‑score cut‑off (default: 3.5). Higher = fewer flagged |
| `--atom-types` | Atom symbols to include in the OLS model (default: H,C,N,O,F,S,Cl,Br) |
| `--skip` | Pass‑through all molecules (no energy filtering) |

#### Stage 3 — `03_filter.py`
```bash
python3 03_filter.py -i filtered_batches/ -o curated_batches/ \
    [--preset neutral_closed_shell] [--allowed-elements C,H,N,O,F,S,Cl,Br] \
    [--min-heavy 4] [--max-heavy 200] [--force-keep-rejected]
```
| Flag | Effect |
|------|--------|
| `--preset` | Pre‑defined filter: `neutral_closed_shell`, `neutral`, or `none` |
| `--allowed-elements` | Comma‑separated allowed atomic symbols |
| `--min-heavy` | Minimum number of heavy (non‑H) atoms |
| `--max-heavy` | Maximum number of heavy (non‑H) atoms |

#### Stage 4 — `04_dedup.py`
```bash
python3 04_dedup.py -i curated_batches/ -o deduped_batches/ [--rejects-dir rejects/04_dedup]
```
Adds `CanonicalSMILES`, `CompoundID` (MD5), `Formula`, and `conformer_duplicate` flag. Duplicate conformers (same CompoundID, same energy) are tagged but kept — they're pruned later by Stage 8.

#### Stage 5 — `05_validate.py`
```bash
python3 05_validate.py -i deduped_batches/ [--rejects-dir rejects/05_validate] [--skip]
```
| Flag | Effect |
|------|--------|
| `--skip` | Skip integrity checks entirely |

#### Stage 6 — `06_stereo_filter.py`
```bash
python3 06_stereo_filter.py -i deduped_batches/ -o stereo_batches/ [--force-keep-rejected]
```
Removes one enantiomer from each racemic pair. Keeps the first canonical SMILES. Molecules with multiple fragments (complexes) are tagged `complex` and passed through.

#### Stage 7 — `07_reorder.py`
```bash
python3 07_reorder.py -i stereo_batches/ -o reordered_batches/ \
    [--backend rdkit] [--workers 4] [--force-keep-rejected]
```
| Flag | Effect |
|------|--------|
| `--backend` | `rdkit` (CanonicalRankAtoms, default) or `amber` (antechamber) |
| `--workers` | Parallel workers for AMBER backend (RDKit is single‑threaded) |

#### Stage 8 — `08_conformer_filter.py`
```bash
python3 08_conformer_filter.py -i reordered_batches/ -o conformer_batches/ \
    [--rmsd-threshold 1.0] [--force-keep-rejected]
```
| Flag | Effect |
|------|--------|
| `--rmsd-threshold` | Heavy‑atom Kabsch RMSD cutoff in Å (default: 1.0). Lower = more conformers kept |

#### Stage 9 — `09_stats.py`
```bash
python3 09_stats.py -i conformer_batches/ -o stats/ \
    [--tanimoto] [--workers 8] [--exclude COLUMN=VALUE]
```
| Flag | Effect |
|------|--------|
| `--tanimoto` | Compute ECFP4 Tanimoto nearest‑neighbour (O(n²), multiprocessed) |
| `--workers` | CPU workers for Tanimoto |
| `--exclude` | Drop rows matching `COLUMN=VALUE` before stats (repeatable) |

#### Stage 10 — `10_extxyz.py`
```bash
python3 10_extxyz.py -i conformer_batches/ -o extxyz/ \
    [--family QM40] [--exclude COLUMN=VALUE]
```
| Flag | Effect |
|------|--------|
| `--family` | Dataset name written to extXYZ metadata (default: QM40) |
| `--exclude` | Drop rows matching `COLUMN=VALUE` before writing (repeatable) |

### Report utility

```bash
python3 tools/report_stats.py stats/stats_summary.csv \
    --total-input 162954 --rejects-dir rejects/
```

Prints curation funnel (per‑stage drop counts), NAT/MolWt/TPSA/Energy descriptor summary, and Tanimoto diversity statistics (if `--tanimoto` was used in Stage 9).

### Running individual stages

```bash
# Validate only (stage 0)
python3 00_validate.py my_dataset.sdf -o my_dataset_valid.sdf

# Chemically filter with custom element set (stage 3)
python3 03_filter.py -i batches/ -o curated/ --preset neutral \
    --allowed-elements C,H,O,N --min-heavy 6

# Tighter conformer pruning (stage 8)
python3 08_conformer_filter.py -i reordered/ -o conformers/ --rmsd-threshold 0.5

# Generate extXYZ with dataset name override (stage 10)
python3 10_extxyz.py -i conformers/ -o extxyz/ --family MyDataset
```

### Adopting for a new dataset

1. Prepare an SDF with the [required tags](#input-sdf). Write a converter script (see `convert_qm40.py` as a template) if your data is in another format.
2. Run `00_validate.py` to check the contract.
3. Adjust `--allowed-elements`, `--min-heavy`, `--max-heavy` in Stage 3 for your chemistry.
4. Set `--family` in Stage 10 for correct extXYZ metadata.
5. Optionally skip Stage 2 (`--skip`) if your dataset has no energy column or you don't trust the OLS model for your atom types.

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
├── tools/                    # Utility scripts (reporting, inspection)
├── Makefile                  # Full pipeline orchestration
├── DATA_PRESERVATION.md      # Archival policy
└── README.md                 # This file
```

## Citation

If you use this pipeline in your research, please cite the accompanying manuscript:

> Quiñonero, J. A. et al. *AIM4ML: Automated Curation Pipeline for Quantum Chemistry Datasets*. (in preparation)

## License

[Specify your license here]
