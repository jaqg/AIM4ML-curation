# AIM4ML — Pipeline Scripts

Processing pipeline for the QM40 → AIM4ML dataset.
Each numbered stage corresponds to a subdirectory; run stages in order.

## Pipeline overview

| Stage | Directory | Description | Status |
|-------|-----------|-------------|--------|
| 00 | `00-utils/` | Sampling and validation utilities | Done |
| 01 | `01-parse/` | Extract XYZ files from QM40 CSV | Done |
| 02 | `02-dedup/` | Assign compound IDs, build SDF/MOL2 | Done |
| 03 | `03-stats/` | Dataset statistics | In progress |
| 04 | `04-qm-inputs/` | Generate QM input files | Pending |
| 05 | `05-qm-outputs/` | Parse QM output files | Pending |
| 06 | `06-promolden/` | PROMOLDEN input/output handling | Pending |
| 07 | `07-features/` | Feature extraction | Pending |

All scripts support two modes via `--full-data`:
- **default (sample):** local paths under `../../samples/qm40/` — for development
- **`--full-data`:** cluster paths under `/datos_pool/mldata1/QMdatasets/QM40/` — for production

---

## Running the pipeline with `make`

A `Makefile` at the root of `scripts/` orchestrates stages 01–03. Run from `scripts/`:

```bash
make                  # full pipeline on cluster data: parse → dedup → validate → stats
make FLAG=            # same pipeline on local sample (no --full-data)
make parse            # run only stage 1 (and any missing prerequisites)
make dedup            # run only stage 2 (and any missing prerequisites)
make validate         # run only stage 3 (and any missing prerequisites)
make stats            # run only stage 4 (and any missing prerequisites)
make check-template   # one-time QA: classify tautomer-prone template-fallback molecules
make clean-stamps     # remove all stamps → force full re-run from scratch
make help             # print usage summary
```

### How it works

Stamp files in `$AIM4ML/.stamps/` track which stages have completed (`parse.done`,
`dedup.done`, `validate.done`, `stats.done`). Make only re-runs a stage if its stamp
is missing. The dependency chain is:

```
parse → dedup → validate → stats
```

`validate` exits non-zero on integrity failures, which stops the pipeline before `stats` runs.

### Expected output during `make all`

```
==> [1/4] parse — extracting XYZ files from xyz.csv
Writing XYZ: 100%|████████████| 162954/162954 [08:12<00:00, 330mol/s]

==> [2/4] dedup — assigning compound IDs, building SDF files
Processing: 100%|████████████| 162954/162954 [12:04<00:00, 225mol/s]

==> [3/4] validate — checking pipeline output integrity
Validating: 100%|████████████| 162954/162954 [02:30<00:00, 1083mol/s]
Results: 162954/162954 molecules passed all checks

==> [4/4] stats — descriptors, Tanimoto similarity, stereo pairs
    [4a] stats_qm40.py (non-chiral FP)
Descriptors: 100%|███████████| 162954/162954 [...]
Fingerprints: 100%|██████████| 162954/162954 [...]
Tanimoto NN: 100%|███████████| 162954/162954 [...]
    [4b] stats_qm40_chiral.py (chiral FP)
    [4c] check_stereo_pairs.py
```

### `check-template` (one-time QA)

Not stamped — always re-runs when called. Reads `logs/topology_fixed.txt` and
`logs/sdf_recovered.txt` from the dedup stage, classifies each molecule as safe
(aromatic-S) or tautomer-prone, and writes flagged molecules to
`logs/template_fallback_tautomer_flags.tsv`. See `qm40-pipeline-plan.md §4.5`.

```bash
make check-template             # uses --full-data by default
make check-template FLAG=       # run on local sample
```

---

## Stage 00 — Utilities (`00-utils/`)

### `sample_qm40.py`
Creates local sample CSVs (first 200 molecules) from the full QM40 dataset for development.

**Input:** Full QM40 CSVs on cluster (`main.csv`, `xyz.csv`, `bond.csv`)  
**Output:** `sample_main.csv`, `sample_xyz.csv`, `sample_bond.csv` (written to CWD)  
**Usage:** Run once from the cluster to generate sample files; not intended for pipeline automation.

```bash
python3 sample_qm40.py
```

---

### `validate_qm40.py`
Data integrity validator for pipeline output. Cross-checks XYZ/SDF files and the mapping CSV
against the raw QM40 source data (9 checks).

**Checks performed:**
1. File existence — every mapping row has an XYZ and SDF on disk
2. NAT consistency — mapping `NAT` matches XYZ atom count
3. Atom ordering — element sequence matches `xyz.csv` exactly
4. Coordinates — XYZ coords match `xyz.csv` `final_x/y/z` (±1e-4 Å)
5. Bond count — SDF bond count matches `bond.csv`
6. Mulliken charges — SDF `MullikenCharges` matches `xyz.csv` charges (±1e-4 e)
7. Charge sum — total Mulliken charge ≈ 0 for neutral molecules (±0.05 e)
8. SMILES — SDF `SMILES` field matches `canonical_SMILES` in mapping
9. ID uniqueness — no MD5 hash collisions in the mapping CSV

**Input:** `xyz.csv`, `bond.csv`, `qm40_mapping.csv`, `mol_files/`  
**Output:** Pass/fail report to stdout  
**Usage:**

```bash
python3 validate_qm40.py               # validate sample output
python3 validate_qm40.py --full-data   # validate full QM40 output on cluster
```

**Dependencies:** `pandas`, `rdkit`

---

## Stage 01 — Parse (`01-parse/`)

### `parse_qm40.py`
Extracts one XYZ file per molecule from the QM40 `xyz.csv` coordinate table.
Uses DFT-optimised coordinates (`final_x/y/z`). Output filename: `{Zinc_id}.xyz`.

**Input:** `xyz.csv`  
**Output:** One `.xyz` file per molecule in `OUTPUT_DIR`  
**XYZ format:**
```
<NAT>
<Zinc_id>  <SMILES>
<El>  <x>  <y>  <z>
...
```
**Usage:**

```bash
python3 parse_qm40.py               # local sample
python3 parse_qm40.py --full-data   # full dataset on cluster
```

**Dependencies:** `pandas`

---

## Stage 02 — Deduplication (`02-dedup/`)

### `dedup_qm40.py`
Assigns compound IDs and builds the canonical molecular files from the per-molecule XYZ files
produced by stage 01. Core steps:

1. **Conformer check** — verifies each `Zinc_id` appears exactly once in `main.csv`; stops if conformers are found (not yet implemented).
2. **Canonicalise SMILES** — uses RDKit to produce a canonical SMILES for each molecule.
3. **Compound ID** — MD5(canonical SMILES)[:12] → `ID` column; unique across the dataset.
4. **Rename XYZ** — copies `{Zinc_id}.xyz` → `mol_{ID}_1.xyz` (`ICONF=1` for QM40).
5. **Build SDF** — 3D geometry + Mulliken charges. Two-level fallback:
   - Primary: `rdDetermineBonds` (connectivity from `bond.csv`, bond orders from Hückel geometry).
   - Fallback: `AssignBondOrdersFromTemplate` from canonical SMILES — fixes aromatic sulfur heterocycles (thiophene, thiazole, etc.) that confuse `DetermineBondOrders`.
6. **Write mapping CSV** — `qm40_mapping.csv` with one row per molecule (see schema below).

**Input:** `main.csv`, `xyz.csv`, `bond.csv`, `xyz_files/`  
**Output:** `mol_files/mol_{ID}_1.{xyz,sdf}`, `qm40_mapping.csv`, `logs/`  
**Usage:**

```bash
python3 dedup_qm40.py               # local sample
python3 dedup_qm40.py --full-data   # full dataset on cluster
```

**Dependencies:** `pandas`, `rdkit`

#### `qm40_mapping.csv` schema

| Column | Type | Description |
|--------|------|-------------|
| `Zinc_id` | str | Original ZINC identifier from QM40 |
| `canonical_SMILES` | str | RDKit canonical SMILES — source of truth for chemical identity |
| `ID` | str | MD5(canonical_SMILES)[:12] — compound identifier |
| `ICONF` | int | Conformer index (always 1 for QM40) |
| `NAT` | int | Number of atoms (including H) |
| `sdf_status` | str | SDF generation outcome |

#### `sdf_status` values

| Value | SDF written? | Bond orders correct? | When |
|-------|-------------|----------------------|------|
| `ok` | Yes | Yes | `DetermineBondOrders` + topology check passed |
| `topology_fixed` | Yes | Yes | Topology wrong; SMILES template fixed it |
| `sdf_recovered` | Yes | Yes | `DetermineBondOrders` raised; SMILES template succeeded |
| `topology_warning` | Yes | **Suspect** | Both approaches failed; bond orders unreliable |
| `sdf_failed` | **No** | — | Both approaches raised; XYZ file is still correct |

Full-run QM40 counts: `ok` ~162 335 · `topology_fixed` 1 303 · `sdf_recovered` ~2 291 · `topology_warning` 204 · `sdf_failed` 211 → **99.74% valid SDFs**.

**Key invariant:** `canonical_SMILES` is always correct for all rows. SDF is the source of truth only for 3D geometry, not chemical identity.

---

### `make_mol2.sh`
Batch-converts all `mol_*.sdf` files in `mol_files/` to MOL2 format using OpenBabel.
Molecules with `sdf_status=sdf_failed` are skipped automatically (no `.sdf` exists for them).

Charges in the MOL2 file are Gasteiger (OpenBabel default). Mulliken charges remain only in the
SDF `MullikenCharges` data field and are **not** transferred to the MOL2 charge column.

**Input:** `mol_files/mol_*.sdf`  
**Output:** `mol_files/mol_*.mol2` (alongside each SDF)  
**Requires:** `obabel` on `PATH` (load the OpenBabel module on the cluster)  
**Usage:**

```bash
bash make_mol2.sh               # local sample
bash make_mol2.sh --full-data   # full dataset on cluster
```

---

### `check_template_fallback.py`
One-time QA script for molecules that were processed via the SMILES template fallback
(`topology_fixed` and `sdf_recovered` rows in `qm40_mapping.csv`). Classifies each
molecule as:

| Label | Meaning |
|-------|---------|
| `SAFE` | Matches only aromatic-S ring patterns — the expected failure class for `DetermineBondOrders`; template assignment is reliable |
| `FLAGGED` | Contains a tautomer-prone group (amide, thioamide, enol, iminol, amidine) — silent wrong bond orders are possible if the DFT structure is a different tautomeric form than the QM40 SMILES |
| `OTHER` | Neither pattern matches — inspect manually |

Full-run QM40 result: all template-fallback molecules were `SAFE` (aromatic-S only) — tautomer risk is negligible.

**Input:** `qm40_mapping.csv`, `logs/topology_fixed.txt`, `logs/sdf_recovered.txt`  
**Output:** `logs/template_fallback_tautomer_flags.tsv` (written only if `FLAGGED` molecules exist)  
**Usage:**

```bash
python3 check_template_fallback.py               # local sample
python3 check_template_fallback.py --full-data   # full dataset on cluster
```

Called via `make check-template` (no stamp — always re-runs).

**Dependencies:** `pandas`, `rdkit`

---

## Stage 03 — Statistics (`03-stats/`)

Three scripts run sequentially inside the `make stats` target (stage 4 in the pipeline).
All three read from `qm40_mapping.csv` or from the stats CSV produced by this stage.

### `stats_qm40.py`
Computes per-molecule descriptors and chemical diversity statistics using non-chirality-aware
Morgan fingerprints. The output `qm40_stats.csv` is the master stats file consumed by
`check_stereo_pairs.py`.

**Descriptors computed:**

| Column | Description |
|--------|-------------|
| `MolWt` | Molecular weight (RDKit, includes H) |
| `TPSA` | Topological polar surface area (Å²) |
| `charge` | Formal charge (0 for all QM40 — neutral singlets) |
| `max_tanimoto` | Nearest-neighbour Tanimoto similarity (Morgan r=2, 2048 bits, non-chiral) — maximum over all other molecules |

The nearest-neighbour Tanimoto uses RDKit's `BulkTanimotoSimilarity` (POPCNT-accelerated,
~26 B comparisons for QM40; completes in under a minute on cluster hardware).

**Input:** `qm40_mapping.csv`  
**Output:** `qm40_stats.csv`, `plots/hist_nat.pdf`, `plots/hist_tpsa.pdf`, `plots/hist_tanimoto.pdf`  
**Usage:**

```bash
python3 stats_qm40.py               # local sample
python3 stats_qm40.py --full-data   # full dataset on cluster
```

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `tqdm`, `rdkit`

---

### `stats_qm40_chiral.py`
Identical to `stats_qm40.py` except Morgan fingerprints are computed with
`includeChirality=True`. This gives a correct Tanimoto distribution for stereoisomer-rich
datasets: enantiomers and diastereomers receive distinct bit vectors instead of Tanimoto = 1.0.

**Context:** the non-chiral run reported mean Tanimoto 0.624 and 15.2% near-duplicates (>0.85),
but `check_stereo_pairs.py` showed that 14.8% of those were stereo pairs — a measurement
artefact, not real chemical redundancy. This script corrects the diversity estimate.

Outputs go to separate files so the non-chiral run is not overwritten.

**Input:** `qm40_mapping.csv`  
**Output:** `qm40_stats_chiral.csv`, `plots/hist_tanimoto_chiral.pdf`  
**Usage:**

```bash
python3 stats_qm40_chiral.py               # local sample
python3 stats_qm40_chiral.py --full-data   # full dataset on cluster
```

**Dependencies:** `pandas`, `numpy`, `matplotlib`, `tqdm`, `rdkit`

---

### `check_stereo_pairs.py`
Decomposes the near-duplicate population (Tanimoto > 0.85) from `stats_qm40.py` into:

- **Stereo pairs** — same 2D molecular graph, different stereochemistry (strip-stereo SMILES
  match). These account for all Tanimoto = 1.000 cases and should go to the **same** data
  split to avoid information leakage.
- **Structural twins** — near-duplicates (Tanimoto > 0.85) that are NOT stereo partners;
  genuinely similar but distinct molecules that also risk data leakage.

The script strips stereo information via `Chem.MolToSmiles(mol, isomericSmiles=False)` and
uses pandas `duplicated` to find groups sharing the same 2D scaffold.

**Input:** `qm40_stats.csv` (output of `stats_qm40.py`)  
**Output:** `logs/stereo_pairs.tsv`, `logs/structural_twins.tsv`  
**Usage:**

```bash
python3 check_stereo_pairs.py               # local sample
python3 check_stereo_pairs.py --full-data   # full dataset on cluster
```

**Dependencies:** `pandas`, `tqdm`, `rdkit`

#### ML splitting implication
Stereo-partner molecules must be kept in the same train/test/val split. Both stereo pairs
and structural twins risk label leakage in a random split. Use scaffold-aware splitting
(Butina / Bemis-Murcko) rather than a random split.
