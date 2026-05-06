# QM40 → AIM4ML processing pipeline
#
# Run from this directory (scripts/):
#   make           → full pipeline with --full-data (default)
#   make stats     → run only stats stage (and dependencies if not stamped)
#   make FLAG=     → run on local sample instead of cluster data
#
# Dependency chain:
#   parse → filter → dedup → reorder → stereo_filter → validate → stats
#
# Stamp files in $(STAMPS)/ track which stages have completed.
# make only re-runs a stage if its stamp is missing or its dependency changed.
# To force a full re-run: make clean-stamps && make

SHELL  := /bin/bash
PYTHON := python3
FLAG   := --full-data

AIM4ML := /datos_pool/mldata1/QMdatasets/QM40/AIM4ML
STAMPS := $(AIM4ML)/.stamps

_PARSE         := $(STAMPS)/parse.done
_FILTER        := $(STAMPS)/filter.done
_DEDUP         := $(STAMPS)/dedup.done
_REORDER       := $(STAMPS)/reorder.done
_STEREO_FILTER := $(STAMPS)/stereo_filter.done
_VALIDATE      := $(STAMPS)/validate.done
_STATS         := $(STAMPS)/stats.done

# ── Default target ────────────────────────────────────────────────────────────
.PHONY: all parse filter dedup reorder stereo_filter validate stats check-template clean-stamps help
all: stats

# ── Stamp directory ───────────────────────────────────────────────────────────
$(STAMPS):
	@mkdir -p $@

# ── Stage 1: parse ────────────────────────────────────────────────────────────
# Reads xyz.csv → writes xyz_files/{Zinc_id}.xyz  (~8 min full data)
parse: $(_PARSE)

$(_PARSE): | $(STAMPS)
	@echo ""
	@echo "==> [1/6] parse — extracting XYZ files from xyz.csv"
	$(PYTHON) 01-parse/parse_qm40.py $(FLAG)
	@touch $@

# ── Stage 2: filter ───────────────────────────────────────────────────────────
# D16 curation filter: neutral (charge=0) + closed-shell (no radicals).
# Reads main.csv → writes filtered_main.csv + logs/filter_rejected.csv
filter: $(_FILTER)

$(_FILTER): $(_PARSE)
	@echo ""
	@echo "==> [2/6] filter — D16 curation filter (neutral + closed-shell)"
	$(PYTHON) 00-utils/filter_qm40.py $(FLAG)
	@touch $@

# ── Stage 3: dedup ────────────────────────────────────────────────────────────
# Assigns IDs (MD5), renames XYZ, builds SDF + Mulliken charges  (~12 min full data)
# Writes mol_files/, qm40_mapping.csv, logs/
dedup: $(_DEDUP)

$(_DEDUP): $(_FILTER)
	@echo ""
	@echo "==> [3/6] dedup — assigning compound IDs, building SDF files"
	$(PYTHON) 02-dedup/dedup_qm40.py $(FLAG) 2>/dev/null
	@touch $@

# ── Stage 4: reorder ──────────────────────────────────────────────────────────
# D17 atom reordering: AMBER canonical ordering via xyz_reord_qm40.sh.
# Reads mol_files/mol_*.xyz → reorders in-place; discards failures.
# Updates qm40_mapping.csv with reorder_status column.
reorder: $(_REORDER)

$(_REORDER): $(_DEDUP)
	@echo ""
	@echo "==> [4/6] reorder — AMBER canonical atom ordering (D17)"
	$(PYTHON) 00-utils/reorder_qm40.py $(FLAG)
	@touch $@

# ── Stage 5: stereo_filter ────────────────────────────────────────────────────
# D09 enantiomer filter: Kabsch RMSD + mirror reflection on heavy atoms.
# Reads mol_files/mol_*.xyz → adds stereo_status column to qm40_mapping.csv.
# Marks removed_enantiomer entries; does NOT delete files (reversible policy).
stereo_filter: $(_STEREO_FILTER)

$(_STEREO_FILTER): $(_REORDER)
	@echo ""
	@echo "==> [5/7] stereo_filter — D09 enantiomer filter (Kabsch RMSD)"
	$(PYTHON) 00-utils/stereo_filter_qm40.py $(FLAG)
	@touch $@

# ── Stage 6: validate ─────────────────────────────────────────────────────────
# 9-check integrity validator. Exits non-zero on failures → stops the pipeline.
validate: $(_VALIDATE)

$(_VALIDATE): $(_STEREO_FILTER)
	@echo ""
	@echo "==> [6/7] validate — checking pipeline output integrity"
	$(PYTHON) 00-utils/validate_qm40.py $(FLAG)
	@touch $@

# ── Stage 7: stats ────────────────────────────────────────────────────────────
# Three scripts run sequentially:
#   stats_qm40.py        → qm40_stats.csv, histograms  (~80 min full data)
#   stats_qm40_chiral.py → qm40_stats_chiral.csv, histograms
#   check_stereo_pairs.py → stereo_pairs.tsv, structural_twins.tsv
stats: $(_STATS)

$(_STATS): $(_VALIDATE)
	@echo ""
	@echo "==> [7/7] stats — descriptors, Tanimoto similarity, stereo pairs"
	@echo "    [7a] stats_qm40.py (non-chiral FP)"
	$(PYTHON) 03-stats/stats_qm40.py $(FLAG)
	@echo "    [7b] stats_qm40_chiral.py (chiral FP)"
	$(PYTHON) 03-stats/stats_qm40_chiral.py $(FLAG)
	@echo "    [7c] check_stereo_pairs.py"
	$(PYTHON) 03-stats/check_stereo_pairs.py $(FLAG)
	@touch $@

# ── One-time QA (no stamp — always re-runs when called) ──────────────────────
# Checks whether SMILES-template fallback molecules are aromatic-S (safe) or
# tautomer-prone (risk of silent wrong bond orders). See qm40-pipeline-plan.md §4.5.
check-template: $(_DEDUP)
	@echo ""
	@echo "==> check-template — QA for tautomer-prone template fallbacks"
	$(PYTHON) 00-utils/check_template_fallback.py $(FLAG)

# ── Maintenance ───────────────────────────────────────────────────────────────
clean-stamps:
	@rm -f $(STAMPS)/*.done
	@echo "Stamps cleared — next 'make' will re-run from scratch."

help:
	@echo ""
	@echo "QM40 → AIM4ML pipeline  |  run from: scripts/"
	@echo "Usage: make [target] [FLAG=--full-data]"
	@echo ""
	@echo "  all              Full pipeline: parse → filter → dedup → reorder → stereo_filter → validate → stats"
	@echo "  parse            Stage 1 — extract XYZ files from xyz.csv                          (~8 min)"
	@echo "  filter           Stage 2 — D16 curation filter (neutral + closed-shell)"
	@echo "  dedup            Stage 3 — assign IDs, rename files, build SDF                     (~12 min)"
	@echo "  reorder          Stage 4 — AMBER canonical atom ordering (D17)"
	@echo "  stereo_filter    Stage 5 — D09 enantiomer filter (Kabsch RMSD)"
	@echo "  validate         Stage 6 — 9-check integrity validator"
	@echo "  stats            Stage 7 — descriptors, Tanimoto, stereo pairs                     (~80 min)"
	@echo "  check-template   One-time QA — tautomer risk in template fallback"
	@echo "  clean-stamps     Remove all stamps → force full re-run"
	@echo ""
	@echo "  Default: FLAG=--full-data (cluster). Local sample: make all FLAG="
	@echo ""
