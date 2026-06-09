# QM40 → AIM4ML processing pipeline
#
# Run from this directory (scripts/):
#   make           → full pipeline with --full-data (default)
#   make timed     → full pipeline + total elapsed summary
#   make stats     → run only stats stage (and dependencies if not stamped)
#   make FLAG=--sample → run on cluster sample (absolute paths, script default)
#
# Dependency chain:
#   parse → filter → energy_prefilter → dedup → validate → stereo_filter → reorder → stats → extxyz
#
# Stamp files in $(STAMPS)/ track which stages have completed.
# make only re-runs a stage if its stamp is missing or its dependency changed.
# Full data re-run: make clean-stamps && make
# Sample re-run:    make clean-sample && make FLAG=--sample
# Stamps are isolated per mode: full → $(AIM4ML)/.stamps; sample → $(SAMPLE_DIR)/.stamps

SHELL   := /bin/bash
PYTHON  := python3
FLAG    := --full-data
WORKERS := 1

AIM4ML      := /datos_pool/mldata1/QMdatasets/QM40/AIM4ML
SAMPLE_DIR  := $(AIM4ML)/samples
QTCOVI_HOST := qtcovi02

ifeq ($(FLAG),--sample)
STAMPS      := $(SAMPLE_DIR)/.stamps
else
STAMPS      := $(AIM4ML)/.stamps
endif

ifeq ($(FLAG),--sample)
_SAMPLE        := $(STAMPS)/sample.done
else
_SAMPLE        :=
endif

_PARSE              := $(STAMPS)/parse.done
_FILTER             := $(STAMPS)/filter.done
_DEDUP              := $(STAMPS)/dedup.done
_REORDER            := $(STAMPS)/reorder.done
_STEREO_FILTER      := $(STAMPS)/stereo_filter.done
_VALIDATE           := $(STAMPS)/validate.done
_STATS              := $(STAMPS)/stats.done
_EXTXYZ             := $(STAMPS)/extxyz.done
_ENERGY_PREFILTER   := $(STAMPS)/energy_prefilter.done

# ── Default target ────────────────────────────────────────────────────────────
.PHONY: all timed sample parse filter dedup reorder stereo_filter validate stats extxyz energy_prefilter selection check-template clean-stamps clean-sample guard-qtcovi help
all: guard-qtcovi extxyz

# ── Host guard (WORKERS > 1 requires qtcovi02) ───────────────────────────────
guard-qtcovi:
	@if [ "$(WORKERS)" -gt 1 ] 2>/dev/null && [ "$$(hostname -s)" != "$(QTCOVI_HOST)" ]; then \
	    echo ""; \
	    echo "ERROR: WORKERS=$(WORKERS) requires running on $(QTCOVI_HOST)."; \
	    echo "       Current host: $$(hostname -s)"; \
	    echo "       ssh qtcovi, then retry."; \
	    echo ""; \
	    exit 1; \
	fi

# ── Timed wrapper (total pipeline elapsed) ───────────────────────────────────
timed:
	@start=$$(date +%s); \
	 $(MAKE) all WORKERS=$(WORKERS) FLAG=$(FLAG); \
	 rc=$$?; \
	 elapsed=$$(( $$(date +%s) - start )); \
	 mins=$$(( elapsed / 60 )); secs=$$(( elapsed % 60 )); \
	 echo ""; \
	 echo "==> Total pipeline time: $${mins}m $${secs}s"; \
	 exit $$rc

# ── Stamp directory ───────────────────────────────────────────────────────────
$(STAMPS):
	@mkdir -p $@

# ── Stage 0: sample (only in sample mode) ────────────────────────────────────
# Reads full xyz/bond/main.csv → writes 200-molecule sample CSVs to SAMPLE_DIR.
ifeq ($(FLAG),--sample)
sample: $(_SAMPLE)

$(_SAMPLE): | $(STAMPS)
	@echo ""
	@echo "==> [0/9] sample — extracting 200-molecule sample from full QM40 data"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) 00-utils/sample_qm40.py; \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc
	@touch $@
else
sample:
	@echo "sample stage only runs with FLAG=--sample"
endif

# ── Stage 1: parse ────────────────────────────────────────────────────────────
# Reads xyz.csv → writes xyz_files/{Zinc_id}.xyz  (~8 min full data)
parse: $(_PARSE)

$(_PARSE): $(_SAMPLE) | $(STAMPS)
	@echo ""
	@echo "==> [1/9] parse — extracting XYZ files from xyz.csv"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) 01-parse/parse_qm40.py $(FLAG); \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc
	@touch $@

# ── Stage 2: filter ───────────────────────────────────────────────────────────
# D16 curation filter: neutral (charge=0) + closed-shell (no radicals).
# Reads main.csv → writes filtered_main.csv + logs/filter_rejected.csv
filter: $(_FILTER)

$(_FILTER): $(_PARSE)
	@echo ""
	@echo "==> [2/9] filter — D16 curation filter (neutral + closed-shell)"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) 00-utils/filter_qm40.py $(FLAG); \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc
	@touch $@

# ── Stage 3: dedup ────────────────────────────────────────────────────────────
# Assigns IDs (MD5), renames XYZ, builds SDF + Mulliken charges  (~12 min full data)
# Writes mol_files/, qm40_mapping.csv, logs/
dedup: $(_DEDUP)

$(_DEDUP): $(_ENERGY_PREFILTER)
	@echo ""
	@echo "==> [4/9] dedup — assigning compound IDs, building SDF files"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) 02-dedup/dedup_qm40.py $(FLAG) 2>/dev/null; \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc
	@touch $@

# ── Stage 4: reorder ──────────────────────────────────────────────────────────
# D17 atom reordering: AMBER canonical ordering via xyz_reord_qm40.sh.
# Reads mol_files/mol_*.xyz → reorders in-place; discards failures.
# Updates qm40_mapping.csv with reorder_status column.
reorder: $(_REORDER)

$(_REORDER): $(_STEREO_FILTER)
	@echo ""
	@echo "==> [7/9] reorder — AMBER canonical atom ordering (D17)"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) 00-utils/reorder_qm40.py $(FLAG) --workers $(WORKERS); \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc
	@touch $@

# ── Stage 5: stereo_filter ────────────────────────────────────────────────────
# D09 enantiomer filter: SMILES-based inversion (exact, no threshold).
# Reads canonical_SMILES column → adds stereo_status to qm40_mapping.csv.
# Marks removed_enantiomer entries; does NOT delete files (reversible policy).
stereo_filter: $(_STEREO_FILTER)

$(_STEREO_FILTER): $(_VALIDATE)
	@echo ""
	@echo "==> [6/9] stereo_filter — D09 enantiomer filter (SMILES-based)"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) 00-utils/stereo_filter_qm40.py $(FLAG); \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc
	@touch $@

# ── Stage 6: validate ─────────────────────────────────────────────────────────
# 9-check integrity validator. Exits non-zero on failures → stops the pipeline.
validate: $(_VALIDATE)

$(_VALIDATE): $(_DEDUP)
	@echo ""
	@echo "==> [5/9] validate — checking dedup output integrity vs xyz.csv"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) 00-utils/validate_qm40.py $(FLAG); \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc
	@touch $@

# ── Stage 7: stats ────────────────────────────────────────────────────────────
# Two scripts run sequentially:
#   stats_qm40.py --chiral → qm40_stats.csv (with max_tanimoto + max_tanimoto_chiral), histograms
#   check_stereo_pairs.py  → stereo_pairs.tsv, structural_twins.tsv
stats: $(_STATS)

$(_STATS): $(_REORDER)
	@echo ""
	@echo "==> [8/9] stats — descriptors, Tanimoto similarity, stereo pairs"
	@echo "    [7a] stats_qm40.py (non-chiral + chiral FP)"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) 03-stats/stats_qm40.py $(FLAG) --workers $(WORKERS) --chiral; \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc
	@echo "    [7b] check_stereo_pairs.py"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) 03-stats/check_stereo_pairs.py $(FLAG); \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc
	@touch $@

# ── Stage 8: extxyz ──────────────────────────────────────────────────────────
# D18: build batched extxyz trajectory files (BATCH_SIZE=5000 structures each).
# Reads mol_files/ + qm40_mapping.csv + filtered_main.csv + stats/qm40_stats.csv
# Writes 04-extxyz/qm40_single_batch{NNNN}.xyz with title-line key,value metadata.
extxyz: $(_EXTXYZ)

$(_EXTXYZ): $(_STATS)
	@echo ""
	@echo "==> [9/9] extxyz — D18 building extxyz trajectory batches"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) 04-extxyz/build_extxyz_qm40.py $(FLAG); \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc
	@touch $@

# ── Stage 3: energy_prefilter ────────────────────────────────────────────────
# QM40-specific energy outlier detection (atom-type OLS regression, residual z-score).
# Reads filtered_main.csv → writes energy_status column back to it.
# dedup reads energy_status and drops flagged molecules before building SDF/mapping.
# In --sample mode: exits gracefully if Internal_E(0K) absent in filtered CSV.
energy_prefilter: $(_ENERGY_PREFILTER)

$(_ENERGY_PREFILTER): $(_FILTER)
	@echo ""
	@echo "==> [3/9] energy_prefilter — QM40 energy outlier detection (atom-type OLS regression)"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) 00-utils/energy_prefilter_qm40.py $(FLAG); \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc
	@touch $@

# ── Selection pipeline (05-selection/) ───────────────────────────────────────
# Delegates to 05-selection/Makefile. Prerequisites: stages 1-9 complete.
# Not included in `all` — selection depends on D21/D22 supervisor decisions.
selection:
	@$(MAKE) -C 05-selection/

# ── One-time QA (no stamp — always re-runs when called) ──────────────────────
# Checks whether SMILES-template fallback molecules are aromatic-S (safe) or
# tautomer-prone (risk of silent wrong bond orders). See qm40-pipeline-plan.md §4.5.
check-template: $(_DEDUP)
	@echo ""
	@echo "==> check-template — QA for tautomer-prone template fallbacks"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) 00-utils/check_template_fallback.py $(FLAG); \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc

# ── Maintenance ───────────────────────────────────────────────────────────────
clean-stamps:
	@rm -f $(STAMPS)/*.done
	@echo "Stamps cleared — next 'make' will re-run from scratch."

clean-sample:
	@rm -rf $(SAMPLE_DIR)
	@echo "Sample directory cleared."

help:
	@echo ""
	@echo "QM40 → AIM4ML pipeline  |  run from: scripts/"
	@echo "Usage: make [target] [FLAG=--full-data] [WORKERS=1]"
	@echo ""
	@echo "  all              Full pipeline: [sample →] parse → filter → energy_prefilter → dedup → validate → stereo_filter → reorder → stats → extxyz"
	@echo "  timed            Same as all + total elapsed summary at end"
	@echo "  sample           Stage 0 — extract 200-mol sample CSVs (FLAG=--sample only)"
	@echo "  parse            Stage 1 — extract XYZ files from xyz.csv                          (~8 min)"
	@echo "  filter           Stage 2 — D16 curation filter (neutral + closed-shell)"
	@echo "  energy_prefilter Stage 3 — QM40 energy outlier detection (atom-type OLS regression)"
	@echo "  dedup            Stage 4 — assign IDs (32-char MD5), rename files, build SDF       (~12 min)"
	@echo "  validate         Stage 5 — 9-check integrity validator (dedup output vs xyz.csv)"
	@echo "  stereo_filter    Stage 6 — D09 enantiomer filter (SMILES-based)"
	@echo "  reorder          Stage 7 — AMBER canonical atom ordering (D17)                     (~4 h, 1 worker)"
	@echo "  stats            Stage 8 — descriptors, Tanimoto, stereo pairs                     (~80 min, 1 worker)"
	@echo "  extxyz           Stage 9 — D18 extxyz trajectory batches (5000 mol/file)"
	@echo "  selection        Delegate to 05-selection/Makefile (prepare_input → scaffold_groups)"
	@echo "  check-template   One-time QA — tautomer risk in template fallback"
	@echo "  clean-stamps     Remove full-data stamps → force full re-run"
	@echo "  clean-sample     Remove entire sample dir (stamps + all generated files)"
	@echo ""
	@echo "  Default: FLAG=--full-data (cluster full data)."
	@echo "  Sample:  FLAG=--sample  (cluster sample, absolute paths)."
	@echo "  Default: WORKERS=1. Parallel: make timed WORKERS=43"
	@echo ""
