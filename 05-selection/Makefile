# AIM4ML selection pipeline — QM40 (Phase 2)
#
# Run from scripts/05-selection/ or via top-level:  make selection
# Prerequisites: pipeline stages 1-9 complete (make all from scripts/).
#
# Dependency chain:
#   prepare_input → scaffold_groups_qm40
#
# Stamp files in $(STAMPS)/ track completed stages.
# Force re-run: make clean-stamps && make

SHELL  := /bin/bash
PYTHON := python3

AIM4ML := /datos_pool/mldata1/QMdatasets/QM40/AIM4ML
STAMPS := $(AIM4ML)/05-selection/.stamps

_PREPARE_INPUT   := $(STAMPS)/prepare_input.done
_SCAFFOLD_GROUPS := $(STAMPS)/scaffold_groups_qm40.done
_SELECTION_MAIN  := $(STAMPS)/selection_main.done

METRIC     ?= morgan
K          ?= 10000
TIER1_MIN  ?= 50
SEED       ?= 42
WORKERS    ?= 1

.PHONY: all prepare_input scaffold_groups_qm40 selection_main clean-stamps help

all: selection_main

# ── Stamp directory ───────────────────────────────────────────────────────────
$(STAMPS):
	@mkdir -p $@

# ── S1: prepare_input ─────────────────────────────────────────────────────────
# Merges qm40_stats.csv + energy_status.csv, applies four filter conditions
# (reorder_status=success, stereo_status=kept, sdf_status≠sdf_failed,
#  energy_status≠flagged), writes selection/qm40_selection_input.csv.
prepare_input: $(_PREPARE_INPUT)

$(_PREPARE_INPUT): | $(STAMPS)
	@echo ""
	@echo "==> [S1] prepare_input — build QM40 selection pool CSV"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) prepare_input.py; \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc
	@touch $@

# ── S2: scaffold_groups_qm40 ─────────────────────────────────────────────────
# Bemis-Murcko scaffold grouping. Reads qm40_selection_input.csv.
# Writes selection/scaffold_groups.csv + selection/mol_scaffold_map.csv.
# Informs D21 (budget allocation) and D22 (diversity dimension).
scaffold_groups_qm40: $(_SCAFFOLD_GROUPS)

$(_SCAFFOLD_GROUPS): $(_PREPARE_INPUT)
	@echo ""
	@echo "==> [S2] scaffold_groups_qm40 — Bemis-Murcko scaffold grouping (D22)"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) scaffold_groups_qm40.py; \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc
	@touch $@

# ── S3: selection_main ────────────────────────────────────────────────────────
# Scaffold-aware MaxMin selection (D21/D22).
# Tier 1: per-scaffold MaxMin (groups > TIER1_MIN); Tier 2: global MaxMin.
# Sensitivity runs: make selection_main TIER1_MIN=20 / TIER1_MIN=100
# D21 trial:        make selection_main METRIC=soap WORKERS=8
selection_main: $(_SELECTION_MAIN)

$(_SELECTION_MAIN): $(_SCAFFOLD_GROUPS)
	@echo ""
	@echo "==> [S3] selection_main — scaffold-aware MaxMin (metric=$(METRIC), k=$(K), tier1_min=$(TIER1_MIN))"
	@echo "    Started:  $$(date '+%Y-%m-%d %H:%M:%S')"
	@start=$$(date +%s); \
	 $(PYTHON) selection_main.py \
	   --metric $(METRIC) --k $(K) --tier1-min $(TIER1_MIN) \
	   --seed $(SEED) --workers $(WORKERS); \
	 rc=$$?; elapsed=$$(( $$(date +%s) - start )); \
	 echo "    Finished: $$(date '+%Y-%m-%d %H:%M:%S')  [elapsed: $${elapsed}s]"; \
	 exit $$rc
	@touch $@

# ── Maintenance ───────────────────────────────────────────────────────────────
clean-stamps:
	@rm -f $(STAMPS)/*.done
	@echo "Selection stamps cleared — next 'make' will re-run from scratch."

help:
	@echo ""
	@echo "AIM4ML selection pipeline  |  run from: scripts/05-selection/"
	@echo "Usage: make [target] [METRIC=morgan|soap] [K=10000] [TIER1_MIN=50] [WORKERS=1]"
	@echo ""
	@echo "  all                  Full pipeline: prepare_input → scaffold_groups → selection_main"
	@echo "  prepare_input        S1 — build QM40 selection pool CSV (149,752 rows)"
	@echo "  scaffold_groups_qm40 S2 — Bemis-Murcko grouping → scaffold_groups.csv"
	@echo "  selection_main       S3 — scaffold-aware MaxMin (D21/D22)"
	@echo "  clean-stamps         Remove selection stamps → force re-run"
	@echo ""
	@echo "  Sensitivity runs (D22-P3):  make selection_main TIER1_MIN=20"
	@echo "                              make selection_main TIER1_MIN=100"
	@echo "  D21 trial (SOAP):           make selection_main METRIC=soap WORKERS=8"
	@echo ""
	@echo "  Prerequisites: pipeline stages 1-9 complete (make all from scripts/)."
	@echo ""
