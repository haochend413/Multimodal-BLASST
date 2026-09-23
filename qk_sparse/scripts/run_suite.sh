#!/usr/bin/env bash
# Run the cross-modality suite and build the report.
#
# Staged so a failure in one modality never costs you the others: the synthetic
# controls run first (they need no network and prove the harness works), then
# each real model in turn with --keep-going.
#
# Usage:
#   ./scripts/run_suite.sh                      # full suite, auto device
#   OUT=run2 SAMPLES=4 ./scripts/run_suite.sh   # more inputs per model
#   ADAPTERS="text video" ./scripts/run_suite.sh
#   DATA_DIR=/path/to/assets ./scripts/run_suite.sh   # offline inputs
set -euo pipefail

cd "$(dirname "$0")/.."

OUT=${OUT:-results}
SAMPLES=${SAMPLES:-2}
LAYERS=${LAYERS:-4}
HEADS=${HEADS:-4}
MAX_BLOCKS=${MAX_BLOCKS:-16}
DENSITY=${DENSITY:-0.05}
PYTHON=${PYTHON:-python}
ADAPTERS=${ADAPTERS:-"synthetic-high synthetic-low text image video audio timeseries"}
DATA_DIR=${DATA_DIR:-}

common=(--out "$OUT" --samples "$SAMPLES" --layers "$LAYERS"
        --heads "$HEADS" --max-blocks "$MAX_BLOCKS" --keep-going)
[[ -n $DATA_DIR ]] && common+=(--data-dir "$DATA_DIR")

echo "=== controls (no network, no weights) ==="
$PYTHON -m qk_sparse run --adapters synthetic-high synthetic-low "${common[@]}"

for adapter in $ADAPTERS; do
  case $adapter in synthetic-*) continue ;; esac
  echo "=== $adapter ==="
  # One adapter per invocation: a CUDA OOM or a missing codec then costs only
  # this modality, and the JSON already written for the others survives.
  $PYTHON -m qk_sparse run --adapters "$adapter" "${common[@]}" || {
    echo "!! $adapter failed; continuing" >&2
  }
done

echo "=== report ==="
$PYTHON -m qk_sparse report "$OUT" --density "$DENSITY" --out "$OUT/report.md"
cat "$OUT/report.md"
