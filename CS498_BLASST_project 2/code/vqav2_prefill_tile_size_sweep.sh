#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
manifests="$HOME/blasst-results"
results="$manifests/prefill-tile-size-experiments"

if (( $# == 0 )); then
  set -- 32 64
fi

for tile in "$@"; do
  run_dir="$results/${tile}x${tile}"
  mkdir -p "$run_dir"
  python "$script_dir/attention_study.py" \
    --benchmark vqav2 \
    --manifest "$manifests/vqav2-random-val-1000-seed20261124.json" \
    --sample-count 50 --decode-tokens 16 --decode-lambda 0.1 \
    --prefill-tile-size "$tile" --prefill-vv-lambda-sweep 1 \
    --prefill-vv-drop-sweep 5,10,20,25,50,75,90,95,99 \
    --random-seed 7 --vqa-shards 1 --vqa-shard-index 0 \
    --output "$run_dir/vqav2-prefill-vv-drop-lambda1-seed7.json"
done
