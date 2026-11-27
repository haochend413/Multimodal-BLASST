#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
manifests="$HOME/blasst-results"
prefill_lambda=1
results="$manifests/decode-lambda-experiments/prefill-lambda-$prefill_lambda"

if (( $# == 0 )); then
  set -- 0.001 0.01 0.1 0.6 1
fi

for decode_lambda in "$@"; do
  run_dir="$results/decode-lambda-$decode_lambda"
  mkdir -p "$run_dir"
  python "$script_dir/attention_study.py" \
    --benchmark vqav2 \
    --manifest "$manifests/vqav2-random-val-1000-seed20261124.json" \
    --sample-count 50 --decode-tokens 16 \
    --lambda "$prefill_lambda" --decode-lambda "$decode_lambda" \
    --decode-tv-drop-sweep 5,10,20,25,35,50,60,75,80,90,95,99,99.5,99.9 \
    --random-seed 7 --vqa-shards 1 --vqa-shard-index 0 \
    --output "$run_dir/vqav2-decode-tv-drop-seed7.json"
done
