#!/usr/bin/env bash
# Compare dense attention vs BLASST (skip-softmax) vs RocketKV on LongBench v1.
#
# For every run this produces, under $OUT/<run>/:
#   cfg.yaml                  the exact LLM API config used
#   samples_longbench.json    accuracy (trtllm-eval)
#   report.json               TTFT / TPOT / throughput (trtllm-bench)
#   iter.log                  per-iteration stats incl. KV-cache usage
# Then run: python compare.py $OUT
#
# Usage:
#   MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507 ./run_sweep.sh
#   NUM_SAMPLES=5 ./run_sweep.sh             # quick smoke test
#   RUNS="dense blasst_sp50" ./run_sweep.sh  # subset of runs
set -euo pipefail

MODEL=${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
OUT=${OUT:-results}
MAX_BATCH=${MAX_BATCH:-64}
MAX_TOKENS=${MAX_TOKENS:-100000}
KV_FRACTION=${KV_FRACTION:-0.8}
NUM_SAMPLES=${NUM_SAMPLES:-}   # per LongBench subtask; empty = full dataset
SKIP_EVAL=${SKIP_EVAL:-0}
SKIP_BENCH=${SKIP_BENCH:-0}

# BLASST thresholds below were calibrated for Qwen3-30B-A3B-Instruct-2507
# (blog16). For any other model, sweep your own values.
sparse_cfg() {
  case $1 in
    dense) ;;
    blasst_sp50) printf 'algorithm: skip_softmax\n  threshold_scale_factor: {prefill: 587.18, decode: 16.52}' ;;
    blasst_sp70) printf 'algorithm: skip_softmax\n  threshold_scale_factor: {prefill: 3293.04, decode: 118.62}' ;;
    blasst_sp90) printf 'algorithm: skip_softmax\n  threshold_scale_factor: {prefill: 18471.56, decode: 852.20}' ;;
    rocket_b*)   printf 'algorithm: rocket\n  prompt_budget: %s' "${1#rocket_b}" ;;
    *) echo "Unknown run: $1" >&2; return 1 ;;
  esac
}
DEFAULT_RUNS="dense blasst_sp50 blasst_sp70 blasst_sp90 rocket_b1024 rocket_b2048 rocket_b4096"
RUNS=${RUNS:-$DEFAULT_RUNS}

# RocketKV needs compute capability >= 10.0 (Blackwell).
compute_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "GPU compute capability: ${compute_cap}"
if [[ ${compute_cap%%.*} -lt 10 ]]; then
  echo "RocketKV is not supported on this GPU; skipping rocket_* runs."
  RUNS=$(tr ' ' '\n' <<<"$RUNS" | grep -v '^rocket_' | tr '\n' ' ')
fi

write_cfg() {
  local run=$1 dir=$2
  # Identical non-sparse settings for every run. Block reuse and chunked
  # prefill are off because RocketKV requires it; keep them off everywhere so
  # the comparison is fair.
  cat >"$dir/cfg.yaml" <<EOF
attn_backend: TRTLLM
enable_chunked_prefill: false
kv_cache_config:
  free_gpu_memory_fraction: ${KV_FRACTION}
  enable_block_reuse: false
EOF
  local sparse
  sparse=$(sparse_cfg "$run")
  if [[ -n $sparse ]]; then
    printf 'sparse_attention_config:\n  %s\n' "$sparse" >>"$dir/cfg.yaml"
  fi
}

# Dense runs first: its dumped prompts are the benchmark dataset for every run,
# so all methods are timed on identical inputs and output lengths.
ordered_runs="dense $(tr ' ' '\n' <<<"$RUNS" | grep -v '^dense$' | tr '\n' ' ')"
dataset="$OUT/dense/dumped_ids.json"

for run in $ordered_runs; do
  sparse_cfg "$run" >/dev/null || exit 1
  dir="$OUT/$run"
  mkdir -p "$dir"
  write_cfg "$run" "$dir"
  echo "=== $run ==="; cat "$dir/cfg.yaml"

  if [[ $SKIP_EVAL != 1 ]]; then
    trtllm-eval --model "$MODEL" \
      --max_batch_size "$MAX_BATCH" --max_num_tokens "$MAX_TOKENS" \
      --config "$dir/cfg.yaml" \
      longbench_v1 \
      --output_path "$dir" --output_dir "$dir" \
      ${NUM_SAMPLES:+--num_samples "$NUM_SAMPLES"} \
      2>&1 | tee "$dir/eval.log"
  fi

  if [[ $SKIP_BENCH != 1 ]]; then
    [[ -f $dataset ]] || { echo "Missing $dataset: run the dense eval first." >&2; exit 1; }
    trtllm-bench --model "$MODEL" throughput \
      --dataset "$dataset" \
      --concurrency "$MAX_BATCH" --max_batch_size "$MAX_BATCH" --max_num_tokens "$MAX_TOKENS" \
      --config "$dir/cfg.yaml" \
      --warmup 0 --streaming \
      --report_json "$dir/report.json" --iteration_log "$dir/iter.log" \
      2>&1 | tee "$dir/bench.log"
  fi
done

echo "Done. Compare with: python $(dirname "$0")/compare.py $OUT"
