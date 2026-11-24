#!/usr/bin/env bash
# Submit from CS498_BLASST_project 2 so the job can find code/attention_study.py.
set -euo pipefail

manifest="${1:?provide the VQA manifest}"
output_dir="${2:?provide a shard output directory}"
shard_count="${3:?provide the shard count}"
shard_index="${SLURM_ARRAY_TASK_ID:?run as a Slurm array task}"
python code/attention_study.py \
  --manifest "$manifest" \
  --vqa-shards "$shard_count" --vqa-shard-index "$shard_index" \
  --lambda 0.1 --decode-tokens 16 --tv-vv-sweep 0.5 --vqa-accuracy \
  --output "$output_dir/shard-$shard_index.json"
