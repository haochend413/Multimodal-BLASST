#!/bin/bash
#SBATCH -A beqg-dtai-gh
#SBATCH -p ghx4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH -t 01:15:00
#SBATCH -J blasst-v2-weights
#SBATCH -o /work/nvme/beqg/zyang21/blasst/logs/%x-%j.out
# Items (1) and (2) of VLM_REGIONS_2.md: dense passes with the weight probe, Qwen (layouts A, B, C) then Llama.
source /work/nvme/beqg/zyang21/blasst/env.sh
python -W ignore vlm2_weights.py --model qwen && python -W ignore vlm2_weights.py --model llama
