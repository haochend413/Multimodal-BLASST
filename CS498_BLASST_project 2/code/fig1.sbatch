#!/bin/bash
#SBATCH -A beqg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160g
#SBATCH --time=01:00:00
#SBATCH --job-name=blasst-fig1
#SBATCH --output=/work/nvme/beqg/zyang21/blasst/logs/%x-%j.out
source /work/nvme/beqg/zyang21/blasst/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -W ignore fig1_sweep.py --lengths 4k 8k 16k 32k --per_task 6 --per_task_32k 3 2>&1 | grep -v -E "torchao|Loading checkpoint"
