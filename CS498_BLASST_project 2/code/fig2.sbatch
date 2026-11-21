#!/bin/bash
# usage: sbatch --job-name=blasst-fig2-<len> --time=HH:MM:SS fig2.sbatch <length> <config> [<config> ...]
#SBATCH -A beqg-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --time=02:00:00
#SBATCH --output=/work/nvme/beqg/zyang21/blasst/logs/%x-%j.out
source /work/nvme/beqg/zyang21/blasst/env.sh
LEN=$1; shift
python -W ignore fig2_ruler.py --length $LEN --tasks niah_multikey_1 vt fwe --n 50 --fit log_fit --configs "$@" \
    --out results/fig2_ruler_${LEN}_$(echo "$@" | tr ' ' '_').csv 2>&1 | grep -v -E "torchao|Loading checkpoint"
