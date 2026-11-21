#!/bin/bash
#SBATCH -A beqg-dtai-gh
#SBATCH -p ghx4
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH -t 02:00:00
#SBATCH -J blasst-v2-pilot
#SBATCH -o /work/nvme/beqg/zyang21/blasst/logs/%x-%j.out
# Item (3) of VLM_REGIONS_2.md: Video-MME accuracy pilot, dense vs one lambda vs T->V protected.
source /work/nvme/beqg/zyang21/blasst/env.sh
python -W ignore vlm2_pilot.py
