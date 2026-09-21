#!/bin/bash
#SBATCH -A betw-dtai-gh
#SBATCH -p ghx4
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=32
#SBATCH --gpus=1 --mem=256G
#SBATCH --time=12:00:00
#SBATCH -J blasst-sweep
#SBATCH -o %x-%j.out

apptainer exec --nv -B $HOME -B /work/nvme/bfvr/$USER /u/hding4/trtllm.sif bash -c '
  export HF_HOME=/work/nvme/bfvr/'"$USER"'/hf_cache
  export PYTHONNOUSERSITE=1
  export PYTHONPATH=/work/nvme/bfvr/'"$USER"'/pylibs
  export TLLM_WORKER_USE_SINGLE_PROCESS=1
  export FLASHINFER_WORKSPACE_BASE=/work/nvme/bfvr/'"$USER"'/flashinfer_cache
  export CC=gcc
  export CXX=g++
  cd ~/Multimodal_BLASST/experiments/blasst
  rm -rf results
  ./run_sweep.sh && python compare.py results
'
sbatch sweep.sbatch
squeue -u $USER
tail -f blasst-sweep-*.out