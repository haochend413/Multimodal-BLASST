# source this in every shell / Slurm job:  source /work/nvme/beqg/zyang21/blasst/env.sh
conda deactivate 2>/dev/null
module reset >/dev/null 2>&1            # 'module purge' breaks the PyTorch module (needs PrgEnv)
module load python/miniforge3_pytorch/2.11.0 >/dev/null 2>&1
export W=/work/nvme/beqg/zyang21
export HF_HOME=$W/hf PIP_CACHE_DIR=$W/pip-cache TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS="ignore::UserWarning"
source $W/blasst/venv/bin/activate
cd $W/blasst
