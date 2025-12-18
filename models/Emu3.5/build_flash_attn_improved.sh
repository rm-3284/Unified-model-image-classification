#!/bin/bash
#SBATCH --job-name=flash_attention # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --cpus-per-task=80        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=400G
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%j/output.log
#SBATCH --error=logs/%j/error.log

module purge
module load anaconda3/2024.02
module load cudatoolkit/12.8
source ~/.bashrc
conda activate Emu3p5
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
# --- Handle temporary directory safely ---
if [ ! -d /scratch/$USER/tmp ]; then
    mkdir -p /scratch/$USER/tmp
fi
export PIP_CACHE_DIR="/n/fs/vision-mix/rm4411/pip_tmp_cache"
export TMPDIR="/n/fs/vision-mix/rm4411/pip_tmp_cache/tmp"
# --- Reduce compiler memory use ---
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9"  # targets A100/L40 only, saves time
export FORCE_CMAKE=1                       # ensures clean build
# --- Install ---
pip install -U pip wheel ninja
ninja --version
# FlashAttention build (may take 1–3 hours)
pip install flash-attn==2.8.3 --no-build-isolation
