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
module load cudatoolkit/12.1
source ~/.bashrc
conda activate qwen_env

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

if [ ! -d /scratch/$USER/tmp ]; then
    mkdir -p /scratch/$USER/tmp
fi
export PIP_CACHE_DIR="/n/fs/vision-mix/rm4411/pip_tmp_cache"
export TMPDIR="/n/fs/vision-mix/rm4411/pip_tmp_cache/tmp"

export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9"  
export FORCE_CMAKE=1                       

export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=0"

#pip install -U pip wheel ninja
#ninja --version

pip install flash-attn==2.7.3 --no-build-isolation
