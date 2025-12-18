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
source ~/.bashrc
conda activate bagel
pip install ninja

ninja --version
echo $?

pip install flash-attn --no-build-isolation
