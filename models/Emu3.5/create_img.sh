#!/bin/bash
#SBATCH --job-name=emu_img_generation  # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:2               # Number of GPUs per node
#SBATCH --cpus-per-task=8          # CPU cores per task
#SBATCH --mem=32G                  # Memory per node
#SBATCH --time=50:00:00            # Time limit (1 hour)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%j/output.log
#SBATCH --error=logs/%j/error.log
#SBATCH --partition=all

# the first argument is the prompt
module purge
module load anaconda3/2024.02
module load cudatoolkit/12.1
source ~/.bashrc
conda activate emu3.5

python inference.py --cfg configs/example_config_t2i_512.py
