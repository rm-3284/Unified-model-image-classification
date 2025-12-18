#!/bin/bash
#SBATCH --job-name=resize_img  # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:0              # Number of GPUs per node
#SBATCH --cpus-per-task=4          # CPU cores per task
#SBATCH --mem=32G                  # Memory per node
#SBATCH --time=10:00:00            # Time limit (1 hour)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%j/output.log
#SBATCH --error=logs/%j/error.log
#SBATCH --partition=all

# the first argument is the prompt
module purge
module load anaconda3/2024.02
source ~/.bashrc
conda activate understand_bias

MODEL=gpt

INPUT_DIR="/n/fs/vision-mix/rm4411/multilingual_images_gpt"
OUTPUT_DIR="/n/fs/vision-mix/rm4411/resized-multilingual-images/${MODEL}"

python resize_img.py "${INPUT_DIR}" "${OUTPUT_DIR}" --model gpt
