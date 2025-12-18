#!/bin/bash
#SBATCH --job-name=bagel_img_generation  # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:1               # Number of GPUs per node
#SBATCH --cpus-per-task=4          # CPU cores per task
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
source ~/.bashrc
conda activate bagel

LANG=zh

python app.py --model_path "/n/fs/vision-mix/rm4411/models/BAGEL-7B-MoT" \
--mode 2 --prompt "/n/fs/vision-mix/rm4411/multilang_prompts/cleaned_multilang_prompts/${LANG}.csv" --save_dir "/n/fs/vision-mix/rm4411/multilang_imgs/BAGEL/${LANG}" \
--image-ratio "1:1"
