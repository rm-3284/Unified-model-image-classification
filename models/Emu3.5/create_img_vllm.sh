#!/bin/bash
#SBATCH --job-name=emu_img_generation  # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:4               # Number of GPUs per node
#SBATCH --cpus-per-task=4          # CPU cores per task                  
#SBATCH --mem-per-gpu=50G                  
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
module load cudatoolkit/12.8
source ~/.bashrc
conda activate Emu3p5

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4

LANG=zh

python inference_vllm.py \
    --save_path "/n/fs/vision-mix/rm4411/multilang_imgs/Emu3.5/${LANG}/" --prompts "/n/fs/vision-mix/rm4411/multilang_prompts/cleaned_multilang_prompts/${LANG}.csv" \
    --start 0 \
    --cfg configs/example_config_t2i_1024.py \
    --tensor-parallel-size 4 --gpu-memory-utilization 0.8
