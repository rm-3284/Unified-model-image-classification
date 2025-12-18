#!/bin/bash
#SBATCH --job-name=qwen  # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:1               # Number of GPUs per node
#SBATCH --cpus-per-task=8          # CPU cores per task
#SBATCH --mem=32G                  # Memory per node
#SBATCH --time=50:00:00            # Time limit (1 hour)
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=rm4411@princeton.edu
#SBATCH --output=logs/%j/output.log
#SBATCH --error=logs/%j/error.log
#SBATCH --partition=all

module purge
module load anaconda3/2024.02
module load cudatoolkit/12.4
source ~/.bashrc
conda activate qwen_env

export MASTER_PORT=$(($SLURM_JOB_ID % 30000 + 20000))
MASTER_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$(host $MASTER_NODE | awk '{print $4}')
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export NUM_GPUS_TO_USE=1
echo "Starting distributed training with MASTER_ADDR=$MASTER_ADDR and RANK=$RANK"

export greedy='false'
export top_p=0.8
export top_k=20
export temperature=0.7
export repetition_penalty=1.0
export presence_penalty=1.5
export out_seq_length=16384

torchrun --nproc_per_node=$NUM_GPUS_TO_USE --master_port=$MASTER_PORT \
    image_description_by_domain.py --image_dir "/n/fs/vision-mix/rm4411/OOD_DOMAIN" --save_dir "/n/fs/vision-mix/rm4411/Qwen3"
