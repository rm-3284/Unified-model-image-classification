#!/bin/bash
#SBATCH --job-name=classifier_train  # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:4              # Number of GPUs per node
#SBATCH --cpus-per-task=4          # CPU cores per task
#SBATCH --mem=128G                  # Memory per node
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


export LD_PRELOAD="/n/fs/vision-mix/rm4411/conda_envs/understand_bias/lib/libstdc++.so.6"

MASTER_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$(host $MASTER_NODE | awk '{print $4}')
export MASTER_PORT=$(($SLURM_JOB_ID % 30000 + 20000))
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export NUM_GPUS_TO_USE=4
echo "Starting distributed training with MASTER_ADDR=$MASTER_ADDR and RANK=$RANK"

OUTPUT_ROOT="/n/fs/vision-mix/rm4411/understand_bias/checkpoints"
TENSORBOARD_ROOT="/n/fs/vision-mix/rm4411/understand_bias/tensorboard-log"

TRAIN_DIR="train"
VLM_NAME="gpt"

torchrun --nproc_per_node=$NUM_GPUS_TO_USE \
    --master_port=$MASTER_PORT \
    main.py \
    --data_names "en,es,ja,tr,zh" --train_dir_name "$TRAIN_DIR" \
    --nb_classes 5 --output_dir "${OUTPUT_ROOT}/${VLM_NAME}" \
    --log_dir "${TENSORBOARD_ROOT}/${VLM_NAME}" \
    --epochs 200 --lr 1e-3 --warmup_epochs 2 --batch_size 64 \
    --vlm_name $VLM_NAME
