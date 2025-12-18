#!/bin/bash
#SBATCH --job-name=sam_train  # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --gres=gpu:2              # Number of GPUs per node
#SBATCH --cpus-per-task=4          # CPU cores per task
#SBATCH --mem=64G                  # Memory per node
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
module load cudatoolkit/12.1
source ~/.bashrc
conda activate bias_sam_stable

export TMPDIR=/n/fs/vision-mix/rm4411/tmp_job_scratch
export OMP_NUM_THREADS=2

# ADD THIS LINE TO DISABLE THE JIT/VTUNE DEPENDENCY:
export LD_BIND_NOW=1
# Set MKL to use the standard GNU threading library
export MKL_THREADING_LAYER=GNU

MASTER_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$(host $MASTER_NODE | awk '{print $4}')
export MASTER_PORT=29506
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export NUM_GPUS_TO_USE=2
echo "Starting distributed training with MASTER_ADDR=$MASTER_ADDR and RANK=$RANK"

torchrun --nproc_per_node=$NUM_GPUS_TO_USE \
    --master_port=$MASTER_PORT \
    transform.py --dataset show --split val

#torchrun --nproc_per_node=$NUM_GPUS_TO_USE transform.py --dataset Janus --split train

#torchrun --nproc_per_node=$NUM_GPUS_TO_USE transform.py --dataset MMaDA --split train

#torchrun --nproc_per_node=$NUM_GPUS_TO_USE transform.py --dataset show --split train
