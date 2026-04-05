#!/bin/bash
#SBATCH --job-name=dsr_sweep
#SBATCH --output=/mnt/home/slee1/ceph/sacmeister/slurm_logs/dsr_sweep_%A_%a.out
#SBATCH --error=/mnt/home/slee1/ceph/sacmeister/slurm_logs/dsr_sweep_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --time=5-00:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --array=0-31

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID, Array Task: $SLURM_ARRAY_TASK_ID"

# Project directory
PROJECT_DIR=$SLURM_SUBMIT_DIR
cd $PROJECT_DIR
echo "Project directory: $PROJECT_DIR"

# Activate virtual environment
echo "Activating virtual environment: /mnt/home/slee1/venvs/sac"
source /mnt/home/slee1/venvs/sac/bin/activate

# ---------------------------------------------------------------------------
# Hyperparameter grid (2 values each, 2^5 = 32 combinations)
# ---------------------------------------------------------------------------
LR_VALUES=(3e-4 1e-3)
EPSILON_DECAY_VALUES=(0.999 0.9995)
BATCH_SIZE_VALUES=(64 128)
TARGET_UPDATE_FREQ_VALUES=(500 2000)
BUFFER_CAPACITY_VALUES=(50000 100000)

# Decode array task ID into grid indices
IDX=$SLURM_ARRAY_TASK_ID
i_lr=$((IDX % 2));              IDX=$((IDX / 2))
i_eps=$((IDX % 2));             IDX=$((IDX / 2))
i_bs=$((IDX % 2));              IDX=$((IDX / 2))
i_tuf=$((IDX % 2));             IDX=$((IDX / 2))
i_buf=$((IDX % 2))

LR=${LR_VALUES[$i_lr]}
EPSILON_DECAY=${EPSILON_DECAY_VALUES[$i_eps]}
BATCH_SIZE=${BATCH_SIZE_VALUES[$i_bs]}
TARGET_UPDATE_FREQ=${TARGET_UPDATE_FREQ_VALUES[$i_tuf]}
BUFFER_CAPACITY=${BUFFER_CAPACITY_VALUES[$i_buf]}

SWEEP_DIR="/mnt/home/slee1/ceph/sacmeister/sweep_${SLURM_ARRAY_JOB_ID}"
EXP_NAME="lr${LR}_ed${EPSILON_DECAY}_bs${BATCH_SIZE}_tuf${TARGET_UPDATE_FREQ}_buf${BUFFER_CAPACITY}"

echo "Hyperparameters:"
echo "  lr=$LR  epsilon_decay=$EPSILON_DECAY  batch_size=$BATCH_SIZE"
echo "  target_update_freq=$TARGET_UPDATE_FREQ  buffer_capacity=$BUFFER_CAPACITY"
echo "  experiment_name=$EXP_NAME"

python deep_srr.py \
    --lr $LR \
    --epsilon_decay $EPSILON_DECAY \
    --batch_size $BATCH_SIZE \
    --target_update_freq $TARGET_UPDATE_FREQ \
    --buffer_capacity $BUFFER_CAPACITY \
    --output_dir "$SWEEP_DIR" \
    --experiment_name "$EXP_NAME"

echo "Job finished: $(date)"
