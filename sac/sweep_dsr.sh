#!/bin/bash
#SBATCH --job-name=dsr_sweep
#SBATCH --output=logs/dsr_%A_%a.out
#SBATCH --error=logs/dsr_%A_%a.err
#SBATCH --array=0-119
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Hyperparameter grid
LEARNING_RATES=(0.001 0.0003 0.0001)
FEATURE_DIMS=(64 128 256)
RECONSTRUCTION=(0 1)  # 0=False, 1=True
BATCH_SIZES=(64 128)
SEEDS=(0 1 2 3 4)

# Total combinations: 3 * 3 * 2 * 2 * 5 = 180
# For smaller sweep: 3 * 2 * 2 * 2 * 5 = 120

num_lr=${#LEARNING_RATES[@]}
num_fd=${#FEATURE_DIMS[@]}
num_recon=${#RECONSTRUCTION[@]}
num_bs=${#BATCH_SIZES[@]}
num_seeds=${#SEEDS[@]}

# Decode array index
idx=$SLURM_ARRAY_TASK_ID
seed_idx=$((idx % num_seeds))
idx=$((idx / num_seeds))
bs_idx=$((idx % num_bs))
idx=$((idx / num_bs))
recon_idx=$((idx % num_recon))
idx=$((idx / num_recon))
fd_idx=$((idx % num_fd))
idx=$((idx / num_fd))
lr_idx=$((idx % num_lr))

LR=${LEARNING_RATES[$lr_idx]}
FD=${FEATURE_DIMS[$fd_idx]}
RECON=${RECONSTRUCTION[$recon_idx]}
BS=${BATCH_SIZES[$bs_idx]}
SEED=${SEEDS[$seed_idx]}

# Fixed hyperparameters
DISCOUNT=0.99
EPS=1.0
EPS_DECAY=0.995
TUF=100
RBS=10000
BURNIN=1000
RECON_COEF=0.1
NUM_EPISODES=2000

# Build reconstruction flag
if [ "$RECON" -eq 1 ]; then
    RECON_FLAG="-recon"
    RECON_STR="recon"
else
    RECON_FLAG=""
    RECON_STR="norecon"
fi

# Results directory
RESULTS_DIR="results/dsr_sweep/lr${LR}_fd${FD}_${RECON_STR}_bs${BS}_seed${SEED}"

echo "Running DSR with:"
echo "  Learning rate: $LR"
echo "  Feature dim: $FD"
echo "  Reconstruction: $RECON"
echo "  Batch size: $BS"
echo "  Seed: $SEED"
echo "  Results dir: $RESULTS_DIR"

# Activate conda environment (adjust as needed)
# source activate sac

python -m sac.run \
    -m dsr \
    -seed $SEED \
    -lr $LR \
    -gamma $DISCOUNT \
    -eps $EPS \
    -eps_decay $EPS_DECAY \
    -bs $BS \
    -tuf $TUF \
    -rbs $RBS \
    -burnin $BURNIN \
    -fd $FD \
    $RECON_FLAG \
    -rc $RECON_COEF \
    -num_ep $NUM_EPISODES \
    -conv \
    -rep pixel \
    -map meister_trimmed.txt \
    -map_yaml meister_trimmed.yaml \
    -test_map_yaml test_meister_trimmed.yaml \
    -abs_results $RESULTS_DIR

echo "Job completed"
