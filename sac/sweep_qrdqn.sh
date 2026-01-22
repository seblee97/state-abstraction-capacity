#!/bin/bash
#SBATCH --job-name=qrdqn_sweep
#SBATCH --output=logs/qrdqn_%A_%a.out
#SBATCH --error=logs/qrdqn_%A_%a.err
#SBATCH --array=0-89
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Create logs directory if it doesn't exist
mkdir -p logs

# Hyperparameter grid
LEARNING_RATES=(0.001 0.0003 0.0001)
NUM_QUANTILES=(50 100 200)
KAPPAS=(0.5 1.0 2.0)
BATCH_SIZES=(64 128)
SEEDS=(0 1 2 3 4)

# Calculate indices from SLURM_ARRAY_TASK_ID
# Total combinations: 3 * 3 * 3 * 2 * 5 = 270, but we'll do 90 (without seed variation first)
# Layout: lr(3) x nq(3) x kappa(3) x bs(2) = 54 configs, x 5 seeds = 270
# For a smaller sweep: lr(3) x nq(3) x bs(2) x seeds(5) = 90

num_lr=${#LEARNING_RATES[@]}
num_nq=${#NUM_QUANTILES[@]}
num_bs=${#BATCH_SIZES[@]}
num_seeds=${#SEEDS[@]}

# Decode array index
idx=$SLURM_ARRAY_TASK_ID
seed_idx=$((idx % num_seeds))
idx=$((idx / num_seeds))
bs_idx=$((idx % num_bs))
idx=$((idx / num_bs))
nq_idx=$((idx % num_nq))
idx=$((idx / num_nq))
lr_idx=$((idx % num_lr))

LR=${LEARNING_RATES[$lr_idx]}
NQ=${NUM_QUANTILES[$nq_idx]}
BS=${BATCH_SIZES[$bs_idx]}
SEED=${SEEDS[$seed_idx]}

# Fixed hyperparameters
KAPPA=1.0
DISCOUNT=0.99
EPS=1.0
EPS_DECAY=0.995
TUF=100
RBS=10000
BURNIN=1000
NUM_EPISODES=2000

# Results directory
RESULTS_DIR="results/qrdqn_sweep/lr${LR}_nq${NQ}_bs${BS}_seed${SEED}"

echo "Running QRDQN with:"
echo "  Learning rate: $LR"
echo "  Num quantiles: $NQ"
echo "  Batch size: $BS"
echo "  Seed: $SEED"
echo "  Results dir: $RESULTS_DIR"

# Activate conda environment (adjust as needed)
# source activate sac

python -m sac.run \
    -m qrdqn \
    -seed $SEED \
    -lr $LR \
    -gamma $DISCOUNT \
    -eps $EPS \
    -eps_decay $EPS_DECAY \
    -bs $BS \
    -tuf $TUF \
    -rbs $RBS \
    -burnin $BURNIN \
    -nq $NQ \
    -kappa $KAPPA \
    -num_ep $NUM_EPISODES \
    -conv \
    -rep pixel \
    -map meister_trimmed.txt \
    -map_yaml meister_trimmed.yaml \
    -test_map_yaml test_meister_trimmed.yaml \
    -abs_results $RESULTS_DIR

echo "Job completed"
