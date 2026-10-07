#!/bin/bash
#SBATCH --job-name=dgi_multiseed
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x-%A-%a.out
#SBATCH --array=0-4

set -euo pipefail

# Array layout: 5 seeds (0 1 2 3 4)
# Submit: sbatch scripts/run_dgi_multiseed.sbatch

SEEDS=(0 1 2 3 4)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Array task ${SLURM_ARRAY_TASK_ID}: DGI DirectedGCN, seed=${SEED} ==="

# ---------------- software stack ----------------
module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0

# ---------------- create throw-away venv in $TMPDIR ----------------
python -m venv "$TMPDIR/dgi"
source "$TMPDIR/dgi/bin/activate"
pip install --upgrade pip
pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# ---------------- run DGI pipeline ----------------
CONFIG=configs/experiments/maritime.yaml

# Step 1: Train DGI, save embeddings + t-SNE
python experiments/maritime/run_dgi_training.py \
    --config "$CONFIG" \
    --seed "$SEED"

# Step 2: Link prediction (primary evaluation)
python experiments/maritime/run_link_prediction.py \
    --config "$CONFIG" \
    --seed "$SEED"

# Step 3: Clustering analysis
python experiments/maritime/run_clustering.py \
    --config "$CONFIG" \
    --seed "$SEED"

echo "Done: DGI DirectedGCN, seed=${SEED}"
