#!/bin/bash
#SBATCH --job-name=raw_no_conn
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/%x-%A.out

set -euo pipefail

# Feature-removal robustness for the raw-features baseline. Runs the raw-features
# baseline with the degree/connectivity/activity-count node features removed
# (exclude_connectivity_features=true, geography kept) and re-evaluates all three
# next-port-prediction framings (dot uses the fair cosine + self-excluded
# setting). Compare against the full raw-features row of the main results table.
#
# Single seed (42) is sufficient for the robustness check: the raw embedding is
# deterministic; only the MLP/LSTM predictors carry seed variance, which is tiny.
# Submit: sbatch scripts/run_raw_features_no_conn.sbatch
#
# For the full 6-seed version (to match the published n=6 raw row): run seed 42
# first (this script) so the _no_conn cache is built, then launch the other seeds,
# e.g. add `#SBATCH --array=0-4`, set SEED from an array, and remove this line.

SEED=42

echo "=== raw_features (no connectivity), seed=${SEED} ==="

# ---------------- software stack ----------------
module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0

# ---------------- throw-away venv in $TMPDIR ----------------
python -m venv "$TMPDIR/dgi"
source "$TMPDIR/dgi/bin/activate"
pip install --upgrade pip
pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
# Note: raw_features needs no torch-cluster / pyg-lib (those are Node2Vec-only).

# ---------------- run ----------------
CONFIG=configs/experiments/maritime_no_connectivity.yaml

echo "Running: python experiments/maritime/run_baseline.py --config $CONFIG --method raw_features --seed $SEED"
python experiments/maritime/run_baseline.py \
    --config "$CONFIG" \
    --method raw_features \
    --seed "$SEED"

echo "Done. Results in results/maritime/embeddings/raw_features_no_conn_seed${SEED}/"
