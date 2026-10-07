#!/bin/bash
#SBATCH --job-name=dgi_geo_ablation
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x-%j.out

set -euo pipefail

# Geographic feature ablation (maritime_no_geo.yaml, seed 42)
# Removes geohash + country + continent + sub_region one-hots (~300 of 441 dims)
# Purpose: remove directly encoded geography from the input features; test
#          whether DGI learns geographic proximity from graph structure alone.

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

# ---------------- run geo ablation ----------------
CONFIG=configs/experiments/maritime_no_geo.yaml
SEED=42

echo "=== Geographic Ablation (seed=$SEED) ==="

# Step 1: Train DGI on geo-stripped features
echo "--- Step 1: DGI training ---"
python experiments/maritime/run_dgi_training.py \
    --config "$CONFIG" \
    --seed "$SEED"

# Step 2: Sub-region classification (sub_region absent from input features)
echo "--- Step 2: Classification (sub-region) ---"
python experiments/maritime/run_classification.py \
    --config "$CONFIG" \
    --seed "$SEED"

# Step 3: Link prediction (confirm MLP ~37% holds without geo features)
echo "--- Step 3: Link prediction ---"
python experiments/maritime/run_link_prediction.py \
    --config "$CONFIG" \
    --seed "$SEED"

# Step 4: Raw features baseline on geo-stripped inputs
echo "--- Step 4: Raw features baseline (no geo) ---"
python experiments/maritime/run_baseline.py \
    --config "$CONFIG" \
    --method raw_features \
    --seed "$SEED"

echo "=== Geo ablation complete! ==="
