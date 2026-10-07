#!/bin/bash
#SBATCH --job-name=dgi_ecdgcn
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x-%j.out

set -euo pipefail

# ECDGCN encoder ablation (seed 42): edge-conditioned message passing with
# per-parameter-group gradient clipping. The edge_mlp gets its own clip norm
# (edge_grad_clip_norm=5.0) separate from fc_in/fc_out/fc_self
# (grad_clip_norm=1.0), preventing edge gradients from consuming the entire
# clipping budget over 121K edges.

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

# ---------------- run ECDGCN experiment ----------------
CONFIG=configs/experiments/maritime_ecdgcn.yaml
SEED=42

echo "=== ECDGCN with per-group gradient clipping (seed=$SEED) ==="

echo "--- Step 1: DGI training (ECDGCN) ---"
python experiments/maritime/run_dgi_training.py \
    --config "$CONFIG" \
    --seed "$SEED"

echo "--- Step 2: Link prediction (ECDGCN) ---"
python experiments/maritime/run_link_prediction.py \
    --config "$CONFIG" \
    --seed "$SEED"

echo "=== ECDGCN experiment complete! ==="
