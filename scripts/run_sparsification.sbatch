#!/bin/bash
#SBATCH --job-name=dgi_sparsification
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x-%j.out

set -euo pipefail

# Graph sparsification ablation (seed 42): retains only the top-K outgoing
# edges per node by full-period journey count.
# K=10 (~avg degree 10), K=20 (~avg degree 20), baseline is K=inf (original, avg degree 69.5).
# Link prediction evaluation uses the original full-graph transitions throughout,
# so metrics are directly comparable across K values.
# Purpose: if DGI improves at lower K, neighborhood dilution on the dense graph
# is confirmed as the root cause.

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

# ---------------- run sparsification experiments ----------------
SEED=42

for K in 10 20; do
    CONFIG=configs/experiments/maritime_k${K}.yaml
    echo "=== Sparsification K=$K (seed=$SEED) ==="

    echo "--- Step 1: DGI training (K=$K) ---"
    python experiments/maritime/run_dgi_training.py \
        --config "$CONFIG" \
        --seed "$SEED"

    echo "--- Step 2: Link prediction (K=$K) ---"
    python experiments/maritime/run_link_prediction.py \
        --config "$CONFIG" \
        --seed "$SEED"

    echo "=== K=$K complete! ==="
done

echo "=== Sparsification experiments complete! ==="
