#!/bin/bash
#SBATCH --job-name=lstm_sensitivity
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x-%j.out

set -euo pipefail

# ---------------- software stack --------------#
module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0

# ---------------- create throw-away venv in $TMPDIR (reuse if exists) -------------#
if [ ! -d "$TMPDIR/dgi" ]; then
    python -m venv "$TMPDIR/dgi"
    source "$TMPDIR/dgi/bin/activate"
    pip install --upgrade pip
    pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
    pip install -r requirements.txt
else
    source "$TMPDIR/dgi/bin/activate"
fi

# ---------------- run LSTM sensitivity sweep -------------#
# Override seed/encoder with:
#   sbatch --export=ALL,SEED=123,ENCODER=dgcn run_lstm_sensitivity.sbatch
SEED=${SEED:-42}
ENCODER=${ENCODER:-dgcn}
CONFIG=configs/experiments/maritime.yaml
EMB_PATH="results/maritime/embeddings/${ENCODER}_seed${SEED}/embeddings.npy"

python experiments/maritime/run_lstm_sensitivity.py \
    --config "$CONFIG" \
    --seed "$SEED" \
    --embeddings "$EMB_PATH" \
    --max-lens "1,2,3,5,7,10,15" \
    --n-runs 3

echo "LSTM sensitivity sweep complete (encoder=${ENCODER}, seed=${SEED})!"
