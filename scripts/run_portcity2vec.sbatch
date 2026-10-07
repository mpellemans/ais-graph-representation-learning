#!/bin/bash
#SBATCH --job-name=portcity2vec
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/%x-%A-%a.out
#SBATCH --array=0-5

set -euo pipefail

# PortCity2Vec baseline (Cai et al., 2026): weighted biased random walk
# (p=q=0.5) + skip-gram on the same filtered maritime graph as the other
# baselines. One array element per seed.
#
# Array layout: 6 seeds = 6 jobs
#   SEED = SEEDS[SLURM_ARRAY_TASK_ID]
#
# Submit: sbatch scripts/run_portcity2vec.sbatch
# Single seed: sbatch --array=5 scripts/run_portcity2vec.sbatch   # seed 42

SEEDS=(0 1 2 3 4 42)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== Array task ${SLURM_ARRAY_TASK_ID}: method=portcity2vec, seed=${SEED} ==="

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
# PortCity2Vec needs only torch + numpy + pandas/pyarrow (no torch-cluster).

# ---------------- run baseline ----------------
CONFIG=configs/experiments/maritime.yaml

echo "Running: python experiments/maritime/run_baseline.py --config $CONFIG --method portcity2vec --seed $SEED"
python experiments/maritime/run_baseline.py \
    --config "$CONFIG" \
    --method portcity2vec \
    --seed "$SEED"

echo "Done: method=portcity2vec, seed=${SEED}"
