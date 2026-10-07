#!/bin/bash
#SBATCH --job-name=baselines_multiseed
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x-%A-%a.out
#SBATCH --array=0-29

set -euo pipefail

# All five embedding baselines over all six paper seeds (n=6). Markov and
# PortCity2Vec run via their own scripts (run_markov.sbatch,
# run_portcity2vec.sbatch).
#
# Array layout: 5 methods x 6 seeds = 30 jobs
#   ARRAY_ID = seed_idx * 5 + method_idx
#   method_idx = ARRAY_ID % 5  (0=raw_features, 1=node2vec, 2=graphsage, 3=gae, 4=hope)
#   seed_idx   = ARRAY_ID / 5  (0-5 → seeds 0 1 2 3 4 42)
#
# Submit: sbatch scripts/run_baselines_multiseed.sbatch
# To run a single array element: sbatch --array=3 scripts/run_baselines_multiseed.sbatch

METHODS=(raw_features node2vec graphsage gae hope)
SEEDS=(0 1 2 3 4 42)

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))

METHOD=${METHODS[$METHOD_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== Array task ${SLURM_ARRAY_TASK_ID}: method=${METHOD}, seed=${SEED} ==="

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

# Node2Vec needs a random-walk backend (pyg-lib / torch-cluster); install both
# from the PyG pre-built wheel index (strip the +cuXXX suffix from
# torch.__version__ to avoid a double-suffix URL).
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_TAG=$(python -c "import torch; v=torch.version.cuda; print('cu'+''.join(v.split('.')[:2]))")
PYG_WHL="https://data.pyg.org/whl/torch-${TORCH}+${CUDA_TAG}.html"
pip install pyg-lib -f "${PYG_WHL}"
pip install torch-cluster --no-build-isolation -f "${PYG_WHL}"

# ---------------- run baseline ----------------
CONFIG=configs/experiments/maritime.yaml

echo "Running: python experiments/maritime/run_baseline.py --config $CONFIG --method $METHOD --seed $SEED"
python experiments/maritime/run_baseline.py \
    --config "$CONFIG" \
    --method "$METHOD" \
    --seed "$SEED"

echo "Done: method=${METHOD}, seed=${SEED}"
