#!/bin/bash
#SBATCH --job-name=dgi_node2vec
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x-%A-%a.out
#SBATCH --array=0-5

set -euo pipefail

# Node2Vec baseline for all six seeds (standard DeepWalk parameters, p=q=1),
# trained for 100 epochs to match the training budget of the other baselines
# (GraphSAGE 100 / GAE 200). Produces the Node2Vec row of the main results
# table across all three framings; the dot-product framing uses the fair
# setting (cosine + self-exclusion) via run_baseline.py.
#
# Array layout: 6 jobs, one per seed.
#   Submit: sbatch scripts/run_node2vec.sbatch
#   Single seed: sbatch --array=5 scripts/run_node2vec.sbatch   (seed 42)

SEEDS=(0 1 2 3 4 42)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
EPOCHS=100

echo "=== Node2Vec: seed=${SEED}, epochs=${EPOCHS} ==="

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

# Node2Vec needs a random-walk backend: torch_geometric requires pyg-lib (or
# torch-cluster on older versions) for its Node2Vec loader. Install BOTH from
# the PyG pre-built wheel index. Strip the +cuXXX suffix from torch.__version__
# to avoid a double-suffix URL.
TORCH=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_TAG=$(python -c "import torch; v=torch.version.cuda; print('cu'+''.join(v.split('.')[:2]))")
PYG_WHL="https://data.pyg.org/whl/torch-${TORCH}+${CUDA_TAG}.html"
echo "PyG wheel index: ${PYG_WHL}"
pip install pyg-lib -f "${PYG_WHL}"
pip install torch-cluster --no-build-isolation -f "${PYG_WHL}"

# Sanity check: print versions so any remaining failure is easy to diagnose.
python -c "import torch, torch_geometric as g; print('torch', torch.__version__, '| PyG', g.__version__)"
python -c "import pyg_lib; print('pyg_lib', pyg_lib.__version__)" || echo "WARN: pyg_lib import failed"
python -c "import torch_cluster; print('torch_cluster import OK')" || echo "WARN: torch_cluster import failed"

# ---------------- run ----------------
CONFIG=configs/experiments/maritime.yaml

echo "Running: python experiments/maritime/run_baseline.py --config $CONFIG --method node2vec --seed $SEED --epochs $EPOCHS"
python experiments/maritime/run_baseline.py \
    --config "$CONFIG" \
    --method node2vec \
    --seed "$SEED" \
    --epochs "$EPOCHS"

echo "Done: node2vec seed=${SEED}, epochs=${EPOCHS}"
