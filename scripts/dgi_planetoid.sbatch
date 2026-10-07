#!/bin/bash
#SBATCH --job-name=dgi_exp
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/%x-%j.out

set -euo pipefail

# ---------------- software stack --------------#
module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0

# ---------------- create throw-away venv in $TMPDIR -------------#
python -m venv "$TMPDIR/dgi"
source "$TMPDIR/dgi/bin/activate"
pip install --upgrade pip
pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# ---------------- run experiment -------------#
# Default: run cora experiment
# Override with: sbatch --export=ALL,CONFIG=configs/experiments/pubmed.yaml dgi_planetoid.sbatch
CONFIG=${CONFIG:-configs/experiments/cora.yaml}

python experiments/run_planetoid.py --config "$CONFIG"
