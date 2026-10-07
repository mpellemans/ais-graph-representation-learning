#!/bin/bash
#SBATCH --job-name=dgi_benchmarks
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
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

# ---------------- run all benchmark experiments -------------#
echo "Running Cora experiment..."
python experiments/run_planetoid.py --config configs/experiments/cora.yaml

echo "Running Citeseer experiment..."
python experiments/run_planetoid.py --config configs/experiments/citeseer.yaml

echo "Running Pubmed experiment..."
python experiments/run_planetoid.py --config configs/experiments/pubmed.yaml

echo "All benchmark experiments complete!"
