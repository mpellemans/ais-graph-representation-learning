#!/bin/bash
#SBATCH --job-name=dgi_baselines
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x-%j.out

set -euo pipefail

# All five embedding baselines (raw_features, node2vec, graphsage, gae, hope)
# for a single seed. Markov and PortCity2Vec run via their own scripts
# (run_markov.sbatch, run_portcity2vec.sbatch); use run_baselines_multiseed.sbatch
# for the full 5-methods x 6-seeds array.
# Override seed with: sbatch --export=ALL,SEED=123 run_baselines.sbatch

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
CUDA=$(python -c "import torch; v=torch.version.cuda; print('cu'+''.join(v.split('.')[:2]))")
PYG_WHL="https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html"
pip install pyg-lib -f "${PYG_WHL}"
pip install torch-cluster --no-build-isolation -f "${PYG_WHL}"

# ---------------- run all baselines ----------------
SEED=${SEED:-42}
CONFIG=configs/experiments/maritime.yaml

for METHOD in raw_features node2vec graphsage gae hope; do
    echo "=== Running baseline: $METHOD (seed=$SEED) ==="
    python experiments/maritime/run_baseline.py \
        --config "$CONFIG" \
        --method "$METHOD" \
        --seed "$SEED"
done

echo "All baselines complete (seed=$SEED)!"
