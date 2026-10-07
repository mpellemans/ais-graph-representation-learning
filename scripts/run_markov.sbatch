#!/bin/bash
#SBATCH --job-name=dgi_markov
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/%x-%j.out

set -euo pipefail

module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0

python -m venv "$TMPDIR/dgi"
source "$TMPDIR/dgi/bin/activate"
pip install --upgrade pip
pip install torch==2.10.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

CONFIG=configs/experiments/maritime.yaml
SEED=42

for ORDER in 1 2 3 5 7 10; do
    echo "=== Markov n=$ORDER ==="
    python experiments/maritime/run_baseline.py \
        --config "$CONFIG" \
        --method "markov_n${ORDER}" \
        --seed "$SEED" \
        --max-len "$ORDER"
done

echo "=== Markov baselines complete! ==="
echo "Aggregating results..."
python experiments/maritime/aggregate_results.py results/maritime/embeddings
