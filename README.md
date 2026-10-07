# Unsupervised Graph Representation Learning from AIS Maritime Traffic

Code release accompanying the paper *"Unsupervised Graph Representation Learning
from AIS Maritime Traffic"*.

The paper investigates whether unsupervised graph representation learning can
produce meaningful and operationally useful port embeddings from a global,
directed, attributed port-visit network built from 6.15 million AIS-derived
vessel journeys. It evaluates Deep Graph Infomax (DGI) with a DirectedGCN
encoder against Raw Features, Node2Vec, GraphSAGE, GAE, HOPE, PortCity2Vec, and
higher-order Markov chains on structural analysis tasks and next-port
prediction, alongside a replication of DGI on the standard Planetoid
benchmarks (Cora, Citeseer, Pubmed).

This repository provides the full implementation for transparency: the model,
preprocessing, baselines, evaluation protocols, and experiment entry points
that produced results reported in the paper. The licensed AIS data cannot
be redistributed (see Data Availability), so the published numbers cannot be
regenerated from this repository; a schema-faithful synthetic sample
makes every pipeline runnable end-to-end.

## Repository layout

```
src/                  Core library
  models/             DGI model (encoder selected by config)
  layers/             GCN, DirectedGCN, EdgeConditionedDGCN, discriminator, readout
  preprocessing/      Feature/adjacency preprocessing + maritime pipeline
  data/               Planetoid and Maritime dataset loaders (with caching)
  training/           DGITrainer, early stopping
  evaluation/         Linear probe, link prediction (dot-product / MLP / LSTM)
  baselines/          RawFeatures, Node2Vec, GraphSAGE, GAE, HOPE, PortCity2Vec, Markov
  config/             Dataclass configs + YAML loading
  utils/              Visualizations, legacy loaders
experiments/
  run_planetoid.py    Planetoid benchmark entry point
  maritime/           Maritime entry points (see mapping table below)
configs/              base.yaml + per-experiment configs
scripts/              Synthetic-data generator, EDA, figure scripts, SLURM jobs
tests/                Test suite (fully synthetic data, no licensed files needed)
data/planetoid/       Public Planetoid benchmark files (included)
docs/data_schema.md   Schema of the (withheld) maritime parquet files
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.12 is recommended (experiments used Python 3.12.3, PyTorch
2.10.0, CUDA 12.6 on the Snellius national cluster; everything also runs on
CPU). Two extra dependencies are documented in `requirements.txt` because they
need wheels matched to your torch build: `pyg-lib`/`torch-cluster` (the
random-walk backend required only for the Node2Vec baseline; see
`scripts/run_node2vec.sbatch` for the install recipe) and `basemap`/`pyproj`
(only for the port-network figure).

## Data availability

- **Maritime AIS data (withheld).** The port-visit network is derived from AIS
  vessel-tracking data licensed from S&P Global (Sea-web) and cannot be
  redistributed. This includes the raw parquet files, all processed caches,
  learned embeddings, and model checkpoints. `docs/data_schema.md` documents
  the exact schemas so license holders can reconstruct compatible inputs.
- **Planetoid benchmarks (included).** The standard public Cora, Citeseer, and
  Pubmed files ship in `data/planetoid/`, so the replication study is fully
  reproducible from this repository.
- **PortCity2Vec comparison data (not redistributed).** The original
  PortCity2Vec datasets by Cai et al. (2026) are publicly available under
  CC BY 4.0 at https://doi.org/10.6084/m9.figshare.29222318 . Our PortCity2Vec
  baseline (`src/baselines/portcity2vec.py`) is a re-implementation trained on
  our own graph and does not require those files.
- **Synthetic sample (generated).** `python scripts/make_synthetic_data.py`
  writes `data/synthetic/{nodes,edges,journeys}.parquet` with the documented
  schemas, so every maritime entry point runs end-to-end without licensed
  data. Numbers produced on synthetic data are *not* the paper's numbers.

## Quickstart (no licensed data required)

```bash
# Replicate DGI on Cora (public data, included)
python experiments/run_planetoid.py --config configs/experiments/cora.yaml

# Maritime pipeline on the synthetic sample
python scripts/make_synthetic_data.py
python experiments/maritime/run_dgi_training.py   --config configs/experiments/synthetic.yaml --seed 42
python experiments/maritime/run_link_prediction.py --config configs/experiments/synthetic.yaml --seed 42
python experiments/maritime/run_clustering.py      --config configs/experiments/synthetic.yaml --seed 42
```

## Cluster execution

The `scripts/*.sbatch` files are the SLURM jobs used on the Snellius national
cluster (single-seed runs, multi-seed arrays, baselines, ablations). They
document the exact software stack of the paper runs (`module load CUDA/12.6.0
Python/3.12.3`, `torch==2.10.0+cu126`) and can be adapted to other clusters.

## Tests

The test suite uses only synthetic data (no licensed files needed). There is no
pytest dependency; run the files directly:

```bash
python tests/test_maritime_preprocessing.py
python tests/test_link_prediction.py
python tests/test_baselines.py
python tests/test_ecdgcn.py
```

## License and citation

The code is released under the MIT License (see `LICENSE`). If you use it,
please cite the paper (see `CITATION.cff`):

> Pellemans, M. (2026). Unsupervised Graph Representation Learning from AIS
> Maritime Traffic. In Press at Maritime Transport Research.
