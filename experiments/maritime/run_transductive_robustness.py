"""Transductive full-window robustness check for the PortCity2Vec dot-product result.

The reported PortCity2Vec embedding is trained on full-period edge weights
(total_journeys over Feb 2022 - Aug 2023), which include the evaluation window
(transductive setting). This script retrains PortCity2Vec with TRAIN-PERIOD-ONLY
edge weights (same filtered topology, edges reweighted by train transition
counts) and re-evaluates the dot-product framing. If dot H@1 is essentially
unchanged, the ranking signal is robust to restricting the weights to the
training period and does not depend on information from the evaluation window.

Run:
    python experiments/maritime/run_transductive_robustness.py \
        --config configs/experiments/maritime.yaml --seed 42
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

from common import set_seed, get_device, load_maritime_dataset
from src.config import load_config
from src.baselines.portcity2vec import PortCity2VecBaseline
from src.evaluation.link_prediction import (
    DotProductLinkPredictor,
    _build_warm_port_mapping,
)


class TrainPeriodPortCity2Vec(PortCity2VecBaseline):
    """PortCity2Vec variant whose edge weights come from TRAIN transitions only."""

    def __init__(self, train_transitions, **kwargs):
        super().__init__(**kwargs)
        self._train_transitions = train_transitions

    def _edge_weights(self, dataset) -> np.ndarray:
        edge_index = dataset.edge_index
        # train-period directed counts
        out_train = defaultdict(lambda: defaultdict(float))
        for t in self._train_transitions:
            out_train[int(t[0])][int(t[1])] += 1.0
        E = edge_index.shape[1]
        weights = np.empty(E, dtype=np.float64)
        n_zero = 0
        for e in range(E):
            s = int(edge_index[0, e])
            d = int(edge_index[1, e])
            w = out_train.get(s, {}).get(d, 0.0)
            if w == 0.0:
                n_zero += 1
            weights[e] = w
        print(f"  [train-period weights] {n_zero}/{E} filtered edges have 0 "
              f"train-period journeys (kept at floor weight).")
        if self.log_weights:
            weights = np.log1p(weights)
        weights = np.maximum(weights, 1e-12)
        return weights


def dot_h1(embeddings, dataset, device):
    train = getattr(dataset, "train_transitions", np.array([]))
    val = getattr(dataset, "val_transitions", np.array([]))
    test = getattr(dataset, "test_transitions", np.array([]))
    warm, g2w = _build_warm_port_mapping(train, val, test)
    res = DotProductLinkPredictor().evaluate(
        embeddings, test, warm, g2w, device=device,
        normalize=True, exclude_self=True,
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    config = load_config(args.config)
    device = get_device()
    set_seed(args.seed)
    dataset = load_maritime_dataset(config)

    train = dataset.train_transitions
    print("=" * 70)
    print("PortCity2Vec transductive robustness: train-period vs full-period weights")
    print("=" * 70)

    # Train-period-weights variant (same topology + config)
    print("\n[1/1] Retraining PortCity2Vec with TRAIN-PERIOD edge weights...")
    set_seed(args.seed)
    bl = TrainPeriodPortCity2Vec(train, epochs=args.epochs)
    emb_train = bl.get_embeddings(dataset, device=device)
    res = dot_h1(emb_train, dataset, device)

    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(f"  PortCity2Vec dot H@1, TRAIN-period weights : {res.hits_at_1 * 100:.2f}%  "
          f"(MRR {res.mrr:.4f})")
    print(f"  PortCity2Vec dot H@1, FULL-period  weights : 36.40% (reported, seed 42)")
    print(f"  Train-period frequency oracle (Markov n=1) : 37.34%")
    print(f"  Full-period  frequency oracle              : 38.30%")
    print("\nInterpretation: if the train-period number is close to the reported")
    print("full-period number, the ranking signal does not depend on information")
    print("from the evaluation window.")


if __name__ == "__main__":
    main()
