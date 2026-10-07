"""Run a single baseline method and evaluate on next-port prediction.

Usage:
    python experiments/maritime/run_baseline.py \
        --config configs/experiments/maritime.yaml \
        --method raw_features \
        --seed 42

    python experiments/maritime/run_baseline.py \
        --config configs/experiments/maritime.yaml \
        --method node2vec --seed 42

Supported methods: raw_features, node2vec, graphsage, gae, hope, portcity2vec,
markov_n{1,2,3,5,7,10}

Outputs (in results/maritime/embeddings/{method}_seed{seed}/):
    embeddings.npy              - (N, D) float32 node embeddings
    embedding_results.json      - method name, shape, timing
    link_prediction_results.json
    vessel_type_metrics.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the project root is importable when run directly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
# Also allow bare 'from common import ...' like other maritime scripts
sys.path.insert(0, str(Path(__file__).parent))

from common import (
    add_common_args,
    set_seed,
    get_device,
    build_output_dir,
    load_maritime_dataset,
)
from src.config import load_config
from src.evaluation.link_prediction import (
    DotProductLinkPredictor,
    MLPLinkPredictor,
    LSTMLinkPredictor,
    _build_warm_port_mapping,
    evaluate_score_matrix,
    evaluate_sequence_predictor,
)
from src.utils.visualizations import plot_vessel_type_metrics

SUPPORTED_METHODS = [
    "raw_features", "node2vec", "graphsage", "gae", "hope", "portcity2vec",
    "markov_n1", "markov_n2", "markov_n3", "markov_n5", "markov_n7", "markov_n10",
]
MARKOV_METHODS = {"markov_n1", "markov_n2", "markov_n3", "markov_n5", "markov_n7", "markov_n10"}


def _markov_order(method: str) -> int:
    return int(method.split("_n")[1])


def build_baseline(method: str, args):
    """Instantiate the requested baseline."""
    if method == "raw_features":
        from src.baselines.raw_features import RawFeaturesBaseline
        return RawFeaturesBaseline()

    elif method == "node2vec":
        from src.baselines.node2vec import Node2VecBaseline
        # 100 epochs matches the training budget used for the published results
        # (GraphSAGE 100 / GAE 200); a 1-epoch run undertrains Node2Vec.
        epochs = args.epochs if args.epochs is not None else 100
        return Node2VecBaseline(epochs=epochs)

    elif method == "graphsage":
        from src.baselines.graphsage import GraphSAGEBaseline
        epochs = args.epochs if args.epochs is not None else 100
        return GraphSAGEBaseline(epochs=epochs)

    elif method == "gae":
        from src.baselines.gae import GAEBaseline
        epochs = args.epochs if args.epochs is not None else 200
        return GAEBaseline(epochs=epochs)

    elif method == "hope":
        from src.baselines.hope import HOPEBaseline
        return HOPEBaseline()

    elif method == "portcity2vec":
        from src.baselines.portcity2vec import PortCity2VecBaseline
        # Weighted biased random walk (p=q=0.5) + skip-gram on the same filtered
        # graph as the other baselines; only the weighting and p/q bias differ.
        epochs = args.epochs if args.epochs is not None else 5
        return PortCity2VecBaseline(epochs=epochs)

    elif method in MARKOV_METHODS:
        from src.baselines.markov import MarkovBaseline, HigherOrderMarkovBaseline
        order = _markov_order(method)
        if order == 1:
            return MarkovBaseline()
        return HigherOrderMarkovBaseline(order=order)

    else:
        raise ValueError(f"Unknown method: {method}. Choose from {SUPPORTED_METHODS}")


def run_markov_evaluation(baseline, dataset, device, output_dir, max_len=5):
    """Run dot-product and LSTM evaluation for a Markov baseline (no MLP)."""
    train_trans = getattr(dataset, 'train_transitions', np.array([]))
    val_trans = getattr(dataset, 'val_transitions', np.array([]))
    test_trans = getattr(dataset, 'test_transitions', np.array([]))

    if len(train_trans) == 0 or len(test_trans) == 0:
        print("  Warning: no transitions available. Skipping evaluation.")
        return None

    warm_ports, global_to_warm = _build_warm_port_mapping(train_trans, val_trans, test_trans)
    vessel_type_map = getattr(dataset, 'vessel_type_map', None)

    print(f"  Warm ports: {len(warm_ports)}")
    print(f"  Train transitions: {len(train_trans):,}")
    print(f"  Val transitions:   {len(val_trans):,}")
    print(f"  Test transitions:  {len(test_trans):,}")

    # Fit the Markov model on training transitions
    print("\n  Fitting Markov model on training transitions...")
    t0 = time.time()
    baseline.fit(train_trans, warm_ports)
    print(f"  Fit time: {time.time() - t0:.1f}s")

    results = {}

    # Framing A: Dot-Product (score matrix)
    print("\n  Framing A: Dot-Product (score matrix)...")
    t0 = time.time()
    dp_result = evaluate_score_matrix(
        baseline.score_matrix(), test_trans, warm_ports, global_to_warm,
        vessel_type_map=vessel_type_map, device=device,
    )
    print(f"    Hits@1={dp_result.hits_at_1:.4f}, Hits@5={dp_result.hits_at_5:.4f}, "
          f"Hits@10={dp_result.hits_at_10:.4f}, MRR={dp_result.mrr:.4f} "
          f"({time.time() - t0:.1f}s)")
    results["dot_product"] = {
        "hits_at_1": round(dp_result.hits_at_1, 4),
        "hits_at_5": round(dp_result.hits_at_5, 4),
        "hits_at_10": round(dp_result.hits_at_10, 4),
        "mrr": round(dp_result.mrr, 4),
        "num_transitions": dp_result.num_transitions,
    }
    if dp_result.per_vessel_type:
        results["dot_product"]["per_vessel_type"] = dp_result.per_vessel_type

    # Framing C: LSTM (sequence predictor — no training needed)
    print(f"\n  Framing C: LSTM (sequence predictor, max_len={max_len})...")
    t0 = time.time()
    lstm_result = evaluate_sequence_predictor(
        baseline, test_trans, warm_ports, global_to_warm,
        max_len=max_len, vessel_type_map=vessel_type_map,
    )
    print(f"    Hits@1={lstm_result.hits_at_1:.4f}, Hits@5={lstm_result.hits_at_5:.4f}, "
          f"Hits@10={lstm_result.hits_at_10:.4f}, MRR={lstm_result.mrr:.4f} "
          f"({time.time() - t0:.1f}s)")
    results["lstm"] = {
        "hits_at_1": round(lstm_result.hits_at_1, 4),
        "hits_at_5": round(lstm_result.hits_at_5, 4),
        "hits_at_10": round(lstm_result.hits_at_10, 4),
        "mrr": round(lstm_result.mrr, 4),
        "num_transitions": lstm_result.num_transitions,
    }
    if lstm_result.per_vessel_type:
        results["lstm"]["per_vessel_type"] = lstm_result.per_vessel_type

    # Note: MLP framing is skipped for Markov (method-agnostic ~37%, no value)

    # Save results
    results_path = output_dir / "link_prediction_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Per-vessel-type JSON and plot
    per_vt_by_framing = {
        framing: results[framing]["per_vessel_type"]
        for framing in ["dot_product", "lstm"]
        if framing in results and "per_vessel_type" in results[framing]
        and results[framing]["per_vessel_type"]
    }
    if per_vt_by_framing:
        vt_path = output_dir / "vessel_type_metrics.json"
        with open(vt_path, "w") as f:
            json.dump(per_vt_by_framing, f, indent=2, default=str)
        print(f"\nVessel-type metrics saved to {vt_path}")
        plot_vessel_type_metrics(per_vt_by_framing, str(output_dir))

    return results


def run_link_prediction(embeddings_np, dataset, device, output_dir, max_len=5):
    """Run all three link-prediction framings and save results."""
    train_trans = getattr(dataset, 'train_transitions', np.array([]))
    val_trans = getattr(dataset, 'val_transitions', np.array([]))
    test_trans = getattr(dataset, 'test_transitions', np.array([]))

    if len(train_trans) == 0 or len(test_trans) == 0:
        print("  Warning: no transitions available. Skipping link prediction.")
        return None

    warm_ports, global_to_warm = _build_warm_port_mapping(train_trans, val_trans, test_trans)
    vessel_type_map = getattr(dataset, 'vessel_type_map', None)

    print(f"  Warm ports: {len(warm_ports)}")
    print(f"  Train transitions: {len(train_trans):,}")
    print(f"  Val transitions:   {len(val_trans):,}")
    print(f"  Test transitions:  {len(test_trans):,}")
    if vessel_type_map:
        print(f"  Vessel type map:   {len(vessel_type_map):,} vessels")
    else:
        print("  Vessel type map:   not available")

    results = {}

    # Framing A: Dot-Product
    print("\n  Framing A: Dot-Product...")
    dp = DotProductLinkPredictor()
    # Fair retrieval setting: cosine similarity (norm-invariant) with the origin
    # masked from its own candidate ranking. See Eq. dot_score in the paper.
    dp_result = dp.evaluate(
        embeddings_np, test_trans, warm_ports, global_to_warm,
        vessel_type_map=vessel_type_map, device=device,
        normalize=True, exclude_self=True,
    )
    results["dot_product"] = {
        "hits_at_1": round(dp_result.hits_at_1, 4),
        "hits_at_5": round(dp_result.hits_at_5, 4),
        "hits_at_10": round(dp_result.hits_at_10, 4),
        "mrr": round(dp_result.mrr, 4),
        "num_transitions": dp_result.num_transitions,
    }
    if dp_result.per_vessel_type:
        results["dot_product"]["per_vessel_type"] = dp_result.per_vessel_type
    print(f"    Hits@1={dp_result.hits_at_1:.4f}, Hits@5={dp_result.hits_at_5:.4f}, "
          f"Hits@10={dp_result.hits_at_10:.4f}, MRR={dp_result.mrr:.4f}")

    # Framing B: MLP
    print("\n  Framing B: MLP (5 runs)...")
    mlp = MLPLinkPredictor()
    mlp_result = mlp.train_and_evaluate(
        embeddings_np, train_trans, val_trans, test_trans,
        warm_ports, global_to_warm,
        vessel_type_map=vessel_type_map, device=device,
    )
    results["mlp"] = {
        "hits_at_1": round(mlp_result.hits_at_1, 4),
        "hits_at_5": round(mlp_result.hits_at_5, 4),
        "hits_at_10": round(mlp_result.hits_at_10, 4),
        "mrr": round(mlp_result.mrr, 4),
        "num_transitions": mlp_result.num_transitions,
    }
    if mlp_result.per_vessel_type:
        results["mlp"]["per_vessel_type"] = mlp_result.per_vessel_type
    print(f"    Mean: Hits@1={mlp_result.hits_at_1:.4f}, Hits@5={mlp_result.hits_at_5:.4f}, "
          f"Hits@10={mlp_result.hits_at_10:.4f}, MRR={mlp_result.mrr:.4f}")

    # Framing C: LSTM
    print(f"\n  Framing C: LSTM (5 runs, max_len={max_len})...")
    lstm = LSTMLinkPredictor()
    lstm_result = lstm.train_and_evaluate(
        embeddings_np, train_trans, val_trans, test_trans,
        warm_ports, global_to_warm,
        vessel_type_map=vessel_type_map, device=device,
        max_len=max_len,
    )
    results["lstm"] = {
        "hits_at_1": round(lstm_result.hits_at_1, 4),
        "hits_at_5": round(lstm_result.hits_at_5, 4),
        "hits_at_10": round(lstm_result.hits_at_10, 4),
        "mrr": round(lstm_result.mrr, 4),
        "num_transitions": lstm_result.num_transitions,
    }
    if lstm_result.per_vessel_type:
        results["lstm"]["per_vessel_type"] = lstm_result.per_vessel_type
    print(f"    Mean: Hits@1={lstm_result.hits_at_1:.4f}, Hits@5={lstm_result.hits_at_5:.4f}, "
          f"Hits@10={lstm_result.hits_at_10:.4f}, MRR={lstm_result.mrr:.4f}")

    # Save results
    results_path = output_dir / "link_prediction_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Per-vessel-type JSON and plot
    per_vt_by_framing = {}
    for framing in ["dot_product", "mlp", "lstm"]:
        if framing in results and "per_vessel_type" in results[framing]:
            pvt = results[framing]["per_vessel_type"]
            if pvt:
                per_vt_by_framing[framing] = pvt

    if per_vt_by_framing:
        vt_path = output_dir / "vessel_type_metrics.json"
        with open(vt_path, "w") as f:
            json.dump(per_vt_by_framing, f, indent=2, default=str)
        print(f"\nVessel-type metrics saved to {vt_path}")
        plot_vessel_type_metrics(per_vt_by_framing, str(output_dir))

    return results


def main():
    parser = argparse.ArgumentParser(description="Maritime baseline evaluation")
    add_common_args(parser)
    parser.add_argument(
        "--method", type=str, required=True, choices=SUPPORTED_METHODS,
        help=f"Baseline method: {', '.join(SUPPORTED_METHODS)}",
    )
    parser.add_argument(
        "--max-len", type=int, default=5,
        help="LSTM context window size (default: 5)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print(f"Maritime Baseline: {args.method}")
    print("=" * 60)

    # Load config and apply overrides
    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.epochs is not None:
        config.training.epochs = args.epochs

    seed = config.seed
    device = get_device()
    set_seed(seed)

    output_dir = build_output_dir(config, args.method, seed)
    dataset = load_maritime_dataset(config)

    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Device:     {device} ({gpu_name})")
    else:
        print(f"  Device:     {device} (no CUDA)")
    print(f"  Method:     {args.method}")
    print(f"  Seed:       {seed}")
    print(f"  Nodes:      {dataset.num_nodes}")
    print(f"  Features:   {dataset.num_features}")
    print(f"  Output:     {output_dir}")

    baseline = build_baseline(args.method, args)

    # ---- Markov methods: score-matrix path (no embeddings) ----
    if args.method in MARKOV_METHODS:
        print(f"\nRunning Markov baseline (order={_markov_order(args.method)})...")
        lp_results = run_markov_evaluation(
            baseline, dataset, device, output_dir, max_len=args.max_len,
        )
        if lp_results is None:
            print("No results (missing transitions).")
            return

        # Save minimal embedding_results.json (no actual embeddings file)
        with open(output_dir / "embedding_results.json", "w") as f:
            json.dump({"method": args.method, "seed": seed, "shape": None}, f, indent=2)

        # Summary table
        print("\n" + "=" * 60)
        print(f"SUMMARY: {args.method} (seed={seed})")
        print("=" * 60)
        print(f"\n  {'Framing':<14} {'Hits@1':>8} {'Hits@5':>8} {'Hits@10':>8} {'MRR':>8}")
        print(f"  {'-'*48}")
        for framing in ["dot_product", "lstm"]:
            if framing in lp_results:
                r = lp_results[framing]
                print(f"  {framing:<14} {r['hits_at_1']:>8.4f} {r['hits_at_5']:>8.4f} "
                      f"{r['hits_at_10']:>8.4f} {r['mrr']:>8.4f}")
        print(f"\nResults saved to {output_dir}")
        print("=" * 60)
        return

    # ---- Embedding-based methods ----
    # Generate embeddings
    print(f"\nGenerating embeddings ({args.method})...")
    t0 = time.time()
    embeddings = baseline.get_embeddings(dataset, device=device)
    elapsed = time.time() - t0

    print(f"  Shape: {embeddings.shape}, dtype: {embeddings.dtype}, time: {elapsed:.1f}s")

    # Save embeddings
    emb_path = output_dir / "embeddings.npy"
    np.save(emb_path, embeddings.astype(np.float32))
    print(f"  Saved: {emb_path}")

    # Save embedding metadata
    emb_results = {
        "method": args.method,
        "seed": seed,
        "shape": list(embeddings.shape),
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(output_dir / "embedding_results.json", "w") as f:
        json.dump(emb_results, f, indent=2)

    # Run link prediction
    print("\nEvaluating: Link prediction (next-port)...")
    lp_results = run_link_prediction(
        embeddings, dataset, device, output_dir, max_len=args.max_len,
    )

    if lp_results is None:
        print("No link prediction results (missing transitions).")
        return

    # Summary table
    print("\n" + "=" * 60)
    print(f"SUMMARY: {args.method} (seed={seed})")
    print("=" * 60)
    print(f"\n  {'Framing':<14} {'Hits@1':>8} {'Hits@5':>8} {'Hits@10':>8} {'MRR':>8}")
    print(f"  {'-'*48}")
    for framing in ["dot_product", "mlp", "lstm"]:
        if framing in lp_results:
            r = lp_results[framing]
            print(f"  {framing:<14} {r['hits_at_1']:>8.4f} {r['hits_at_5']:>8.4f} "
                  f"{r['hits_at_10']:>8.4f} {r['mrr']:>8.4f}")

    print(f"\nResults saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
