"""Evaluate DGI embeddings on next-port prediction (3 framings).

Usage:
    python experiments/maritime/run_link_prediction.py --config configs/experiments/maritime.yaml --seed 42
    python experiments/maritime/run_link_prediction.py --config configs/experiments/maritime.yaml --seed 42 --embeddings path/to/embeddings.npy

Outputs (in results/maritime/embeddings/{encoder}_seed{seed}/):
    link_prediction_results.json  - Per-framing Hits@K, MRR, per-vessel-type breakdown
    vessel_type_metrics.json      - Full per-vessel-type table (all qualifying types, all framings)
    maritime_vessel_type_link_prediction.pdf
"""
import argparse
import json

import numpy as np

from common import add_common_args, setup_experiment, load_embeddings

from src.evaluation.link_prediction import (
    DotProductLinkPredictor,
    MLPLinkPredictor,
    LSTMLinkPredictor,
    _build_warm_port_mapping,
)
from src.utils.visualizations import plot_vessel_type_metrics


def evaluate_link_prediction(embeddings_np, dataset, device, max_len=5):
    """Evaluate embeddings on next-port prediction (3 framings).

    Returns:
        Dict with results for each framing, or None if no transitions.
    """
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
        print("  Vessel type map:   not available (per-vessel-type metrics skipped)")

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

    return results


def main():
    parser = argparse.ArgumentParser(description="Maritime link prediction evaluation")
    add_common_args(parser)
    parser.add_argument(
        "--embeddings", type=str, default=None,
        help="Path to .npy embeddings file (default: auto-detect from output_dir)"
    )
    parser.add_argument(
        "--max-len", type=int, default=5,
        help="LSTM context window size (default: 5)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Maritime Link Prediction")
    print("=" * 60)

    config, dataset, device, encoder_name, output_dir = setup_experiment(args)

    # Load embeddings
    if args.embeddings:
        emb_path = args.embeddings
    else:
        emb_path = output_dir / "embeddings.npy"

    print(f"\n  Loading embeddings from {emb_path}")
    embeddings_np = load_embeddings(emb_path)
    print(f"  Embedding shape: {embeddings_np.shape}")

    # Run link prediction
    print("\nEvaluating: Link prediction (next-port)...")
    lp_results = evaluate_link_prediction(embeddings_np, dataset, device, max_len=args.max_len)

    if lp_results is None:
        print("No link prediction results (missing transitions).")
        return

    # Save full results
    results_path = output_dir / "link_prediction_results.json"
    with open(results_path, "w") as f:
        json.dump(lp_results, f, indent=2, default=str)

    # Print summary table
    print("\n" + "=" * 60)
    print("LINK PREDICTION SUMMARY")
    print("=" * 60)
    print(f"\n  {'Framing':<14} {'Hits@1':>8} {'Hits@5':>8} {'Hits@10':>8} {'MRR':>8} {'N':>10}")
    print(f"  {'-'*58}")
    for framing in ["dot_product", "mlp", "lstm"]:
        if framing in lp_results:
            r = lp_results[framing]
            print(f"  {framing:<14} {r['hits_at_1']:>8.4f} {r['hits_at_5']:>8.4f} "
                  f"{r['hits_at_10']:>8.4f} {r['mrr']:>8.4f} {r['num_transitions']:>10,}")

    # Per-vessel-type breakdowns (console)
    for framing in ["dot_product", "mlp", "lstm"]:
        if framing in lp_results and "per_vessel_type" in lp_results[framing]:
            pvt = lp_results[framing]["per_vessel_type"]
            if pvt:
                print(f"\n  Per-vessel-type ({framing}):")
                sorted_vt = sorted(pvt.items(), key=lambda x: x[1]["count"], reverse=True)
                for vt, m in sorted_vt[:5]:
                    print(f"    {vt:<25} Hits@1={m['hits_at_1']:.4f} MRR={m['mrr']:.4f} (n={m['count']:,})")

    # Build per_vt_by_framing dict for plots and separate JSON
    per_vt_by_framing = {}
    for framing in ["dot_product", "mlp", "lstm"]:
        if framing in lp_results and "per_vessel_type" in lp_results[framing]:
            pvt = lp_results[framing]["per_vessel_type"]
            if pvt:
                per_vt_by_framing[framing] = pvt

    if per_vt_by_framing:
        # Save separate vessel type metrics JSON
        vt_metrics_path = output_dir / "vessel_type_metrics.json"
        with open(vt_metrics_path, "w") as f:
            json.dump(per_vt_by_framing, f, indent=2, default=str)
        print(f"\nVessel-type metrics saved to {vt_metrics_path}")

        # Generate vessel-type bar chart
        print("Generating vessel-type link prediction plot...")
        plot_vessel_type_metrics(per_vt_by_framing, str(output_dir))
        print(f"  Saved: maritime_vessel_type_link_prediction.pdf")
    else:
        print("\n  No per-vessel-type data available (vessel_type_map missing or no qualifying types).")

    print(f"\nResults saved to {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
