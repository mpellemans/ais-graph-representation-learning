"""LSTM window-size sensitivity analysis for next-port prediction.

Sweeps max_len values and reports performance + mean effective context length.
Shows where additional context stops helping, justifying the choice of max_len=5.

Usage:
    python experiments/maritime/run_lstm_sensitivity.py \\
        --config configs/experiments/maritime.yaml --seed 42 \\
        [--embeddings path/to/embeddings.npy] \\
        [--max-lens 1,2,3,5,7,10,15]

Outputs (in results/maritime/embeddings/{encoder}_seed{seed}/):
    lstm_sensitivity_results.json  - Per max_len metrics + context stats
    lstm_sensitivity_plot.pdf      - Dual-axis line chart
"""
import argparse
import json

import numpy as np

from common import add_common_args, setup_experiment, load_embeddings

from src.evaluation.link_prediction import (
    LSTMLinkPredictor,
    build_sequences,
    _build_warm_port_mapping,
)
from src.utils.visualizations import plot_lstm_sensitivity, plot_lstm_sensitivity_k


def run_sensitivity_sweep(embeddings_np, dataset, device, max_lens, n_runs=3):
    """Sweep max_len values and record LSTM performance + context stats.

    Args:
        embeddings_np: (N, D) node embeddings
        dataset: MaritimeDataset with train/val/test_transitions
        device: torch device
        max_lens: list of int window sizes to evaluate
        n_runs: number of LSTM runs per max_len (3 for speed)

    Returns:
        List of result dicts, one per max_len.
    """
    train_trans = getattr(dataset, 'train_transitions', np.array([]))
    val_trans = getattr(dataset, 'val_transitions', np.array([]))
    test_trans = getattr(dataset, 'test_transitions', np.array([]))

    if len(train_trans) == 0 or len(test_trans) == 0:
        print("  Warning: no transitions available.")
        return []

    warm_ports, global_to_warm = _build_warm_port_mapping(train_trans, val_trans, test_trans)
    vessel_type_map = getattr(dataset, 'vessel_type_map', None)

    print(f"  Warm ports: {len(warm_ports)}")
    print(f"  Train transitions: {len(train_trans):,}")
    print(f"  Test transitions:  {len(test_trans):,}")

    # Compute reference test sequence count (invariant to max_len)
    # Every transition generates exactly one sequence regardless of window size.
    ref_test_seqs = build_sequences(test_trans, max_len=max(max_lens))
    num_test_seqs_ref = len(ref_test_seqs)
    print(f"  Reference test sequences (max_len={max(max_lens)}): {num_test_seqs_ref:,}")

    results = []
    lstm = LSTMLinkPredictor()

    for max_len in max_lens:
        print(f"\n{'='*50}")
        print(f"  max_len = {max_len}  ({n_runs} runs)")
        print(f"{'='*50}")

        # Compute mean effective context length for test sequences
        test_seqs = build_sequences(test_trans, max_len=max_len)
        mean_ctx_len = float(np.mean([len(s[0]) for s in test_seqs])) if test_seqs else 0.0
        num_test_seqs = len(test_seqs)

        # Assert invariance: number of sequences should not change with max_len
        if num_test_seqs != num_test_seqs_ref:
            print(f"  WARNING: num_test_seqs={num_test_seqs} != ref={num_test_seqs_ref} "
                  f"(unexpected for max_len={max_len})")

        print(f"  Test sequences: {num_test_seqs:,} | Mean context len: {mean_ctx_len:.2f}")

        result = lstm.train_and_evaluate(
            embeddings_np, train_trans, val_trans, test_trans,
            warm_ports, global_to_warm,
            vessel_type_map=vessel_type_map, device=device,
            n_runs=n_runs,
            max_len=max_len,
        )

        entry = {
            "max_len": max_len,
            "hits_at_1": round(result.hits_at_1, 4),
            "hits_at_5": round(result.hits_at_5, 4),
            "hits_at_10": round(result.hits_at_10, 4),
            "mrr": round(result.mrr, 4),
            "mean_context_len": round(mean_ctx_len, 3),
            "num_test_seqs": num_test_seqs,
        }
        results.append(entry)
        print(f"  --> Hits@1={result.hits_at_1:.4f}, MRR={result.mrr:.4f}, "
              f"mean_ctx={mean_ctx_len:.2f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="LSTM max_len sensitivity analysis")
    add_common_args(parser)
    parser.add_argument(
        "--embeddings", type=str, default=None,
        help="Path to .npy embeddings file (default: auto-detect from output_dir)"
    )
    parser.add_argument(
        "--max-lens", type=str, default="1,2,3,5,7,10,15",
        help="Comma-separated list of max_len values to sweep (default: 1,2,3,5,7,10,15)"
    )
    parser.add_argument(
        "--n-runs", type=int, default=3,
        help="Number of LSTM runs per max_len (default: 3 for speed)"
    )
    parser.add_argument(
        "--chosen-max-len", type=int, default=5,
        help="Chosen max_len to highlight on plot (default: 5)"
    )
    args = parser.parse_args()

    max_lens = [int(x.strip()) for x in args.max_lens.split(",")]
    max_lens = sorted(set(max_lens))

    print("=" * 60)
    print("LSTM Sensitivity Analysis")
    print(f"  max_lens: {max_lens}")
    print(f"  n_runs per max_len: {args.n_runs}")
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

    # Run sweep
    sweep_results = run_sensitivity_sweep(
        embeddings_np, dataset, device, max_lens, n_runs=args.n_runs
    )

    if not sweep_results:
        print("No results produced (missing data). Exiting.")
        return

    # Save JSON
    output = {"max_len_sweep": sweep_results}
    json_path = output_dir / "lstm_sensitivity_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSensitivity results saved to {json_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SENSITIVITY SUMMARY")
    print("=" * 60)
    print(f"\n  {'max_len':>8} {'MRR':>8} {'Hits@1':>8} {'Hits@10':>8} {'mean_ctx':>10} {'n_seqs':>10}")
    print(f"  {'-'*54}")
    for r in sweep_results:
        marker = " <--" if r['max_len'] == args.chosen_max_len else ""
        print(f"  {r['max_len']:>8} {r['mrr']:>8.4f} {r['hits_at_1']:>8.4f} "
              f"{r['hits_at_10']:>8.4f} {r['mean_context_len']:>10.2f} "
              f"{r['num_test_seqs']:>10,}{marker}")

    # Generate plots
    plot_lstm_sensitivity(sweep_results, args.chosen_max_len, str(output_dir))
    plot_lstm_sensitivity_k(sweep_results, args.chosen_max_len, str(output_dir))
    print(f"\nPlots saved to {output_dir}/lstm_sensitivity_plot.pdf")
    print(f"                and {output_dir}/lstm_sensitivity_plot_k.pdf")
    print("=" * 60)


if __name__ == "__main__":
    main()
