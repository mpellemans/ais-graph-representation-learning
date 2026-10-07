"""Sub-region classification evaluation of DGI embeddings.

With the standard config (exclude_geo_features=False), sub_region is one-hot
encoded in node features — results are diagnostic only.

With exclude_geo_features=True, sub_region is absent from features. Accuracy
then measures whether DGI learns geographic proximity from graph structure alone.

With --raw-features, the same 50-run logistic-regression protocol probes the RAW
node-feature matrix instead of learned embeddings (no DGI). Combined with the
no-geo config this quantifies how much regional signal the non-geographic
features carry by themselves, versus what DGI's message passing adds. A
majority-class baseline is printed for reference.

Usage:
    python experiments/maritime/run_classification.py --config configs/experiments/maritime.yaml --seed 42
    python experiments/maritime/run_classification.py --config configs/experiments/maritime_no_geo.yaml --seed 42
    python experiments/maritime/run_classification.py --config configs/experiments/maritime.yaml --seed 42 --embeddings path/to/embeddings.npy
    python experiments/maritime/run_classification.py --config configs/experiments/maritime_no_geo.yaml --seed 42 --raw-features

Outputs (in results/maritime/embeddings/{encoder}{data_suffix}_seed{seed}/):
    classification_results.json          - Sub-region accuracy (mean +/- std, 50 runs)
    maritime_sub_region_confusion.pdf    - Sub-region confusion matrix
    (--raw-features writes classification_results_raw_features.json and
     maritime_sub_region_raw_confusion.pdf instead)
"""
import argparse
import json

import numpy as np
import torch
from sklearn.metrics import classification_report

from common import add_common_args, setup_experiment, load_embeddings, load_label_info

from src.evaluation import LinearProbeEvaluator
from src.utils import visualizations


def labels_to_onehot(labels_int, num_classes):
    """Convert 1D integer labels to one-hot tensor [1, N, C]."""
    N = len(labels_int)
    onehot = np.zeros((N, num_classes), dtype=np.float32)
    onehot[np.arange(N), labels_int] = 1.0
    return torch.FloatTensor(onehot[np.newaxis])


def main():
    parser = argparse.ArgumentParser(description="Maritime sub-region classification")
    add_common_args(parser)
    parser.add_argument(
        "--embeddings", type=str, default=None,
        help="Path to .npy embeddings file (default: auto-detect from output_dir)"
    )
    parser.add_argument(
        "--raw-features", action="store_true",
        help="Probe the raw node-feature matrix instead of learned embeddings (no DGI)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Maritime Classification — Sub-region"
          + (" (raw features, no DGI)" if args.raw_features else ""))
    print("=" * 60)

    config, dataset, device, encoder_name, output_dir = setup_experiment(args)

    exclude_geo = getattr(config.data, 'exclude_geo_features', False)
    if not exclude_geo:
        print("\n  NOTE: sub_region is directly encoded in the node features — "
              "results are diagnostic only.")
    else:
        print("\n  NOTE: Geographic features excluded — accuracy measures what is "
              "learned from topology and non-geographic attributes.")

    # Load embeddings (or the raw node-feature matrix)
    if args.raw_features:
        embeddings_np = dataset.features.toarray().astype(np.float32)
        print(f"\n  Probing RAW node features: shape {embeddings_np.shape}")
        # Majority-class baseline on the test split, for reference
        test_labels = dataset.labels[dataset.idx_test]
        vals, counts = np.unique(test_labels, return_counts=True)
        majority = counts.max() / counts.sum() * 100
        print(f"  Majority-class baseline (test): {majority:.2f}%")
    else:
        if args.embeddings:
            emb_path = args.embeddings
        else:
            emb_path = output_dir / "embeddings.npy"

        print(f"\n  Loading embeddings from {emb_path}")
        embeddings_np = load_embeddings(emb_path)
        print(f"  Embedding shape: {embeddings_np.shape}")

    # Label info
    label_info = load_label_info(config.data.data_dir)
    sub_region_names = label_info.get("sub_regions", [f"Class_{i}" for i in range(dataset.num_classes)])

    # Prepare tensors
    embeddings_tensor = torch.FloatTensor(embeddings_np[np.newaxis]).to(device)
    idx_train_t = torch.LongTensor(dataset.idx_train).to(device)
    idx_test_t = torch.LongTensor(dataset.idx_test).to(device)

    results = {"classification": []}
    output_dir_str = str(output_dir)

    # Sub-region classification (50 runs)
    print("\nEvaluating: Sub-region classification (50 runs)...")
    num_sub_classes = len(sub_region_names)
    sub_labels_onehot = labels_to_onehot(dataset.labels, num_sub_classes).to(device)

    sub_evaluator = LinearProbeEvaluator(
        embedding_dim=embeddings_np.shape[1],
        num_classes=num_sub_classes,
        n_runs=50,
        device=device,
    )
    sub_results = sub_evaluator.evaluate(embeddings_tensor, sub_labels_onehot, idx_train_t, idx_test_t)

    print(f"\nSub-region: {sub_results.mean_accuracy:.2f} +/- {sub_results.std_accuracy:.2f}%")

    present_labels = sorted(set(np.concatenate([dataset.labels[dataset.idx_train],
                                                 dataset.labels[dataset.idx_test]])))
    present_names = [sub_region_names[i] for i in present_labels if i < len(sub_region_names)]
    sub_report = classification_report(
        sub_results.labels, sub_results.predictions,
        labels=present_labels, target_names=present_names, zero_division=0,
    )
    print(sub_report)

    results["classification"].append({
        "task": "Sub-region (raw features)" if args.raw_features else "Sub-region",
        "mean_accuracy": round(sub_results.mean_accuracy, 2),
        "std_accuracy": round(sub_results.std_accuracy, 2),
        "n_runs": 50,
        "geo_features_included": not exclude_geo,
    })

    # Confusion matrix
    cm_prefix = "maritime_sub_region_raw" if args.raw_features else "maritime_sub_region"
    visualizations.plot_confusion_matrix(
        sub_results.labels, sub_results.predictions,
        present_names, [],
        cm_prefix, normalize=False, output_dir=output_dir_str,
    )

    # Save results
    results_name = ("classification_results_raw_features.json"
                    if args.raw_features else "classification_results.json")
    results_path = output_dir / results_name
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
