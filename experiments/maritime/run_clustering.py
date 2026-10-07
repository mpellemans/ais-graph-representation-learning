"""K-means clustering of DGI embeddings against sub-region labels.

Clusters the learned embeddings with K-means (k = number of sub-regions) and
scores the clustering against the sub-region labels (Silhouette, NMI, ARI,
Davies-Bouldin), then renders the cluster visualizations.

Usage:
    python experiments/maritime/run_clustering.py --config configs/experiments/maritime.yaml --seed 42
    python experiments/maritime/run_clustering.py --config configs/experiments/maritime.yaml --seed 42 --embeddings path/to/embeddings.npy

Outputs (in results/maritime/embeddings/{encoder}_seed{seed}/):
    analysis_results.json  - K-means clustering metrics
    maritime_kmeans_cluster_tsne.pdf
    maritime_cluster_region_heatmap.pdf
    maritime_geographic_clusters.pdf
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
    davies_bouldin_score,
)
from sklearn.cluster import KMeans

from common import add_common_args, setup_experiment, load_embeddings, load_label_info

from src.utils.visualizations import (
    compute_tsne_embeddings,
    plot_cluster_tsne,
    plot_cluster_region_heatmap,
    plot_geographic_clusters,
)


def evaluate_clustering(embeddings, labels, n_clusters):
    """Evaluate embeddings with K-means clustering.

    Returns:
        Tuple of (results dict, km_labels array).
    """
    results = {}

    kmeans = KMeans(
        n_clusters=n_clusters, init="k-means++",
        n_init=10, max_iter=300, random_state=0,
    )
    km_labels = kmeans.fit_predict(embeddings)

    results["kmeans"] = {
        "silhouette": round(float(silhouette_score(embeddings, km_labels)), 4),
        "nmi": round(float(normalized_mutual_info_score(labels, km_labels)), 4),
        "ari": round(float(adjusted_rand_score(labels, km_labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(embeddings, km_labels)), 4),
        "n_clusters": n_clusters,
    }
    return results, km_labels


def main():
    parser = argparse.ArgumentParser(description="Maritime clustering analysis")
    add_common_args(parser)
    parser.add_argument(
        "--embeddings", type=str, default=None,
        help="Path to .npy embeddings file (default: auto-detect from output_dir)"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Maritime Clustering Analysis")
    print("=" * 60)

    config, dataset, device, encoder_name, output_dir = setup_experiment(args)

    # Load label info for sub-region names
    label_info = load_label_info(config.data.data_dir)
    sub_region_to_idx = label_info.get('sub_region_mapping', {})
    sub_region_names = {v: k for k, v in sub_region_to_idx.items()}  # {idx: name}

    # Load embeddings
    if args.embeddings:
        emb_path = args.embeddings
    else:
        emb_path = output_dir / "embeddings.npy"

    print(f"\n  Loading embeddings from {emb_path}")
    embeddings_np = load_embeddings(emb_path)
    print(f"  Embedding shape: {embeddings_np.shape}")

    # Compute t-SNE once (shared coordinates for all cluster plots)
    print("\nComputing t-SNE embeddings...")
    tsne_df = compute_tsne_embeddings(embeddings_np, dataset.labels, sub_region_names)

    # Clustering
    print("\nEvaluating: Clustering...")
    n_sub_regions = len(np.unique(dataset.labels))
    clustering_results, km_labels = evaluate_clustering(
        embeddings_np, dataset.labels, n_clusters=n_sub_regions
    )

    # Print summary
    print("\n" + "=" * 60)
    print("CLUSTERING SUMMARY")
    print("=" * 60)

    km = clustering_results["kmeans"]
    print(f"\n  K-means (k={km['n_clusters']}):")
    print(f"    Silhouette:     {km['silhouette']:.4f}")
    print(f"    NMI:            {km['nmi']:.4f}")
    print(f"    ARI:            {km['ari']:.4f}")
    print(f"    Davies-Bouldin: {km['davies_bouldin']:.4f}")

    # Save results
    results_path = output_dir / "analysis_results.json"
    with open(results_path, "w") as f:
        json.dump({"clustering": clustering_results}, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Clustering visualizations
    print("\nGenerating clustering plots...")
    output_dir_str = str(output_dir)

    plot_cluster_tsne(tsne_df, km_labels, "kmeans", "maritime", output_dir_str)
    print(f"  Saved: maritime_kmeans_cluster_tsne.pdf")

    plot_cluster_region_heatmap(km_labels, dataset.labels, sub_region_names, "maritime", output_dir_str)
    print(f"  Saved: maritime_cluster_region_heatmap.pdf")

    nodes_parquet = Path(config.data.data_dir) / "nodes.parquet"
    port_to_idx = getattr(dataset, 'port_to_idx', {})
    plot_geographic_clusters(port_to_idx, km_labels, nodes_parquet, "maritime", output_dir_str)
    print(f"  Saved: maritime_geographic_clusters.pdf")

    print("=" * 60)


if __name__ == "__main__":
    main()
