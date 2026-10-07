"""Regenerate maritime_geographic_clusters.pdf as a two-panel comparison.

Left panel: K-means clusters fit on standard DGI embeddings (433-dim input).
Right panel: K-means clusters fit on no-geo DGI embeddings (50-dim input,
geographic features removed).

Both panels plot ports at true coordinates colored by cluster, demonstrating
that DGI recovers geography from topology even when geographic input is removed.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    normalized_mutual_info_score,
    adjusted_rand_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.visualizations import plot_geographic_clusters_comparison
from src.data.maritime import MaritimeDataset

EMB_ROOT = ROOT / "results" / "maritime" / "embeddings"
EMB_STD = EMB_ROOT / "dgcn_seed42" / "embeddings.npy"
EMB_NOGEO = EMB_ROOT / "dgcn_no_geo_seed42" / "embeddings.npy"
DATA_DIR = ROOT / "data" / "maritime"

emb_std = np.load(EMB_STD)
emb_nogeo = np.load(EMB_NOGEO)
print(f"  std embeddings:    {emb_std.shape}")
print(f"  no-geo embeddings: {emb_nogeo.shape}")

# port_to_idx and labels are identical across configs (graph filtering does not
# depend on which features are kept). Load once from the standard cache.
dataset = MaritimeDataset(data_dir=str(DATA_DIR))
dataset.load()
n_clusters = len(np.unique(dataset.labels))
print(f"  n_clusters (sub-regions): {n_clusters}")


def fit_and_score(embeddings, labels, name):
    km = KMeans(
        n_clusters=n_clusters, init="k-means++",
        n_init=10, max_iter=300, random_state=0,
    ).fit_predict(embeddings)
    sil = silhouette_score(embeddings, km)
    nmi = normalized_mutual_info_score(labels, km)
    ari = adjusted_rand_score(labels, km)
    print(f"  {name}: Silhouette={sil:.4f}  NMI={nmi:.4f}  ARI={ari:.4f}")
    return km


print("\nFitting K-means on each embedding set...")
km_std = fit_and_score(emb_std, dataset.labels, "standard")
km_nogeo = fit_and_score(emb_nogeo, dataset.labels, "no-geo  ")

# Both panels written into the standard seed-42 dir so the LaTeX include path
# stays unchanged.
out_dir = EMB_ROOT / "dgcn_seed42"
plot_geographic_clusters_comparison(
    port_to_idx=dataset.port_to_idx,
    cluster_labels_a=km_std,
    cluster_labels_b=km_nogeo,
    subtitle_a="(a) Standard configuration (433 input features)",
    subtitle_b="(b) Geographic features removed",
    nodes_parquet_path=DATA_DIR / "nodes.parquet",
    dataset_str="maritime",
    output_dir=str(out_dir),
    suptitle="K-means clusters of DGI port embeddings, projected onto port coordinates",
)
print(f"\n  Saved: {out_dir / 'maritime_geographic_clusters.pdf'}")
