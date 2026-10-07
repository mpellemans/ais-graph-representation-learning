"""Embedding analysis: Analyses A–E.

Analysis A — Feature correlation:
    Pearson r between each raw feature and the first 50 PCA dimensions of DGI
    embeddings. Identifies which feature groups (geographic, traffic, connectivity)
    are preserved in the embedding space.

Analysis B — Geographic cluster quality:
    Cosine similarity of embedding pairs within the same sub-region vs. across
    regions. Optionally compares DGI to a second embedding (e.g. HOPE or
    PortCity2Vec) to establish whether its geometric structure is captured by DGI.

Analysis C — Network position analysis:
    Correlates embedding L2 norm and the first PCA dimension with network
    centrality features (hub_score, in_degree, out_degree, net_flow).

Analysis D — Vessel type specialization:
    For the two largest vessel-type groups (Container and Bulk Dry), retrieves
    the top-10 nearest neighbours in embedding space and reports what fraction
    share the same dominant vessel type.

Analysis E — Case studies:
    For selected landmark ports (Singapore, Rotterdam, Antwerp + 2 smaller
    ports), reports the top-10 nearest neighbours in DGI embedding space, along
    with their sub-region, country, and cosine distance.

Usage:
    python experiments/maritime/run_embedding_analysis.py \\
        --config configs/experiments/maritime.yaml \\
        --embeddings results/maritime/embeddings/dgcn_seed42/embeddings.npy

    # Compare DGI to HOPE (optional second embedding)
    python experiments/maritime/run_embedding_analysis.py \\
        --config configs/experiments/maritime.yaml \\
        --embeddings results/maritime/embeddings/dgcn_seed42/embeddings.npy \\
        --compare-embeddings results/maritime/embeddings/hope_seed42/embeddings.npy \\
        --compare-label HOPE

    # DGI vs PortCity2Vec (geo-similarity boxplot + case-study tables in the paper)
    python experiments/maritime/run_embedding_analysis.py \\
        --config configs/experiments/maritime.yaml \\
        --embeddings results/maritime/embeddings/dgcn_seed42/embeddings.npy \\
        --compare-embeddings results/maritime/embeddings/portcity2vec_seed42/embeddings.npy \\
        --compare-label PortCity2Vec --output-suffix _portcity2vec

Outputs (in same directory as --embeddings; Analyses B and E honor --output-suffix):
    feature_correlation.json        — top correlated features per group
    feature_correlation_heatmap.pdf — top-30 features × top-10 PCA dims heatmap
    geographic_similarity.json      — within vs. across region cosine similarity
    geographic_similarity_boxplot.pdf
    network_position.json           — centrality vs. embedding norm/PCA1 correlations
    network_position_scatter.pdf    — scatter plots (4 centrality measures)
    vessel_type_specialization.json — nearest-neighbor vessel-type analysis
    case_studies.json               — nearest neighbors for landmark ports
    case_studies_table.txt          — formatted table (copy into paper)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

# Make project importable when run directly
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.preprocessing.maritime import (
    NODE_STANDARD_COLS, NODE_LOG_COLS, NODE_PASSTHROUGH_COLS,
    NODE_CATEGORICAL_COLS, GEOHASH_LENGTH,
)


# ---------------------------------------------------------------------------
# Feature name reconstruction
# ---------------------------------------------------------------------------

def build_feature_names(metadata: dict) -> list[str]:
    """Reconstruct ordered feature names matching the node feature matrix.

    Feature layout (from preprocess_node_features):
        [0:160]       geohash one-hot (160 = GEOHASH_LENGTH * 32)
        [160:160+S]   standard-scaled continuous (S = len(NODE_STANDARD_COLS))
        [160+S]       vessel_type_entropy (passthrough)
        [160+S+1 ...]  one-hot categoricals (country, continent, sub_region,
                       dominant_vessel_type)
    """
    names = []
    # Geohash dims
    for pos in range(GEOHASH_LENGTH):
        for ch in range(32):
            names.append(f"geohash_pos{pos}_ch{ch}")

    # Standard-scaled continuous
    for col in NODE_STANDARD_COLS:
        names.append(col)

    # Passthrough
    for col in NODE_PASSTHROUGH_COLS:
        names.append(col)

    # Categorical one-hots (order from metadata)
    cat_mappings = metadata.get("preprocessing_info", {}).get("node_category_mappings", {})
    for col in NODE_CATEGORICAL_COLS:
        categories = cat_mappings.get(col, [])
        for cat in categories:
            names.append(f"{col}_{cat}")

    return names


def group_features(feature_names: list[str]) -> dict[str, list[int]]:
    """Return dict mapping group label -> list of feature indices."""
    groups = {
        "geohash": [],
        "traffic": [],
        "connectivity": [],
        "vessel": [],
        "categorical": [],
    }
    traffic_cols = {
        "total_visits", "unique_vessels", "total_arrivals", "total_departures",
        "mean_dwell_time_hours", "median_dwell_time_hours", "std_dwell_time_hours",
        "outgoing_journeys", "incoming_journeys",
    }
    connectivity_cols = {
        "out_degree", "in_degree", "total_degree", "net_flow", "hub_score",
    }
    vessel_cols = {
        "mean_deadweight", "max_deadweight", "min_deadweight",
        "mean_draught", "max_draught", "mean_vessel_age", "vessel_type_entropy",
    }

    for i, name in enumerate(feature_names):
        if name.startswith("geohash"):
            groups["geohash"].append(i)
        elif any(name.startswith(cat + "_") for cat in NODE_CATEGORICAL_COLS):
            groups["categorical"].append(i)
        elif name in connectivity_cols:
            groups["connectivity"].append(i)
        elif name in traffic_cols:
            groups["traffic"].append(i)
        elif name in vessel_cols:
            groups["vessel"].append(i)

    return groups


# ---------------------------------------------------------------------------
# Analysis A — Feature correlation
# ---------------------------------------------------------------------------

def analysis_a_feature_correlation(embeddings, features_dense, feature_names, output_dir):
    """Compute Pearson r between raw features and PCA dims of embeddings."""
    print("\n--- Analysis A: Feature Correlation ---")

    n_pca = min(50, embeddings.shape[1])
    pca = PCA(n_components=n_pca, random_state=42)
    emb_pca = pca.fit_transform(embeddings)

    n_features = features_dense.shape[1]
    # Compute max absolute correlation of each feature across all PCA dims
    max_r = np.zeros(n_features)
    best_pca_dim = np.zeros(n_features, dtype=int)

    for f_idx in range(n_features):
        feat_col = features_dense[:, f_idx]
        if feat_col.std() < 1e-9:
            continue
        rs = np.array([
            stats.pearsonr(feat_col, emb_pca[:, pc])[0]
            for pc in range(n_pca)
        ])
        max_r[f_idx] = np.max(np.abs(rs))
        best_pca_dim[f_idx] = int(np.argmax(np.abs(rs)))

    # Top 30 features by max |r|
    top30_idx = np.argsort(max_r)[::-1][:30]
    top30_names = [feature_names[i] if i < len(feature_names) else f"feat_{i}"
                   for i in top30_idx]
    top30_r = max_r[top30_idx].tolist()

    # Per-group mean |r|
    groups = group_features(feature_names)
    group_stats = {}
    for group, idxs in groups.items():
        if idxs:
            group_stats[group] = {
                "mean_max_r": float(np.mean(max_r[idxs])),
                "max_max_r": float(np.max(max_r[idxs])),
                "n_features": len(idxs),
            }
    print(f"  PCA dims used: {n_pca} (explains "
          f"{pca.explained_variance_ratio_.sum()*100:.1f}% variance)")
    for group, gs in sorted(group_stats.items(), key=lambda x: -x[1]["mean_max_r"]):
        print(f"  {group:<14} mean |r| = {gs['mean_max_r']:.4f}  "
              f"max |r| = {gs['max_max_r']:.4f}  (n={gs['n_features']})")

    result = {
        "top30_features": [
            {"name": name, "max_abs_r": round(r, 4), "best_pca_dim": int(best_pca_dim[top30_idx[i]])}
            for i, (name, r) in enumerate(zip(top30_names, top30_r))
        ],
        "group_stats": group_stats,
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    # Save JSON
    out_path = output_dir / "feature_correlation.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")

    # Heatmap: top-30 features x top-10 PCA dims
    n_pca_plot = min(10, n_pca)
    corr_matrix = np.zeros((30, n_pca_plot))
    for row_i, f_idx in enumerate(top30_idx):
        feat_col = features_dense[:, f_idx]
        if feat_col.std() < 1e-9:
            continue
        for pc in range(n_pca_plot):
            corr_matrix[row_i, pc] = stats.pearsonr(feat_col, emb_pca[:, pc])[0]

    fig, ax = plt.subplots(figsize=(9, 11))
    short_names = [n.replace("geohash_pos", "geo_p").replace("country_", "ctry_")
                   .replace("sub_region_", "sr_").replace("dominant_vessel_type_", "dvt_")
                   .replace("continent_", "cont_")[:30]
                   for n in top30_names]
    sns.heatmap(
        corr_matrix,
        xticklabels=[f"PC{i+1}" for i in range(n_pca_plot)],
        yticklabels=short_names,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        ax=ax, cbar_kws={"label": "Pearson r"},
    )
    ax.set_xlabel("PCA dimension of DGI embeddings", fontsize=14)
    ax.set_ylabel("Raw feature", fontsize=14)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=12)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Pearson r", fontsize=13)
    plt.tight_layout()
    heatmap_path = output_dir / "feature_correlation_heatmap.pdf"
    plt.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {heatmap_path}")

    return result


# ---------------------------------------------------------------------------
# Analysis B — Geographic cluster quality
# ---------------------------------------------------------------------------

def analysis_b_geographic_similarity(embeddings, labels, output_dir,
                                     compare_embeddings=None,
                                     compare_label="HOPE", suffix=""):
    """Cosine similarity within vs. across sub-regions."""
    print("\n--- Analysis B: Geographic Cluster Quality ---")
    compare_key = compare_label.lower().replace(" ", "")

    def compute_within_across(embs, labs):
        embs_norm = normalize(embs, norm="l2")
        unique_labels = np.unique(labs)
        within_sims = []
        across_sims = []

        # Sample at most 500 pairs per category to keep it tractable
        rng = np.random.default_rng(42)
        for lbl in unique_labels:
            mask = labs == lbl
            idxs = np.where(mask)[0]
            other_idxs = np.where(~mask)[0]

            # Within-region pairs
            if len(idxs) >= 2:
                n_pairs = min(500, len(idxs) * (len(idxs) - 1) // 2)
                pairs = rng.choice(len(idxs), size=(n_pairs, 2), replace=True)
                pairs = pairs[pairs[:, 0] != pairs[:, 1]]
                for a, b in pairs[:200]:
                    within_sims.append(float(embs_norm[idxs[a]] @ embs_norm[idxs[b]]))

            # Across-region pairs
            if len(idxs) > 0 and len(other_idxs) > 0:
                n_pairs = min(200, len(idxs) * len(other_idxs))
                i_sample = rng.choice(idxs, size=min(n_pairs, len(idxs)), replace=True)
                j_sample = rng.choice(other_idxs, size=min(n_pairs, len(other_idxs)), replace=True)
                for a, b in zip(i_sample[:100], j_sample[:100]):
                    across_sims.append(float(embs_norm[a] @ embs_norm[b]))

        return np.array(within_sims), np.array(across_sims)

    within_dgi, across_dgi = compute_within_across(embeddings, labels)
    print(f"  DGI: within mean={within_dgi.mean():.4f}±{within_dgi.std():.4f}, "
          f"across mean={across_dgi.mean():.4f}±{across_dgi.std():.4f}")
    t_stat, p_val = stats.ttest_ind(within_dgi, across_dgi, equal_var=False)
    print(f"  DGI: Welch t={t_stat:.2f}, p={p_val:.4g}")

    result = {
        "dgi": {
            "within_mean": float(within_dgi.mean()),
            "within_std": float(within_dgi.std()),
            "across_mean": float(across_dgi.mean()),
            "across_std": float(across_dgi.std()),
            "t_stat": float(t_stat),
            "p_value": float(p_val),
        }
    }

    if compare_embeddings is not None:
        within_cmp, across_cmp = compute_within_across(compare_embeddings, labels)
        t_stat_h, p_val_h = stats.ttest_ind(within_cmp, across_cmp, equal_var=False)
        print(f"  {compare_label}: within mean={within_cmp.mean():.4f}±{within_cmp.std():.4f}, "
              f"across mean={across_cmp.mean():.4f}±{across_cmp.std():.4f}")
        result[compare_key] = {
            "within_mean": float(within_cmp.mean()),
            "within_std": float(within_cmp.std()),
            "across_mean": float(across_cmp.mean()),
            "across_std": float(across_cmp.std()),
            "t_stat": float(t_stat_h),
            "p_value": float(p_val_h),
        }

    # Save JSON
    out_path = output_dir / f"geographic_similarity{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")

    # Box plot
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_data = []
    if compare_embeddings is not None:
        plot_data += [
            {"Similarity": v, "Type": "Within region", "Method": "DGI"} for v in within_dgi
        ] + [
            {"Similarity": v, "Type": "Across regions", "Method": "DGI"} for v in across_dgi
        ] + [
            {"Similarity": v, "Type": "Within region", "Method": compare_label} for v in within_cmp
        ] + [
            {"Similarity": v, "Type": "Across regions", "Method": compare_label} for v in across_cmp
        ]
        df_plot = pd.DataFrame(plot_data)
        sns.boxplot(data=df_plot, x="Method", y="Similarity", hue="Type",
                    palette=["#2196F3", "#FF9800"], ax=ax, showfliers=False)
    else:
        plot_data = [
            {"Similarity": v, "Type": "Within region"} for v in within_dgi
        ] + [
            {"Similarity": v, "Type": "Across regions"} for v in across_dgi
        ]
        df_plot = pd.DataFrame(plot_data)
        sns.boxplot(data=df_plot, x="Type", y="Similarity",
                    palette=["#2196F3", "#FF9800"], ax=ax, showfliers=False)

    ax.set_title("Cosine Similarity: Within vs. Across Sub-regions")
    ax.set_ylabel("Cosine similarity")
    plt.tight_layout()
    boxplot_path = output_dir / f"geographic_similarity_boxplot{suffix}.pdf"
    plt.savefig(boxplot_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {boxplot_path}")

    return result


# ---------------------------------------------------------------------------
# Analysis C — Network position analysis
# ---------------------------------------------------------------------------

def analysis_c_network_position(embeddings, feature_names, features_dense, output_dir):
    """Correlate embedding norm and PCA1 with network centrality features."""
    print("\n--- Analysis C: Network Position Analysis ---")

    centrality_cols = ["hub_score", "net_flow", "in_degree", "out_degree",
                       "total_degree", "total_visits"]
    # Find indices of centrality features
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    centrality_idxs = {col: name_to_idx[col] for col in centrality_cols if col in name_to_idx}

    if not centrality_idxs:
        print("  Warning: no centrality features found in feature matrix. Skipping.")
        return {}

    pca = PCA(n_components=1, random_state=42)
    pca1 = pca.fit_transform(embeddings).ravel()
    emb_norm = np.linalg.norm(embeddings, axis=1)

    result = {}
    corr_rows = []
    for col, idx in centrality_idxs.items():
        feat_col = features_dense[:, idx]
        r_norm, p_norm = stats.pearsonr(feat_col, emb_norm)
        r_pca1, p_pca1 = stats.pearsonr(feat_col, pca1)
        result[col] = {
            "r_with_norm": round(float(r_norm), 4),
            "p_norm": float(p_norm),
            "r_with_pca1": round(float(r_pca1), 4),
            "p_pca1": float(p_pca1),
        }
        corr_rows.append({"feature": col, "r_norm": r_norm, "r_pca1": r_pca1})
        print(f"  {col:<20} r(norm)={r_norm:+.4f}  r(PCA1)={r_pca1:+.4f}")

    out_path = output_dir / "network_position.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")

    # Scatter plots: 4 panels (hub_score, net_flow, in_degree, out_degree)
    plot_cols = [c for c in ["hub_score", "net_flow", "in_degree", "out_degree"]
                 if c in centrality_idxs]
    if plot_cols:
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        axes = axes.ravel()
        for i, col in enumerate(plot_cols[:4]):
            ax = axes[i]
            feat_col = features_dense[:, centrality_idxs[col]]
            ax.scatter(feat_col, emb_norm, alpha=0.3, s=10, rasterized=True)
            r, p = stats.pearsonr(feat_col, emb_norm)
            ax.set_xlabel(col.replace("_", " ").title() + " (scaled)", fontsize=16)
            ax.set_ylabel("Embedding L2 norm", fontsize=16)
            ax.set_title(f"r = {r:+.3f}  (p={p:.3g})", fontsize=16)
            ax.tick_params(axis='both', labelsize=13)
        for j in range(len(plot_cols), 4):
            axes[j].set_visible(False)
        plt.suptitle("Network Centrality vs. DGI Embedding Norm", fontsize=18)
        plt.tight_layout()
        scatter_path = output_dir / "network_position_scatter.pdf"
        plt.savefig(scatter_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {scatter_path}")

    return result


# ---------------------------------------------------------------------------
# Analysis D — Vessel type specialization
# ---------------------------------------------------------------------------

def analysis_d_vessel_type(embeddings, port_to_idx, nodes_df, output_dir, top_k=10):
    """For top vessel-type groups, report NN fraction with the same dominant type."""
    print("\n--- Analysis D: Vessel Type Specialization ---")

    idx_to_port = {v: k for k, v in port_to_idx.items()}

    # Get the dominant_vessel_type for each port
    port_dvt = nodes_df.set_index("port_name")["dominant_vessel_type"].to_dict()
    port_dvt_arr = np.array([
        port_dvt.get(idx_to_port.get(i, ""), "Unknown")
        for i in range(len(embeddings))
    ])

    # Focus on top-2 vessel types by count
    from collections import Counter
    type_counts = Counter(port_dvt_arr)
    top_types = [t for t, _ in type_counts.most_common(5)
                 if t not in ("Unknown", "nan", "") and type_counts[t] >= 10][:2]

    if not top_types:
        print("  Not enough vessel type data. Skipping.")
        return {}

    embs_norm = normalize(embeddings, norm="l2")
    result = {}

    for vtype in top_types:
        mask = port_dvt_arr == vtype
        type_idxs = np.where(mask)[0]
        print(f"\n  Vessel type: {vtype} (n={len(type_idxs)} ports)")

        # For each port of this type, find top_k NN (excluding self)
        same_type_fracs = []
        sample_idxs = type_idxs
        # sample_idxs = type_idxs[:min(50, len(type_idxs))]  # cap for speed

        for port_idx in sample_idxs:
            sims = embs_norm[port_idx] @ embs_norm.T
            sims[port_idx] = -1  # exclude self
            nn_idxs = np.argsort(sims)[::-1][:top_k]
            nn_types = port_dvt_arr[nn_idxs]
            same_type_fracs.append((nn_types == vtype).mean())

        mean_frac = float(np.mean(same_type_fracs))
        std_frac = float(np.std(same_type_fracs))
        print(f"  Mean fraction of top-{top_k} NN with same type: {mean_frac:.3f} ± {std_frac:.3f}")
        print(f"  (Baseline random: {mask.mean():.3f})")

        result[vtype] = {
            "n_ports": int(len(type_idxs)),
            "base_rate": round(float(mask.mean()), 4),
            "mean_nn_same_type_frac": round(mean_frac, 4),
            "std_nn_same_type_frac": round(std_frac, 4),
        }

        # Example: top port of this type + its NN
        if len(sample_idxs) > 0:
            port_idx = sample_idxs[0]
            port_name = idx_to_port.get(port_idx, f"port_{port_idx}")
            sims = embs_norm[port_idx] @ embs_norm.T
            sims[port_idx] = -1
            nn_idxs = np.argsort(sims)[::-1][:5]
            nn_names = [idx_to_port.get(int(i), f"port_{i}") for i in nn_idxs]
            nn_types = [port_dvt_arr[i] for i in nn_idxs]
            result[vtype]["example_port"] = port_name
            result[vtype]["example_nn"] = [
                {"port": n, "type": t} for n, t in zip(nn_names, nn_types)
            ]
            print(f"  Example ({port_name}): NN = {', '.join(nn_names)}")

    out_path = output_dir / "vessel_type_specialization.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {out_path}")

    return result


# ---------------------------------------------------------------------------
# Analysis E — Case studies
# ---------------------------------------------------------------------------

CASE_STUDY_PORTS = [
    "Rotterdam",
    "Singapore",
    "Antwerp",
    "Hamburg",
    "Port Klang",      # major Asian hub
    "Constanta",       # smaller Black Sea port
]


def analysis_e_case_studies(embeddings, port_to_idx, nodes_df, output_dir,
                             top_k=10, compare_embeddings=None,
                             compare_label="HOPE", suffix=""):
    """Nearest-neighbor case studies for landmark ports."""
    print("\n--- Analysis E: Case Studies ---")
    compare_key = compare_label.lower().replace(" ", "")

    idx_to_port = {v: k for k, v in port_to_idx.items()}
    port_meta = nodes_df.set_index("port_name").to_dict("index")

    embs_norm = normalize(embeddings, norm="l2")
    if compare_embeddings is not None:
        cmp_norm = normalize(compare_embeddings, norm="l2")

    all_results = {}
    table_lines = []
    table_lines.append("=" * 100)
    table_lines.append("CASE STUDIES — Top-10 Nearest Neighbours in DGI Embedding Space")
    table_lines.append("=" * 100)

    for query_port in CASE_STUDY_PORTS:
        # Find the port index (case-insensitive prefix match)
        port_idx = port_to_idx.get(query_port)
        if port_idx is None:
            # Try case-insensitive
            for pname, pidx in port_to_idx.items():
                if pname.lower() == query_port.lower():
                    port_idx = pidx
                    query_port = pname
                    break
        if port_idx is None:
            print(f"  Port '{query_port}' not found in graph. Skipping.")
            continue

        meta = port_meta.get(query_port, {})
        print(f"\n  Query: {query_port} (country={meta.get('country', '?')}, "
              f"sub_region={meta.get('sub_region', '?')})")

        # DGI nearest neighbours
        sims = embs_norm[port_idx] @ embs_norm.T
        sims[port_idx] = -1
        nn_idxs = np.argsort(sims)[::-1][:top_k]
        nn_info = []
        for nn_idx in nn_idxs:
            nn_name = idx_to_port.get(int(nn_idx), f"port_{nn_idx}")
            nn_meta = port_meta.get(nn_name, {})
            nn_info.append({
                "port": nn_name,
                "country": nn_meta.get("country", "?"),
                "sub_region": nn_meta.get("sub_region", "?"),
                "cosine_sim": round(float(sims[nn_idx]), 4),
            })
            print(f"    [{sims[nn_idx]:.4f}] {nn_name} ({nn_meta.get('country','?')}, "
                  f"{nn_meta.get('sub_region','?')})")

        port_result = {
            "query_country": meta.get("country", "?"),
            "query_sub_region": meta.get("sub_region", "?"),
            "dgi_nn": nn_info,
        }

        # Comparison-embedding nearest neighbours (optional)
        if compare_embeddings is not None:
            cmp_sims = cmp_norm[port_idx] @ cmp_norm.T
            cmp_sims[port_idx] = -1
            cmp_nn_idxs = np.argsort(cmp_sims)[::-1][:top_k]
            cmp_nn_info = []
            for nn_idx in cmp_nn_idxs:
                nn_name = idx_to_port.get(int(nn_idx), f"port_{nn_idx}")
                nn_meta = port_meta.get(nn_name, {})
                cmp_nn_info.append({
                    "port": nn_name,
                    "country": nn_meta.get("country", "?"),
                    "sub_region": nn_meta.get("sub_region", "?"),
                    "cosine_sim": round(float(cmp_sims[nn_idx]), 4),
                })
            port_result[f"{compare_key}_nn"] = cmp_nn_info

        all_results[query_port] = port_result

        # Add to the formatted table
        table_lines.append(f"\n  {query_port} ({meta.get('country','?')}, {meta.get('sub_region','?')})")
        table_lines.append(f"  {'Rank':<5} {'Port':<30} {'Country':<20} {'Sub-region':<25} {'Sim':>6}")
        table_lines.append("  " + "-" * 90)
        for rank, nn in enumerate(nn_info, 1):
            table_lines.append(
                f"  {rank:<5} {nn['port']:<30} {nn['country']:<20} "
                f"{nn['sub_region']:<25} {nn['cosine_sim']:>6.4f}"
            )

    table_lines.append("\n" + "=" * 100)
    table_str = "\n".join(table_lines)

    # Save outputs
    out_json = output_dir / f"case_studies{suffix}.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_json}")

    out_txt = output_dir / f"case_studies_table{suffix}.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"  Saved: {out_txt}")
    print("\n" + table_str)

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Maritime embedding analysis (Analyses A-E)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to maritime YAML config")
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to .npy DGI embeddings file")
    parser.add_argument("--compare-embeddings", type=str, default=None,
                        help="(Optional) Path to a second .npy embedding for "
                             "comparison in Analyses B and E (e.g. HOPE, PortCity2Vec)")
    parser.add_argument("--compare-label", type=str, default="HOPE",
                        help="Display label for the comparison embedding "
                             "(also used as JSON key, lowercased)")
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix appended to Analysis B/E output filenames "
                             "(e.g. _portcity2vec)")
    parser.add_argument("--skip", nargs="+", default=[],
                        choices=["A", "B", "C", "D", "E"],
                        help="Skip specific analyses")
    args = parser.parse_args()

    print("=" * 70)
    print("Maritime Embedding Analysis (Analyses A–E)")
    print("=" * 70)

    # --- Load config & embeddings ---
    config = load_config(args.config)
    data_dir = Path(config.data.data_dir)
    emb_path = Path(args.embeddings)
    output_dir = emb_path.parent

    print(f"  Embeddings: {emb_path}")
    print(f"  Data dir:   {data_dir}")
    print(f"  Output dir: {output_dir}")

    embeddings = np.load(emb_path).astype(np.float32)
    print(f"  Embedding shape: {embeddings.shape}")

    compare_embeddings = None
    if args.compare_embeddings:
        compare_embeddings = np.load(args.compare_embeddings).astype(np.float32)
        print(f"  {args.compare_label} embeddings: {compare_embeddings.shape}")

    # --- Load cached features & metadata ---
    cache_path = data_dir / "processed"
    node_feat_sparse = sp.load_npz(cache_path / "node_features.npz")
    features_dense = np.array(node_feat_sparse.todense(), dtype=np.float32)
    labels = np.load(cache_path / "labels.npy")

    with open(cache_path / "metadata.json") as f:
        metadata = json.load(f)
    port_to_idx = metadata.get("port_to_idx", {})

    feature_names = build_feature_names(metadata)
    # Truncate to actual feature dim if needed
    feature_names = feature_names[:features_dense.shape[1]]

    print(f"  Nodes: {features_dense.shape[0]}, Features: {features_dense.shape[1]}")
    print(f"  Feature names reconstructed: {len(feature_names)}")

    # --- Load nodes.parquet for geographic info ---
    nodes_parquet = data_dir / "nodes.parquet"
    if nodes_parquet.exists():
        nodes_df = pd.read_parquet(nodes_parquet)
        # Keep only ports that are in the graph
        idx_to_port = {v: k for k, v in port_to_idx.items()}
        graph_ports = set(port_to_idx.keys())
        nodes_df = nodes_df[nodes_df["port_name"].isin(graph_ports)].copy()
    else:
        print("  Warning: nodes.parquet not found — some analyses may be limited")
        nodes_df = pd.DataFrame(columns=["port_name", "country", "sub_region",
                                         "dominant_vessel_type", "latitude", "longitude"])

    print()

    # --- Run analyses ---
    if "A" not in args.skip:
        analysis_a_feature_correlation(embeddings, features_dense, feature_names, output_dir)

    if "B" not in args.skip:
        analysis_b_geographic_similarity(embeddings, labels, output_dir,
                                         compare_embeddings=compare_embeddings,
                                         compare_label=args.compare_label,
                                         suffix=args.output_suffix)

    if "C" not in args.skip:
        analysis_c_network_position(embeddings, feature_names, features_dense, output_dir)

    if "D" not in args.skip:
        if nodes_df.empty or "dominant_vessel_type" not in nodes_df.columns:
            print("\n--- Analysis D: Skipped (dominant_vessel_type not in nodes_df) ---")
        else:
            analysis_d_vessel_type(embeddings, port_to_idx, nodes_df, output_dir)

    if "E" not in args.skip:
        if nodes_df.empty:
            print("\n--- Analysis E: Skipped (nodes_df not available) ---")
        else:
            analysis_e_case_studies(embeddings, port_to_idx, nodes_df, output_dir,
                                    compare_embeddings=compare_embeddings,
                                    compare_label=args.compare_label,
                                    suffix=args.output_suffix)

    print("\n" + "=" * 70)
    print("All analyses complete.")
    print(f"Outputs saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
