import os

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def compute_tsne_embeddings(feat, lab, label_dict):
    # Compute tsne embeddings
    tsne = TSNE(perplexity=30, n_components=2, init='pca', max_iter=2500, random_state=12)
    embed = tsne.fit_transform(np.array(feat))
    embed= pd.DataFrame(embed).rename(columns={0:'tsne_1', 1:'tsne_2'})
    lab_arr = np.array(lab)
    if lab_arr.ndim == 1:
        labels_tsne = pd.Series(lab_arr, name='label')
    else:
        labels_tsne = pd.DataFrame(lab).idxmax(1).reset_index().rename(columns={0:'label'}).drop(columns='index')
    embed['label'] = labels_tsne
    embed['label'] = embed.label.apply(lambda x: label_dict[x])
    return embed

def plot_raw_tsne_embeddings(embed, dataset_str, output_dir='results/plots'):
    # Plot embeddings in 2D-space
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.scatterplot(data = embed, x = 'tsne_1', y = 'tsne_2', hue = 'label', palette='husl', legend=False)
    ax.set_xlabel('$TSNE-1$')
    ax.set_ylabel('$TSNE-2$')
    ax.set_title('{}: raw features'.format(dataset_str))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '{}_feature_embeddings.pdf'.format(dataset_str)))
    plt.close(fig)
    #plt.show()

def plot_learned_tsne_embeddings(embed, dataset_str, output_dir='results/plots'):
    # Plot embeddings in 2D-space
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.scatterplot(data = embed, x = 'tsne_1', y = 'tsne_2', hue = 'label', palette='husl', legend=False)
    ax.set_xlabel('$TSNE-1$')
    ax.set_ylabel('$TSNE-2$')
    ax.set_title('{}: learned DGI'.format(dataset_str))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '{}_learned_embeddings.pdf'.format(dataset_str)))
    plt.close(fig)
    #plt.show()

def plot_loss_curve(data, title, ylabel, dataset_str, output_dir='results/plots'):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax = sns.lineplot(data=data)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Epoch')
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, '{}_training_loss.pdf'.format(dataset_str)))
    plt.close(fig)
    #plt.show()

def print_classification_report(y_true, y_pred, lab_names):
    print(classification_report(
        y_true,
        y_pred,
        target_names=lab_names,
        zero_division=0
    ))

def plot_confusion_matrix(labels, predictions, classes, losses, dataset_str, normalize=False, output_dir='results/plots'):
    # ---- data ----------------------------------------------------------------
    cm = confusion_matrix(labels, predictions, labels=range(len(classes)))

    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    losses = np.asarray(losses)

    # ---- figure --------------------------------------------------------------
    fig, (ax_cm, ax_loss) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(10, 5),
        dpi=120,
        gridspec_kw={"width_ratios": [3, 2]},
        constrained_layout=True,
    )

    # Confusion‑matrix heatmap
    sns.heatmap(
        cm,
        ax=ax_cm,
        cmap="Blues",
        annot=True,
        fmt=".2f" if normalize else "d",
        square=True,
        cbar=False,
        linewidths=0.5,
        xticklabels=classes,
        yticklabels=classes,
        annot_kws={"fontsize": 8},
    )
    ax_cm.set_title(
        f"Confusion matrix{' (normalised)' if normalize else ''}", fontsize=12
    )
    ax_cm.set_xlabel("Predicted label")
    ax_cm.set_ylabel("True label")
    ax_cm.tick_params(axis="x", rotation=90)

    # Loss curve
    sns.lineplot(x=np.arange(len(losses)), y=losses, ax=ax_loss)
    ax_loss.set_title("Training loss", fontsize=12)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("BCE with logits")

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, '{}_cm_loss.pdf'.format(dataset_str)), bbox_inches="tight")
    plt.close(fig)
    #plt.show()

def plot_cluster_tsne(tsne_df, cluster_labels, title_suffix, dataset_str, output_dir):
    """Plot t-SNE scatter colored by cluster assignment (K-means or HDBSCAN)."""
    os.makedirs(output_dir, exist_ok=True)
    plot_df = tsne_df.copy()

    cluster_strs = []
    for c in cluster_labels:
        if c == -1:
            cluster_strs.append("Noise")
        else:
            cluster_strs.append(f"Cluster {c}")
    plot_df['label'] = cluster_strs

    unique_labels = sorted(set(cluster_strs), key=lambda x: (x == "Noise", x))
    non_noise = [l for l in unique_labels if l != "Noise"]
    has_noise = "Noise" in unique_labels

    palette = dict(zip(non_noise, sns.color_palette("tab20", len(non_noise))))
    if has_noise:
        palette["Noise"] = "grey"

    hue_order = non_noise + (["Noise"] if has_noise else [])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df, x='tsne_1', y='tsne_2', hue='label',
        palette=palette, hue_order=hue_order, s=20, alpha=0.7, ax=ax,
    )
    ax.set_title(f'{dataset_str}: {title_suffix} clusters')
    ax.set_xlabel('TSNE-1')
    ax.set_ylabel('TSNE-2')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'{dataset_str}_{title_suffix}_cluster_tsne.pdf'),
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_cluster_region_heatmap(cluster_labels, region_labels, region_names, dataset_str, output_dir):
    """Heatmap of K-means cluster vs sub-region co-occurrence (row-normalized)."""
    os.makedirs(output_dir, exist_ok=True)
    cluster_ids = sorted(set(cluster_labels))
    n_clusters = len(cluster_ids)
    unique_regions = sorted(set(region_labels))
    region_name_list = [region_names.get(r, str(r)) for r in unique_regions]

    matrix = np.zeros((n_clusters, len(unique_regions)))
    for ci, cid in enumerate(cluster_ids):
        mask = np.array(cluster_labels) == cid
        region_subset = np.array(region_labels)[mask]
        for j, r in enumerate(unique_regions):
            matrix[ci, j] = (region_subset == r).sum()

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix = matrix / row_sums

    row_labels = [f"C{cid}" for cid in cluster_ids]
    fig_h = max(5, n_clusters * 0.35)
    fig_w = max(8, len(unique_regions) * 0.6)
    annotate = (n_clusters * len(unique_regions)) <= 200

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        matrix, ax=ax,
        xticklabels=region_name_list,
        yticklabels=row_labels,
        cmap='Blues', fmt='.2f', annot=annotate,
        linewidths=0.3,
    )
    ax.set_title(f'{dataset_str}: K-means cluster vs sub-region')
    ax.set_xlabel('Sub-region')
    ax.set_ylabel('Cluster')
    ax.tick_params(axis='x', rotation=90, labelsize=7)
    ax.tick_params(axis='y', rotation=0, labelsize=7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'{dataset_str}_cluster_region_heatmap.pdf'),
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_geographic_clusters(port_to_idx, cluster_labels, nodes_parquet_path, dataset_str, output_dir):
    """World map scatter plot of port locations colored by K-means cluster."""
    os.makedirs(output_dir, exist_ok=True)

    if not port_to_idx:
        print("  Skipping geographic cluster plot: port_to_idx not available.")
        return

    nodes_df = pd.read_parquet(nodes_parquet_path, columns=['port_name', 'latitude', 'longitude'])
    nodes_df = nodes_df.drop_duplicates('port_name').set_index('port_name')

    idx_to_port = {v: k for k, v in port_to_idx.items()}
    records = []
    for idx in range(len(cluster_labels)):
        port_name = idx_to_port.get(idx)
        if port_name is None or port_name not in nodes_df.index:
            continue
        row = nodes_df.loc[port_name]
        records.append({
            'longitude': float(row['longitude']),
            'latitude': float(row['latitude']),
            'cluster': f"C{cluster_labels[idx]}",
        })

    if not records:
        print("  Skipping geographic cluster plot: no matching ports found.")
        return

    plot_df = pd.DataFrame(records)
    unique_clusters = sorted(set(plot_df['cluster']), key=lambda c: int(c[1:]))
    palette = dict(zip(unique_clusters, sns.color_palette("husl", len(unique_clusters))))

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(
        data=plot_df, x='longitude', y='latitude', hue='cluster',
        hue_order=unique_clusters, palette=palette,
        alpha=0.85, s=22, linewidth=0, ax=ax, legend=True,
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect('equal')
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.5)
    ax.legend(
        title='K-means cluster', title_fontsize=11, fontsize=9,
        loc='center left', bbox_to_anchor=(1.02, 0.5),
        ncol=2, frameon=False, markerscale=1.6,
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f'{dataset_str}_geographic_clusters.pdf'),
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_geographic_clusters_comparison(
    port_to_idx, cluster_labels_a, cluster_labels_b,
    subtitle_a, subtitle_b,
    nodes_parquet_path, dataset_str, output_dir,
    suptitle=None,
    output_name='maritime_geographic_clusters.pdf',
):
    """Two-panel world map: K-means cluster assignments from two embedding configurations.

    Each panel plots ports at their true coordinates, colored by cluster ID in that
    panel's embedding space. Cluster colors are assigned by the longitude of each
    cluster's geographic centroid using a perceptually ordered colormap, so
    geographically coherent clusters take on similar hues and clusters whose members
    span hemispheres get an in-between color (i.e., visually flagged as non-regional).
    """
    import matplotlib
    from matplotlib.colors import to_hex

    os.makedirs(output_dir, exist_ok=True)

    if not port_to_idx:
        print("  Skipping comparison plot: port_to_idx not available.")
        return

    nodes_df = pd.read_parquet(nodes_parquet_path, columns=['port_name', 'latitude', 'longitude'])
    nodes_df = nodes_df.drop_duplicates('port_name').set_index('port_name')
    idx_to_port = {v: k for k, v in port_to_idx.items()}

    def build_records(cluster_labels):
        records = []
        for idx in range(len(cluster_labels)):
            port_name = idx_to_port.get(idx)
            if port_name is None or port_name not in nodes_df.index:
                continue
            row = nodes_df.loc[port_name]
            records.append({
                'longitude': float(row['longitude']),
                'latitude': float(row['latitude']),
                'cluster': f"C{cluster_labels[idx]}",
            })
        return pd.DataFrame(records)

    df_a = build_records(cluster_labels_a)
    df_b = build_records(cluster_labels_b)
    if df_a.empty or df_b.empty:
        print("  Skipping comparison plot: empty records.")
        return

    def palette_by_centroid_longitude(df):
        # Median longitude per cluster, ranked, then mapped through a sequential colormap.
        centroids = df.groupby('cluster')['longitude'].median().sort_values()
        clusters_ordered = centroids.index.tolist()
        # matplotlib >= 3.9: colormaps registry replaces cm.get_cmap
        cmap = matplotlib.colormaps['turbo'].resampled(len(clusters_ordered))
        return {
            c: to_hex(cmap(i / max(len(clusters_ordered) - 1, 1)))
            for i, c in enumerate(clusters_ordered)
        }, clusters_ordered

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.0))
    for ax, df, subtitle in zip(axes, (df_a, df_b), (subtitle_a, subtitle_b)):
        palette, hue_order = palette_by_centroid_longitude(df)
        sns.scatterplot(
            data=df, x='longitude', y='latitude', hue='cluster',
            hue_order=hue_order, palette=palette,
            alpha=0.85, s=20, linewidth=0, ax=ax, legend=False,
        )
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect('equal')
        ax.set_title(subtitle, fontsize=13)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, linestyle=':', linewidth=0.4, alpha=0.5)

    if suptitle:
        plt.suptitle(suptitle, fontsize=15)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, output_name),
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_vessel_type_metrics(per_vt_by_framing, output_dir):
    """Grouped bar chart of per-vessel-type link prediction metrics (top/bottom 5 by MRR)."""
    os.makedirs(output_dir, exist_ok=True)
    framings = [f for f in ["dot_product", "mlp", "lstm"] if per_vt_by_framing.get(f)]
    if not framings:
        return

    ref_framing = "dot_product" if "dot_product" in per_vt_by_framing else framings[0]
    ref_data = per_vt_by_framing[ref_framing]
    sorted_vts = sorted(ref_data.items(), key=lambda x: x[1].get("mrr", 0), reverse=True)
    top5 = [vt for vt, _ in sorted_vts[:5]]
    bottom5 = [vt for vt, _ in sorted_vts[-5:]]
    selected_vts = list(dict.fromkeys(top5 + bottom5))

    rows = []
    for vt in selected_vts:
        for framing in framings:
            m = per_vt_by_framing[framing].get(vt, {})
            rows.append({
                'vessel_type': vt,
                'framing': framing,
                'mrr': m.get('mrr', 0),
                'hits_at_1': m.get('hits_at_1', 0),
            })
    plot_df = pd.DataFrame(rows)

    palette = {"dot_product": "#4C72B0", "mlp": "#DD8452", "lstm": "#55A868"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, title in zip(axes, ['mrr', 'hits_at_1'], ['MRR', 'Hits@1']):
        sns.barplot(
            data=plot_df, x='vessel_type', y=metric, hue='framing',
            palette=palette, ax=ax,
        )
        ax.set_title(f'Per-vessel-type {title}')
        ax.set_xlabel('')
        ax.set_ylabel(title)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, 'maritime_vessel_type_link_prediction.pdf'),
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_lstm_sensitivity(results, chosen_max_len, output_dir):
    """Dual-axis line chart: LSTM performance vs max_len window size."""
    os.makedirs(output_dir, exist_ok=True)
    max_lens = [r['max_len'] for r in results]
    mrrs = [r['mrr'] for r in results]
    hits1 = [r['hits_at_1'] for r in results]
    ctx_lens = [r['mean_context_len'] for r in results]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(max_lens, mrrs, color='#4C72B0', marker='o', label='MRR')
    ax1.plot(max_lens, hits1, color='#DD8452', marker='s', linestyle='--', label='Hits@1')
    ax1.axvline(x=chosen_max_len, color='black', linestyle='--', alpha=0.5,
                label=f'chosen={chosen_max_len}')
    ax1.set_xlabel('max_len (LSTM window size)')
    ax1.set_ylabel('Metric value')
    ax1.legend(loc='lower right')

    ax2 = ax1.twinx()
    ax2.plot(max_lens, ctx_lens, color='grey', marker='^', linestyle=':', label='Mean context len')
    ax2.set_ylabel('Mean effective context length', color='grey')
    ax2.tick_params(axis='y', labelcolor='grey')
    ax2.legend(loc='upper left')

    plt.title('LSTM sensitivity to window size (max_len)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_sensitivity_plot.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_lstm_sensitivity_k(results, chosen_k, output_dir):
    """Single-axis line chart: LSTM Hits@1 and MRR vs window length k."""
    os.makedirs(output_dir, exist_ok=True)
    ks = [r['max_len'] for r in results]
    mrrs = [r['mrr'] for r in results]
    hits1 = [r['hits_at_1'] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, mrrs, color='#4C72B0', marker='o', label='MRR')
    ax.plot(ks, hits1, color='#DD8452', marker='s', linestyle='--', label='Hits@1')
    ax.axvline(x=chosen_k, color='black', linestyle='--', alpha=0.5,
               label=f'$k={chosen_k}$')
    ax.set_xlabel('Window length $k$')
    ax.set_ylabel('Metric value')
    ax.legend(loc='lower right')

    plt.title('LSTM sensitivity to window length $k$')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lstm_sensitivity_plot_k.pdf'), bbox_inches='tight')
    plt.close(fig)


def plot_tsne_comparison(df_raw, df_dgi, dataset_str, output_dir='results/plots'):

    sns.set_theme(style="white", context="talk")
    hue_order = sorted(df_raw['label'].unique())
    palette_map = sns.color_palette('husl', len(hue_order))

    fig, axes = plt.subplots(1, 2,
                             figsize=(10, 5),
                             sharex=True,
                             sharey=True,
                             dpi=120,
                             constrained_layout=True)

    # ── Left panel ────────────────────────────────
    sns.scatterplot(
        data=df_raw, x='tsne_1', y='tsne_2', hue='label', hue_order=hue_order,
        palette=palette_map, s=40, alpha=0.85, legend=False, ax=axes[0])
    axes[0].set_title('{}: raw features'.format(dataset_str))
    axes[0].set_xlabel('$T\\!SNE\\text{-}1$')
    axes[0].set_ylabel('$T\\!SNE\\text{-}2$')

 # ── Right panel ───────────────────────────────
    sc = sns.scatterplot(
        data=df_dgi, x='tsne_1', y='tsne_2', hue='label', hue_order=hue_order,
        palette=palette_map, s=40, alpha=0.85, legend='full', ax=axes[1])
    axes[1].set_title('{}: learned DGI'.format(dataset_str))
    axes[1].set_xlabel('$T\\!SNE\\text{-}1$')
    axes[1].set_ylabel('')

 # ── Shared legend ────
    handles, labels = axes[1].get_legend_handles_labels()
    sc.legend_.remove()

    fig.legend(
        handles, labels,
        loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0))

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, '{}_tsne_comparison.pdf'.format(dataset_str)), bbox_inches="tight")
    plt.close(fig)





