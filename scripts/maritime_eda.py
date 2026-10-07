"""Maritime dataset exploratory data analysis.

Loads the cached maritime dataset and raw parquet files, produces
publication-quality statistics and figures for the paper.

Usage:
    python scripts/maritime_eda.py

Outputs:
    - Console: all statistics
    - results/maritime/plots/*.pdf: 10 publication-quality figures
    - results/maritime/eda_summary.txt: all statistics in text form
"""
import json
import os
import sys
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.maritime import MaritimeDataset
from src.preprocessing.maritime import (
    filter_graph, NODE_STANDARD_COLS, NODE_PASSTHROUGH_COLS,
    NODE_CATEGORICAL_COLS, NODE_EXCLUDE_COLS,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'maritime')
CACHE_DIR = os.path.join(DATA_DIR, 'processed')
PLOT_DIR = os.path.join(PROJECT_ROOT, 'results', 'maritime', 'plots')
SUMMARY_PATH = os.path.join(PROJECT_ROOT, 'results', 'maritime', 'eda_summary.txt')

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
sns.set_theme(style='whitegrid', context='paper', font_scale=1.1)
FIGSIZE_SINGLE = (5, 4)
FIGSIZE_WIDE = (10, 4)
DPI = 300


def save_fig(fig, filename):
    """Save the figure to PLOT_DIR as PDF."""
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, bbox_inches='tight', dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {filename}")


# ===================================================================
# Data loading
# ===================================================================

def load_data():
    """Load cached dataset, raw dataframes, and metadata."""
    # Cached processed dataset
    dataset = MaritimeDataset(data_dir=DATA_DIR).load()

    # Raw parquet files (for columns not in cache)
    nodes_df = pd.read_parquet(os.path.join(DATA_DIR, 'nodes.parquet'))
    edges_df = pd.read_parquet(os.path.join(DATA_DIR, 'edges.parquet'))

    # Filter edges to match the processed graph
    filtered_edges, surviving_ports = filter_graph(edges_df)
    nodes_df = nodes_df[nodes_df['port_name'].isin(surviving_ports)].copy()

    # Metadata
    with open(os.path.join(CACHE_DIR, 'metadata.json'), 'r') as f:
        metadata = json.load(f)

    return dataset, nodes_df, filtered_edges, metadata


# ===================================================================
# Statistics functions
# ===================================================================

def graph_structure_stats(dataset, nodes_df, filtered_edges, metadata, out):
    """Compute and print graph structure statistics."""
    out.write("=" * 70 + "\n")
    out.write("1. GRAPH STRUCTURE\n")
    out.write("=" * 70 + "\n")

    N = dataset.features.shape[0]
    E = dataset.edge_index.shape[1]
    density = E / (N * (N - 1))
    adj = dataset.adj

    # Degree statistics from adjacency
    out_deg = np.array(adj.sum(axis=1)).ravel()
    in_deg = np.array(adj.sum(axis=0)).ravel()
    total_deg = out_deg + in_deg

    # Connected components
    import scipy.sparse.csgraph as csgraph
    n_comp, comp_labels = csgraph.connected_components(adj, directed=False)
    comp_sizes = np.bincount(comp_labels)

    # Reciprocity: fraction of edges (i->j) where (j->i) also exists
    adj_dense = adj.toarray() if sp.issparse(adj) else adj
    reciprocal = np.sum((adj_dense > 0) & (adj_dense.T > 0))
    reciprocity = reciprocal / max(E, 1)

    # Filtering impact
    initial_nodes = 4325  # from schema
    initial_edges = 270272

    lines = [
        f"  Nodes (ports):            {N}",
        f"  Edges (directed routes):  {E}",
        f"  Density:                  {density:.6f}",
        f"  Avg total degree:         {total_deg.mean():.2f}",
        f"  Median total degree:      {np.median(total_deg):.0f}",
        f"  Max total degree:         {total_deg.max():.0f}",
        f"  Avg in-degree:            {in_deg.mean():.2f}",
        f"  Avg out-degree:           {out_deg.mean():.2f}",
        f"  Max in-degree:            {in_deg.max():.0f}",
        f"  Max out-degree:           {out_deg.max():.0f}",
        f"  Reciprocity:              {reciprocity:.4f}",
        f"  Connected components:     {n_comp}",
        f"  Largest component:        {comp_sizes.max()} nodes",
        f"  Filtering impact:",
        f"    Ports removed:          {initial_nodes - N} ({100*(initial_nodes-N)/initial_nodes:.1f}%)",
        f"    Edges removed:          {initial_edges - E} ({100*(initial_edges-E)/initial_edges:.1f}%)",
    ]
    for line in lines:
        out.write(line + "\n")
        print(line)

    return {
        'N': N, 'E': E, 'density': density,
        'in_deg': in_deg, 'out_deg': out_deg, 'total_deg': total_deg,
        'reciprocity': reciprocity, 'n_comp': n_comp,
    }


def feature_dim_stats(dataset, metadata, out):
    """Print feature dimension breakdown."""
    out.write("\n" + "=" * 70 + "\n")
    out.write("2. FEATURE DIMENSIONS\n")
    out.write("=" * 70 + "\n")

    info = metadata['preprocessing_info']
    cat_maps = info['node_category_mappings']

    n_standard = len(info['node_scaler_cols'])
    n_passthrough = len(NODE_PASSTHROUGH_COLS)
    n_geohash = 5 * 32  # 160

    cat_dims = {col: len(cats) for col, cats in cat_maps.items()}
    n_cat_total = sum(cat_dims.values())
    total = n_geohash + n_standard + n_passthrough + n_cat_total

    lines = [
        f"  Node features total:      {dataset.features.shape[1]}",
        f"    Geohash one-hot:        {n_geohash}",
        f"    Standard-scaled:        {n_standard} ({', '.join(info['node_scaler_cols'][:5])}...)",
        f"    Passthrough:            {n_passthrough} ({', '.join(NODE_PASSTHROUGH_COLS)})",
    ]
    for col, n in cat_dims.items():
        lines.append(f"    One-hot {col:25s}: {n}")
    lines.append(f"    Computed total:         {total}")
    lines.append(f"  Edge features total:      {dataset.edge_features.shape[1]}")

    for line in lines:
        out.write(line + "\n")
        print(line)


def geographic_stats(nodes_df, metadata, out):
    """Geographic distribution statistics."""
    out.write("\n" + "=" * 70 + "\n")
    out.write("3. GEOGRAPHIC DISTRIBUTION\n")
    out.write("=" * 70 + "\n")

    # Continent
    continent_counts = nodes_df['continent'].value_counts().sort_values(ascending=False)
    lines = ["  Ports per continent:"]
    for cont, cnt in continent_counts.items():
        lines.append(f"    {cont:25s}: {cnt:4d} ({100*cnt/len(nodes_df):5.1f}%)")

    # Subregion (top 10)
    sr_counts = nodes_df['sub_region'].value_counts().sort_values(ascending=False)
    lines.append("  Ports per sub-region (top 10):")
    for sr, cnt in sr_counts.head(10).items():
        lines.append(f"    {sr:30s}: {cnt:4d} ({100*cnt/len(nodes_df):5.1f}%)")

    # Top countries
    country_counts = nodes_df['country_name'].value_counts().sort_values(ascending=False)
    lines.append("  Top 10 countries:")
    for c, cnt in country_counts.head(10).items():
        lines.append(f"    {c:30s}: {cnt:4d}")

    for line in lines:
        out.write(line + "\n")
        print(line)

    return continent_counts, sr_counts


def edge_weight_stats(filtered_edges, out):
    """Edge weight (journey count) statistics."""
    out.write("\n" + "=" * 70 + "\n")
    out.write("4. EDGE WEIGHTS (JOURNEY COUNTS)\n")
    out.write("=" * 70 + "\n")

    journeys = filtered_edges['total_journeys']
    pcts = [5, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(journeys, pcts)

    lines = [
        f"  Mean journeys/edge:       {journeys.mean():.1f}",
        f"  Std journeys/edge:        {journeys.std():.1f}",
    ]
    for p, v in zip(pcts, pct_vals):
        lines.append(f"  P{p:2d}:                      {v:.0f}")
    lines.append(f"  Max:                      {journeys.max():.0f}")

    # International / intercontinental
    intl_frac = (filtered_edges['pct_international'] > 0.5).mean()
    interc_frac = (filtered_edges['pct_intercontinental'] > 0.5).mean()
    lines.append(f"  International edges:      {intl_frac:.1%}")
    lines.append(f"  Intercontinental edges:   {interc_frac:.1%}")

    # Top 5 routes by journey count
    top_routes = filtered_edges.nlargest(5, 'total_journeys')[
        ['origin_port', 'destination_port', 'total_journeys']
    ]
    lines.append("  Top 5 routes by journey count:")
    for _, row in top_routes.iterrows():
        lines.append(f"    {row['origin_port']:25s} -> {row['destination_port']:25s}: {int(row['total_journeys']):,}")

    for line in lines:
        out.write(line + "\n")
        print(line)


def label_stats(dataset, metadata, out):
    """Label distribution statistics."""
    out.write("\n" + "=" * 70 + "\n")
    out.write("5. LABEL DISTRIBUTIONS\n")
    out.write("=" * 70 + "\n")

    label_info = metadata['label_info']
    sub_regions = label_info['sub_regions']
    sr_mapping = label_info['sub_region_mapping']
    idx_to_sr = {v: k for k, v in sr_mapping.items()}

    labels = dataset.labels
    unique, counts = np.unique(labels, return_counts=True)

    lines = ["  Sub-region classes:"]
    for idx, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        name = idx_to_sr.get(int(idx), f"Class {idx}")
        lines.append(f"    {name:30s}: {cnt:4d} ({100*cnt/len(labels):5.1f}%)")
    lines.append(f"  Total sub-region classes: {len(unique)}")

    # Imbalance ratio
    lines.append(f"  Class imbalance (max/min): {counts.max()}/{counts.min()} = {counts.max()/counts.min():.1f}x")

    # Activity labels
    act = dataset.activity_labels
    act_names = ['Low', 'Medium', 'High']
    act_unique, act_counts = np.unique(act, return_counts=True)
    lines.append("  Activity levels:")
    for idx, cnt in zip(act_unique, act_counts):
        lines.append(f"    {act_names[idx]:10s}: {cnt:4d} ({100*cnt/len(act):5.1f}%)")
    thresholds = label_info['activity_thresholds']
    lines.append(f"  Thresholds: Low < {thresholds['t33']:.0f} visits, High >= {thresholds['t67']:.0f} visits")

    for line in lines:
        out.write(line + "\n")
        print(line)


def temporal_stats(metadata, out):
    """Temporal split statistics."""
    out.write("\n" + "=" * 70 + "\n")
    out.write("6. TEMPORAL SPLITS\n")
    out.write("=" * 70 + "\n")

    cs = metadata['coverage_stats']
    lines = [
        f"  Total ports in graph:     {cs['total_ports']}",
        f"  Warm ports (>=20 trans.):  {cs['warm_ports']} ({100*cs['warm_port_coverage']:.1f}%)",
        f"  Cold ports:               {cs['cold_ports']}",
        f"  Train transitions (total): {cs['total_train_transitions']:,}",
        f"  Val transitions (total):   {cs['total_val_transitions']:,}",
        f"  Test transitions (total):  {cs['total_test_transitions']:,}",
        f"  Warm train transitions:    {cs['warm_train_transitions']:,}",
        f"  Warm val transitions:      {cs['warm_val_transitions']:,}",
        f"  Warm test transitions:     {cs['warm_test_transitions']:,}",
        f"  Train coverage:            {cs['train_coverage']:.4f}",
        f"  Val coverage:              {cs['val_coverage']:.4f}",
        f"  Test coverage:             {cs['test_coverage']:.4f}",
    ]

    for line in lines:
        out.write(line + "\n")
        print(line)


def split_stats(dataset, out):
    """Classification split statistics."""
    out.write("\n" + "=" * 70 + "\n")
    out.write("7. CLASSIFICATION SPLITS (WARM PORTS, 80/10/10)\n")
    out.write("=" * 70 + "\n")

    N = dataset.features.shape[0]
    lines = [
        f"  Train: {len(dataset.idx_train)} ({100*len(dataset.idx_train)/N:.1f}%)",
        f"  Val:   {len(dataset.idx_val)} ({100*len(dataset.idx_val)/N:.1f}%)",
        f"  Test:  {len(dataset.idx_test)} ({100*len(dataset.idx_test)/N:.1f}%)",
        f"  Total: {len(dataset.idx_train)+len(dataset.idx_val)+len(dataset.idx_test)} / {N}",
    ]

    for line in lines:
        out.write(line + "\n")
        print(line)


def top_ports_stats(nodes_df, dataset, metadata, out):
    """Top ports by degree."""
    out.write("\n" + "=" * 70 + "\n")
    out.write("8. TOP 20 PORTS BY TOTAL DEGREE\n")
    out.write("=" * 70 + "\n")

    adj = dataset.adj
    out_deg = np.array(adj.sum(axis=1)).ravel()
    in_deg = np.array(adj.sum(axis=0)).ravel()
    total_deg = out_deg + in_deg

    port_to_idx = metadata['port_to_idx']
    idx_to_port = {v: k for k, v in port_to_idx.items()}

    top_indices = np.argsort(total_deg)[::-1][:20]
    lines = [f"  {'Port':35s} {'In':>5s} {'Out':>5s} {'Total':>6s}"]
    lines.append("  " + "-" * 55)
    for idx in top_indices:
        name = idx_to_port[idx]
        lines.append(f"  {name:35s} {int(in_deg[idx]):5d} {int(out_deg[idx]):5d} {int(total_deg[idx]):6d}")

    for line in lines:
        out.write(line + "\n")
        print(line)

    return top_indices, idx_to_port, total_deg, in_deg, out_deg


# ===================================================================
# Plot functions
# ===================================================================

def plot_degree_distributions(in_deg, out_deg, total_deg):
    """Plot 1: In/out/total degree histograms (log-log)."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    for ax, deg, title in zip(axes, [in_deg, out_deg, total_deg],
                               ['In-degree', 'Out-degree', 'Total degree']):
        # Use log-spaced bins
        bins = np.logspace(np.log10(max(deg.min(), 1)), np.log10(deg.max()), 40)
        ax.hist(deg, bins=bins, color='steelblue', edgecolor='white', linewidth=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(title)
        ax.set_ylabel('Count')
        ax.set_title(title)

    fig.tight_layout()
    save_fig(fig, 'degree_distributions.pdf')


def plot_degree_in_vs_out(nodes_df, in_deg, out_deg, metadata):
    """Plot 2: In-degree vs. out-degree scatter, colored by the continent."""
    port_to_idx = metadata['port_to_idx']
    df_sorted = nodes_df.sort_values('port_name').reset_index(drop=True)

    # Map continents to indices
    continents = []
    for _, row in df_sorted.iterrows():
        name = row['port_name']
        if name in port_to_idx:
            continents.append(row.get('continent') or 'Unknown')

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    scatter_df = pd.DataFrame({
        'In-degree': in_deg, 'Out-degree': out_deg, 'Continent': continents
    })
    sns.scatterplot(
        data=scatter_df, x='In-degree', y='Out-degree', hue='Continent',
        palette='Set2', s=15, alpha=0.7, ax=ax, legend='brief',
    )
    # Diagonal reference line
    max_val = max(in_deg.max(), out_deg.max())
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=0.8)
    ax.set_title('In-degree vs Out-degree')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.8)
    fig.tight_layout()
    save_fig(fig, 'degree_in_vs_out.pdf')


def plot_geographic_distribution(continent_counts, sr_counts):
    """Plot 3: Ports per continent + sub_region bar charts."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Continent bar chart
    continent_counts.plot.bar(ax=axes[0], color='steelblue', edgecolor='white')
    axes[0].set_title('Ports per Continent')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Number of ports')
    axes[0].tick_params(axis='x', rotation=30)

    # Subregion (top 15)
    sr_counts.head(15).plot.barh(ax=axes[1], color='steelblue', edgecolor='white')
    axes[1].set_title('Ports per Sub-region (top 15)')
    axes[1].set_xlabel('Number of ports')
    axes[1].invert_yaxis()

    fig.tight_layout()
    save_fig(fig, 'geographic_distribution.pdf')


def plot_port_map(nodes_df, total_deg, metadata):
    """Plot 4: Lat/lon scatter, size=degree, color=continent."""
    port_to_idx = metadata['port_to_idx']

    # Build plotting dataframe
    rows = []
    for _, row in nodes_df.iterrows():
        name = row['port_name']
        if name in port_to_idx:
            idx = port_to_idx[name]
            rows.append({
                'lat': row['latitude'],
                'lon': row['longitude'],
                'degree': total_deg[idx],
                'continent': row.get('continent') or 'Unknown',
            })
    map_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    scatter = ax.scatter(
        map_df['lon'], map_df['lat'],
        s=np.clip(map_df['degree'] * 0.3, 3, 80),
        c=pd.Categorical(map_df['continent']).codes,
        cmap='Set2', alpha=0.6, edgecolors='none',
    )
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Global Port Network (size = degree)')

    # Custom legend for continents
    map_df['continent'] = map_df['continent'].fillna('Unknown')
    continents = sorted(map_df['continent'].unique())
    cmap = plt.cm.Set2
    handles = [plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=cmap(i / max(len(continents) - 1, 1)),
               markersize=6, label=c)
               for i, c in enumerate(continents)]
    ax.legend(handles=handles, fontsize=7, loc='lower left', framealpha=0.8)

    fig.tight_layout()
    save_fig(fig, 'port_map.pdf')


def plot_edge_weight_distribution(filtered_edges):
    """Plot 5: Journey count histogram (log scale)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    journeys = filtered_edges['total_journeys'].values
    bins = np.logspace(np.log10(journeys.min()), np.log10(journeys.max()), 50)
    ax.hist(journeys, bins=bins, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Journey count per edge')
    ax.set_ylabel('Number of edges')
    ax.set_title('Edge Weight Distribution')
    fig.tight_layout()
    save_fig(fig, 'edge_weight_distribution.pdf')


def plot_node_feature_correlation(nodes_df):
    """Plot 6: Correlation heatmap of continuous node features."""
    # Select continuous columns that exist in filtered nodes
    cont_cols = [c for c in NODE_STANDARD_COLS + NODE_PASSTHROUGH_COLS
                 if c in nodes_df.columns]
    corr = nodes_df[cont_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        corr, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.3,
        xticklabels=True, yticklabels=True,
        cbar_kws={'shrink': 0.7},
    )
    ax.set_title('Node Feature Correlation')
    ax.tick_params(axis='both', labelsize=7)
    fig.tight_layout()
    save_fig(fig, 'node_feature_correlation.pdf')


def plot_label_distributions(dataset, metadata):
    """Plot 7: Sub-region and activity level bar charts."""
    label_info = metadata['label_info']
    sr_mapping = label_info['sub_region_mapping']
    idx_to_sr = {v: k for k, v in sr_mapping.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Sub-region
    labels = dataset.labels
    unique, counts = np.unique(labels, return_counts=True)
    names = [idx_to_sr.get(int(u), f'Class {u}') for u in unique]
    order = np.argsort(counts)[::-1]
    axes[0].barh(
        [names[i] for i in order], [counts[i] for i in order],
        color='steelblue', edgecolor='white'
    )
    axes[0].set_xlabel('Number of ports')
    axes[0].set_title('Sub-region Distribution')
    axes[0].invert_yaxis()
    axes[0].tick_params(axis='y', labelsize=7)

    # Activity level
    act = dataset.activity_labels
    act_names = ['Low', 'Medium', 'High']
    act_unique, act_counts = np.unique(act, return_counts=True)
    axes[1].bar(
        [act_names[i] for i in act_unique], act_counts,
        color=['#3498db', '#f39c12', '#e74c3c'], edgecolor='white'
    )
    axes[1].set_ylabel('Number of ports')
    axes[1].set_title('Activity Level Distribution')

    fig.tight_layout()
    save_fig(fig, 'label_distributions.pdf')


def plot_top_ports(top_indices, idx_to_port, total_deg, in_deg, out_deg):
    """Plot 8: Top 20 ports horizontal bar chart."""
    names = [idx_to_port[i] for i in top_indices]
    degrees = total_deg[top_indices]

    fig, ax = plt.subplots(figsize=(6, 5))
    y_pos = np.arange(len(names))
    in_vals = in_deg[top_indices]
    out_vals = out_deg[top_indices]

    ax.barh(y_pos, in_vals, color='steelblue', label='In-degree', edgecolor='white')
    ax.barh(y_pos, out_vals, left=in_vals, color='coral', label='Out-degree', edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Degree')
    ax.set_title('Top 20 Ports by Total Degree')
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, 'top_ports_degree.pdf')


def plot_tsne(dataset, metadata, label_type='sub_region'):
    """Plot 9/10: t-SNE of raw features."""
    features = dataset.features
    if sp.issparse(features):
        features = features.toarray()

    print(f"  Computing t-SNE ({label_type})...")
    tsne = TSNE(perplexity=30, n_components=2, init='pca', max_iter=2500, random_state=42)
    embed = tsne.fit_transform(features)

    label_info = metadata['label_info']

    if label_type == 'sub_region':
        labels = dataset.labels
        sr_mapping = label_info['sub_region_mapping']
        idx_to_name = {v: k for k, v in sr_mapping.items()}
        label_names = [idx_to_name.get(int(l), f'Class {l}') for l in labels]
        filename = 'maritime_raw_tsne_subregion.pdf'
        title = 'Raw Features t-SNE (sub-region)'
    else:
        labels = dataset.activity_labels
        act_names = {0: 'Low', 1: 'Medium', 2: 'High'}
        label_names = [act_names[int(l)] for l in labels]
        filename = 'maritime_raw_tsne_activity.pdf'
        title = 'Raw Features t-SNE (activity level)'

    df = pd.DataFrame({'tsne_1': embed[:, 0], 'tsne_2': embed[:, 1], 'label': label_names})

    fig, ax = plt.subplots(figsize=(6, 5))
    n_classes = len(set(label_names))
    palette = 'husl' if n_classes > 10 else 'Set2'

    if label_type == 'sub_region':
        # Too many classes for inline legend — use no legend or abbreviated legend
        sns.scatterplot(
            data=df, x='tsne_1', y='tsne_2', hue='label',
            palette=palette, s=10, alpha=0.7, ax=ax, legend=False,
        )
    else:
        sns.scatterplot(
            data=df, x='tsne_1', y='tsne_2', hue='label',
            hue_order=['Low', 'Medium', 'High'],
            palette=['#3498db', '#f39c12', '#e74c3c'],
            s=10, alpha=0.7, ax=ax, legend='brief',
        )
        ax.legend(fontsize=8, loc='upper right')

    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title)
    fig.tight_layout()
    save_fig(fig, filename)


# ===================================================================
# Main
# ===================================================================

def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("Loading data...")
    dataset, nodes_df, filtered_edges, metadata = load_data()

    # Open the summary file
    out = open(SUMMARY_PATH, 'w', encoding='utf-8')
    out.write("Maritime Port-Visit Network - EDA Summary\n")
    out.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n")

    # --- Statistics ---
    print("\n" + "=" * 70)
    print("MARITIME PORT-VISIT NETWORK - EDA")
    print("=" * 70)

    stats = graph_structure_stats(dataset, nodes_df, filtered_edges, metadata, out)
    feature_dim_stats(dataset, metadata, out)
    continent_counts, sr_counts = geographic_stats(nodes_df, metadata, out)
    edge_weight_stats(filtered_edges, out)
    label_stats(dataset, metadata, out)
    temporal_stats(metadata, out)
    split_stats(dataset, out)
    top_info = top_ports_stats(nodes_df, dataset, metadata, out)
    top_indices, idx_to_port, total_deg, in_deg, out_deg = top_info

    out.close()
    print(f"\nSummary written to: {SUMMARY_PATH}")

    # --- Plots ---
    print("\nGenerating plots...")

    plot_degree_distributions(in_deg, out_deg, total_deg)
    plot_degree_in_vs_out(nodes_df, in_deg, out_deg, metadata)
    plot_geographic_distribution(continent_counts, sr_counts)
    plot_port_map(nodes_df, total_deg, metadata)
    plot_edge_weight_distribution(filtered_edges)
    plot_node_feature_correlation(nodes_df)
    plot_label_distributions(dataset, metadata)
    plot_top_ports(top_indices, idx_to_port, total_deg, in_deg, out_deg)
    plot_tsne(dataset, metadata, label_type='sub_region')
    plot_tsne(dataset, metadata, label_type='activity')

    print(f"\nAll plots saved to: {PLOT_DIR}")
    print("Done!")


if __name__ == '__main__':
    main()
