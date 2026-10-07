"""Maritime port-visit network preprocessing.

All maritime-specific preprocessing functions. Pure functions taking
DataFrames/arrays and returning arrays.
"""
import json
from typing import Dict, List, Optional, Set, Tuple

import pygeohash as gh
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Node feature column groups
# ---------------------------------------------------------------------------
NODE_LOG_COLS = [
    'total_visits', 'unique_vessels', 'total_arrivals', 'total_departures',
    'out_degree', 'outgoing_journeys', 'in_degree', 'incoming_journeys',
    'total_degree',
]

NODE_STANDARD_COLS = [
    'mean_dwell_time_hours', 'median_dwell_time_hours', 'std_dwell_time_hours',
    'mean_deadweight', 'max_deadweight', 'min_deadweight',
    'mean_draught', 'max_draught', 'mean_vessel_age',
    'net_flow', 'hub_score',
] + NODE_LOG_COLS  # log applied first, then StandardScaler

NODE_PASSTHROUGH_COLS = ['vessel_type_entropy']
NODE_CATEGORICAL_COLS = ['country', 'continent', 'sub_region', 'dominant_vessel_type']
NODE_EXCLUDE_COLS = ['port_name', 'country_name', 'latitude', 'longitude']

# Degree / connectivity / activity-count node features. These are aggregated over
# the full observation window; the exclude_connectivity_features ablation drops
# them to test whether the raw-features baseline's competitiveness depends on
# these full-period aggregate features.
NODE_CONNECTIVITY_COLS = set(NODE_LOG_COLS) | {'net_flow', 'hub_score'}

# ---------------------------------------------------------------------------
# Edge feature column groups
# ---------------------------------------------------------------------------
EDGE_LOG_COLS = ['total_journeys', 'unique_vessels']

EDGE_STANDARD_COLS = [
    'mean_travel_time_hours', 'median_travel_time_hours', 'std_travel_time_hours',
    'mean_deadweight', 'std_deadweight', 'mean_draught_departure',
    'mean_draught_arrival', 'mean_vessel_age', 'num_vessel_types',
] + EDGE_LOG_COLS

EDGE_PASSTHROUGH_COLS = ['vessel_type_entropy']
EDGE_BINARY_COLS = ['pct_international', 'pct_intercontinental']
EDGE_CATEGORICAL_COLS = ['dominant_vessel_type']
EDGE_EXCLUDE_COLS = [
    'origin_port', 'destination_port',
    'origin_country_first', 'destination_country_first',
    'origin_continent_first', 'destination_continent_first',
    'origin_sub_region_first', 'destination_sub_region_first',
]

# ---------------------------------------------------------------------------
# Filtering thresholds
# ---------------------------------------------------------------------------
MIN_JOURNEYS_PER_EDGE = 3
MIN_PORT_DEGREE = 5
WARM_PORT_THRESHOLD = 20

# ---------------------------------------------------------------------------
# Temporal boundaries
# ---------------------------------------------------------------------------
TRAIN_END = '2022-12-31'
VAL_START = '2023-01-01'
VAL_END = '2023-04-30'
TEST_START = '2023-05-01'

# ---------------------------------------------------------------------------
# Geohash
# ---------------------------------------------------------------------------
GEOHASH_LENGTH = 5
GEOHASH_ALPHABET = '0123456789bcdefghjkmnpqrstuvwxyz'  # 32 chars
_CHAR_TO_IDX = {c: i for i, c in enumerate(GEOHASH_ALPHABET)}


# ===================================================================
# Geohash helpers
# ===================================================================

def encode_geohash(lat: float, lon: float, length: int = GEOHASH_LENGTH) -> str:
    """Encode latitude/longitude to a geohash string."""
    return gh.encode(lat, lon, precision=length)


def geohash_to_onehot(geohash_str: str, length: int = GEOHASH_LENGTH) -> np.ndarray:
    """Character-level one-hot encoding of a geohash string.

    Returns:
        Array of shape (length * 32,) with exactly ``length`` ones.
    """
    vec = np.zeros(length * 32, dtype=np.float32)
    for pos, ch in enumerate(geohash_str[:length]):
        idx = _CHAR_TO_IDX.get(ch)
        if idx is not None:
            vec[pos * 32 + idx] = 1.0
    return vec


# ===================================================================
# Node feature preprocessing
# ===================================================================

def sparsify_graph(edges_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Keep only top-K outgoing edges per source node by journey count.

    Retains the top_k edges with the highest total_journeys for each
    origin_port. Used to reduce over-smoothing on dense graphs.

    Args:
        edges_df: Filtered edges DataFrame with 'origin_port' and 'total_journeys'.
        top_k: Maximum number of outgoing edges to keep per source node.

    Returns:
        Sparsified edges DataFrame.
    """
    return (
        edges_df
        .sort_values('total_journeys', ascending=False)
        .groupby('origin_port')
        .head(top_k)
        .reset_index(drop=True)
    )


def preprocess_node_features(
    nodes_df: pd.DataFrame,
    exclude_geo_features: bool = False,
    exclude_connectivity_features: bool = False,
) -> Tuple[np.ndarray, List[str], dict]:
    """Preprocess raw node DataFrame into a feature matrix.

    Steps:
        1. Sort by port_name for deterministic ordering
        2. Impute missing continuous with median, categoricals with "Unknown"
        3. Log-transform skewed counts
        4. StandardScale continuous features
        5. One-hot encode categoricals
        6. Encode lat/lon as geohash one-hot (160 dims)

    If exclude_connectivity_features is True, the degree/connectivity/activity-count
    columns (NODE_CONNECTIVITY_COLS) are dropped from the continuous features. Used
    to test whether the raw-features baseline depends on these full-period aggregates.

    Returns:
        (feature_matrix [N, D], port_names list, preprocessing_info dict)
    """
    df = nodes_df.copy()
    df = df.sort_values('port_name').reset_index(drop=True)
    port_names = df['port_name'].tolist()

    info: Dict = {}

    # Effective continuous-feature column lists (optionally drop connectivity).
    if exclude_connectivity_features:
        log_cols = [c for c in NODE_LOG_COLS if c not in NODE_CONNECTIVITY_COLS]
        standard_cols = [c for c in NODE_STANDARD_COLS if c not in NODE_CONNECTIVITY_COLS]
    else:
        log_cols = list(NODE_LOG_COLS)
        standard_cols = list(NODE_STANDARD_COLS)

    # --- Impute missing values ---------------------------------------------------
    missing_report = {}
    for col in standard_cols + NODE_PASSTHROUGH_COLS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            missing_report[col] = int(n_missing)

    for col in NODE_CATEGORICAL_COLS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna('Unknown')
            missing_report[col] = int(n_missing)

    info['missing_values'] = missing_report

    # --- Log transform -----------------------------------------------------------
    for col in log_cols:
        df[col] = np.log1p(df[col].values.astype(np.float64))

    # --- Standard scaling --------------------------------------------------------
    scaler = StandardScaler()
    standard_vals = scaler.fit_transform(df[standard_cols].values.astype(np.float64))
    info['node_scaler_mean'] = scaler.mean_.tolist()
    info['node_scaler_scale'] = scaler.scale_.tolist()
    info['node_scaler_cols'] = standard_cols
    info['exclude_connectivity_features'] = exclude_connectivity_features

    # --- Passthrough -------------------------------------------------------------
    passthrough_vals = df[NODE_PASSTHROUGH_COLS].values.astype(np.float32)

    # Columns to exclude from one-hot categoricals when exclude_geo_features=True
    _GEO_CAT_COLS = {'country', 'continent', 'sub_region'}

    # --- One-hot categoricals ----------------------------------------------------
    cat_arrays = []
    cat_mappings: Dict[str, List[str]] = {}
    for col in NODE_CATEGORICAL_COLS:
        if exclude_geo_features and col in _GEO_CAT_COLS:
            continue
        categories = sorted(df[col].unique().tolist())
        cat_mappings[col] = categories
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        onehot = np.zeros((len(df), len(categories)), dtype=np.float32)
        for row_i, val in enumerate(df[col]):
            onehot[row_i, cat_to_idx[val]] = 1.0
        cat_arrays.append(onehot)
    info['node_category_mappings'] = cat_mappings
    info['exclude_geo_features'] = exclude_geo_features

    # --- Geohash encoding --------------------------------------------------------
    if not exclude_geo_features:
        geo_vecs = np.zeros((len(df), GEOHASH_LENGTH * 32), dtype=np.float32)
        for i, (_, row) in enumerate(df.iterrows()):
            lat, lon = row['latitude'], row['longitude']
            if pd.notna(lat) and pd.notna(lon):
                gh_str = encode_geohash(float(lat), float(lon), GEOHASH_LENGTH)
                geo_vecs[i] = geohash_to_onehot(gh_str)
    else:
        geo_vecs = None

    # --- Concatenate all ---------------------------------------------------------
    parts = []
    if geo_vecs is not None:
        parts.append(geo_vecs)                     # 160 (skipped if exclude_geo_features)
    parts += [
        standard_vals.astype(np.float32),          # len(NODE_STANDARD_COLS)
        passthrough_vals,                          # len(NODE_PASSTHROUGH_COLS)
    ] + cat_arrays                                 # one-hot categoricals

    feature_matrix = np.hstack(parts)
    info['feature_dim'] = feature_matrix.shape[1]

    return feature_matrix, port_names, info


# ===================================================================
# Edge feature preprocessing
# ===================================================================

def preprocess_edge_features(
    edges_df: pd.DataFrame,
    port_to_idx: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess the raw edge DataFrame into feature matrix and edge_index.

    Returns:
        (edge_features [E, D], edge_index [2, E])
    """
    df = edges_df.copy()

    # Build edge_index using integer port indices
    src_indices = df['origin_port'].map(port_to_idx).values.astype(np.int64)
    dst_indices = df['destination_port'].map(port_to_idx).values.astype(np.int64)
    edge_index = np.stack([src_indices, dst_indices], axis=0)  # (2, E)

    # --- Impute missing continuous values with median ----------------------------
    for col in EDGE_STANDARD_COLS + EDGE_PASSTHROUGH_COLS:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            df[col] = df[col].fillna(df[col].median())

    # --- Log transform -----------------------------------------------------------
    for col in EDGE_LOG_COLS:
        df[col] = np.log1p(df[col].values.astype(np.float64))

    # --- Standard scaling --------------------------------------------------------
    scaler = StandardScaler()
    standard_vals = scaler.fit_transform(df[EDGE_STANDARD_COLS].values.astype(np.float64))

    # --- Passthrough & binary ----------------------------------------------------
    passthrough_vals = df[EDGE_PASSTHROUGH_COLS].values.astype(np.float32)
    binary_vals = df[EDGE_BINARY_COLS].values.astype(np.float32)

    # --- One-hot categoricals ----------------------------------------------------
    cat_arrays = []
    for col in EDGE_CATEGORICAL_COLS:
        categories = sorted(df[col].dropna().unique().tolist())
        cat_to_idx = {c: i for i, c in enumerate(categories)}
        onehot = np.zeros((len(df), len(categories)), dtype=np.float32)
        for row_i, val in enumerate(df[col]):
            if val in cat_to_idx:
                onehot[row_i, cat_to_idx[val]] = 1.0
        cat_arrays.append(onehot)

    # --- Concatenate all ---------------------------------------------------------
    parts = [
        standard_vals.astype(np.float32),
        passthrough_vals,
        binary_vals,
    ] + cat_arrays

    edge_features = np.hstack(parts)
    return edge_features, edge_index


# ===================================================================
# Graph filtering
# ===================================================================

def filter_graph(
    edges_df: pd.DataFrame,
    min_journeys: int = MIN_JOURNEYS_PER_EDGE,
    min_degree: int = MIN_PORT_DEGREE,
) -> Tuple[pd.DataFrame, Set[str]]:
    """Filter edges and ports by minimum journey count and degree.

    1. Remove self-loops
    2. Remove edges with total_journeys < min_journeys
    3. Iteratively remove ports with degree < min_degree until stable

    Returns:
        (filtered_edges_df, surviving_port_names set)
    """
    df = edges_df.copy()
    initial_edges = len(df)

    # Remove self-loops
    df = df[df['origin_port'] != df['destination_port']]

    # Remove low-journey edges
    df = df[df['total_journeys'] >= min_journeys]

    # Iterative degree pruning
    while True:
        # Count degree for each port (in + out)
        out_deg = df['origin_port'].value_counts()
        in_deg = df['destination_port'].value_counts()
        total_deg = out_deg.add(in_deg, fill_value=0)

        low_degree_ports = set(total_deg[total_deg < min_degree].index)
        if not low_degree_ports:
            break

        df = df[
            ~df['origin_port'].isin(low_degree_ports)
            & ~df['destination_port'].isin(low_degree_ports)
        ]

        if len(df) == 0:
            break

    surviving_ports = set(df['origin_port'].unique()) | set(df['destination_port'].unique())

    return df, surviving_ports


# ===================================================================
# Adjacency matrix
# ===================================================================

def build_adjacency_matrix(edge_index: np.ndarray, num_nodes: int) -> sp.csr_matrix:
    """Build binary directed adjacency matrix from edge_index.

    Args:
        edge_index: Shape (2, E) integer array
        num_nodes: Total number of nodes

    Returns:
        Scipy CSR matrix of shape (num_nodes, num_nodes) with binary values.
    """
    rows = edge_index[0]
    cols = edge_index[1]
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    # Ensure binary (in the case of duplicate edges)
    adj.data[:] = 1.0
    return adj


# ===================================================================
# Temporal splits
# ===================================================================

def compute_warm_ports(
    journeys_df: pd.DataFrame,
    train_end: str = TRAIN_END,
    threshold: int = WARM_PORT_THRESHOLD,
) -> Set[str]:
    """Identify warm ports from training-period journeys.

    A port is warm if it appears >= threshold times as origin or destination
    in the training period.
    """
    train_journeys = journeys_df[journeys_df['departure_time'] <= train_end]
    origin_counts = train_journeys['origin_port'].value_counts()
    dest_counts = train_journeys['destination_port'].value_counts()
    total_counts = origin_counts.add(dest_counts, fill_value=0)
    warm = set(total_counts[total_counts >= threshold].index)
    return warm


def build_temporal_splits(
    journeys_df: pd.DataFrame,
    warm_ports: Set[str],
    port_to_idx: Dict[str, int],
) -> dict:
    """Build temporal train/val/test transition sets.

    Journeys are sorted by vessel_id then departure_time to preserve
    vessel trajectory order. Transitions are filtered to warm-port-to-warm-port
    only.

    Returns:
        Dict with keys: train_transitions, val_transitions, test_transitions,
        coverage_stats.  Each transition set is a list of (origin_idx, dest_idx, vessel_id).
    """
    df = journeys_df.copy()
    df = df.sort_values(['vessel_id', 'departure_time']).reset_index(drop=True)

    # Only keep journeys between ports that are in the graph
    valid_ports = set(port_to_idx.keys())
    df = df[
        df['origin_port'].isin(valid_ports)
        & df['destination_port'].isin(valid_ports)
    ]

    # Temporal assignment
    train_mask = df['departure_time'] <= TRAIN_END
    val_mask = (df['departure_time'] >= VAL_START) & (df['departure_time'] <= VAL_END)
    test_mask = df['departure_time'] >= TEST_START

    # Warm port filter
    warm_mask = (
        df['origin_port'].isin(warm_ports)
        & df['destination_port'].isin(warm_ports)
    )

    def _to_transitions(mask):
        subset = df[mask & warm_mask]
        transitions = []
        for _, row in subset.iterrows():
            transitions.append((
                port_to_idx[row['origin_port']],
                port_to_idx[row['destination_port']],
                row['vessel_id'],
            ))
        return transitions

    train_transitions = _to_transitions(train_mask)
    val_transitions = _to_transitions(val_mask)
    test_transitions = _to_transitions(test_mask)

    # Coverage statistics
    total_train = int(train_mask.sum())
    total_val = int(val_mask.sum())
    total_test = int(test_mask.sum())
    warm_train = len(train_transitions)
    warm_val = len(val_transitions)
    warm_test = len(test_transitions)

    coverage_stats = {
        'total_ports': len(valid_ports),
        'warm_ports': len(warm_ports & valid_ports),
        'cold_ports': len(valid_ports) - len(warm_ports & valid_ports),
        'warm_port_coverage': len(warm_ports & valid_ports) / max(len(valid_ports), 1),
        'total_train_transitions': total_train,
        'total_val_transitions': total_val,
        'total_test_transitions': total_test,
        'warm_train_transitions': warm_train,
        'warm_val_transitions': warm_val,
        'warm_test_transitions': warm_test,
        'train_coverage': warm_train / max(total_train, 1),
        'val_coverage': warm_val / max(total_val, 1),
        'test_coverage': warm_test / max(total_test, 1),
    }

    return {
        'train_transitions': train_transitions,
        'val_transitions': val_transitions,
        'test_transitions': test_transitions,
        'coverage_stats': coverage_stats,
    }


# ===================================================================
# Node labels
# ===================================================================

def build_node_labels(
    nodes_df: pd.DataFrame,
    port_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Build classification labels for ports.

    Primary: sub_region encoded as integers.
    Secondary: activity_level (Low/Medium/High from total_visits terciles).

    Args:
        nodes_df: Raw nodes DataFrame (must contain sub_region and total_visits)
        port_names: Ordered list of port names matching feature matrix row order

    Returns:
        (sub_region_labels [N], activity_labels [N], label_info dict)
    """
    df = nodes_df.copy()
    df = df.set_index('port_name').loc[port_names].reset_index()

    # Sub-region labels
    sub_regions = sorted(df['sub_region'].dropna().unique().tolist())
    sr_to_idx = {sr: i for i, sr in enumerate(sub_regions)}
    # Map sub_region strings to indices; unmapped (NaN) become NaN float
    mapped = df['sub_region'].map(sr_to_idx)
    nan_mask = mapped.isna().values
    if nan_mask.any():
        sr_to_idx['Unknown'] = len(sub_regions)
        sub_regions.append('Unknown')
        mapped = mapped.fillna(sr_to_idx['Unknown'])
    sub_region_labels = mapped.values.astype(np.int64)

    # Activity level labels (terciles of total_visits)
    visits = df['total_visits'].values.astype(np.float64)
    t33 = np.percentile(visits, 33.33)
    t67 = np.percentile(visits, 66.67)
    activity_labels = np.zeros(len(df), dtype=np.int64)
    activity_labels[visits >= t67] = 2  # High
    activity_labels[(visits >= t33) & (visits < t67)] = 1  # Medium
    # Low stays 0

    label_info = {
        'sub_region_mapping': sr_to_idx,
        'sub_regions': sub_regions,
        'num_sub_regions': len(sub_regions),
        'activity_levels': ['Low', 'Medium', 'High'],
        'activity_thresholds': {'t33': float(t33), 't67': float(t67)},
    }

    return sub_region_labels, activity_labels, label_info


# ===================================================================
# Graph statistics
# ===================================================================

def print_graph_statistics(
    num_nodes: int,
    num_edges: int,
    adj: sp.spmatrix,
    initial_nodes: int = 0,
    initial_edges: int = 0,
) -> None:
    """Print graph statistics per tech spec Section 1.3."""
    density = num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    avg_degree = num_edges / num_nodes if num_nodes > 0 else 0

    # Connected components (treat as undirected for component count)
    import scipy.sparse.csgraph as csgraph
    n_components, _ = csgraph.connected_components(adj, directed=False)

    print("=" * 60)
    print("Maritime Graph Statistics")
    print("=" * 60)
    print(f"  Nodes (ports):         {num_nodes}")
    print(f"  Edges (routes):        {num_edges}")
    print(f"  Average degree:        {avg_degree:.2f}")
    print(f"  Density:               {density:.6f}")
    print(f"  Connected components:  {n_components}")
    if initial_nodes > 0:
        print(f"  Ports removed:         {initial_nodes - num_nodes}")
    if initial_edges > 0:
        print(f"  Edges removed:         {initial_edges - num_edges}")
    print("=" * 60)


# ===================================================================
# Utility: random split
# ===================================================================

def random_split(
    indices: List[int],
    ratios: List[float],
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split indices into train/val/test by ratios (e.g. [0.8, 0.1, 0.1]).

    Returns:
        (train_indices, val_indices, test_indices)
    """
    rng = np.random.RandomState(seed)
    indices = list(indices)
    rng.shuffle(indices)

    n = len(indices)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    return indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]
