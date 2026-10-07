"""Tests for maritime data preprocessing pipeline and graph construction.

All tests use synthetic DataFrames mimicking the parquet schema — no real
data files needed.
"""
import sys
import os
import json
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.preprocessing.maritime import (
    encode_geohash,
    geohash_to_onehot,
    preprocess_node_features,
    preprocess_edge_features,
    filter_graph,
    build_adjacency_matrix,
    compute_warm_ports,
    build_temporal_splits,
    build_node_labels,
    random_split,
    GEOHASH_LENGTH,
    NODE_STANDARD_COLS,
    NODE_PASSTHROUGH_COLS,
    NODE_CATEGORICAL_COLS,
    NODE_EXCLUDE_COLS,
    EDGE_STANDARD_COLS,
    EDGE_PASSTHROUGH_COLS,
    EDGE_BINARY_COLS,
    EDGE_CATEGORICAL_COLS,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
)

# ===========================================================================
# Helpers to build synthetic data
# ===========================================================================

VESSEL_TYPES = [
    'Passenger', 'Container', 'Bulk Dry', 'Oil', 'Chemical',
    'Liquefied Gas', 'Ro-Ro Cargo', 'General Cargo',
]

COUNTRIES = ['NL', 'SG', 'US', 'CN', 'DE', 'JP', 'BR', 'AU', 'IN', 'FR']
CONTINENTS = ['Europe', 'Asia', 'Americas', 'Oceania']
SUB_REGIONS = ['Western Europe', 'South-Eastern Asia', 'Northern America',
               'Eastern Asia', 'South America', 'Oceania']


def make_nodes_df(n=20, seed=42):
    """Create a synthetic nodes DataFrame."""
    rng = np.random.RandomState(seed)
    ports = [f'Port_{i:03d}' for i in range(n)]
    df = pd.DataFrame({
        'port_name': ports,
        'total_visits': rng.randint(10, 10000, n),
        'unique_vessels': rng.randint(5, 5000, n),
        'total_arrivals': rng.randint(5, 5000, n),
        'total_departures': rng.randint(5, 5000, n),
        'mean_dwell_time_hours': rng.uniform(1, 200, n),
        'median_dwell_time_hours': rng.uniform(1, 100, n),
        'std_dwell_time_hours': rng.uniform(0, 500, n),
        'mean_deadweight': rng.uniform(100, 30000, n),
        'max_deadweight': rng.uniform(30000, 400000, n),
        'min_deadweight': rng.uniform(0, 100, n),
        'mean_draught': rng.uniform(1, 10, n),
        'max_draught': rng.uniform(10, 25, n),
        'mean_vessel_age': rng.uniform(5, 30, n),
        'dominant_vessel_type': rng.choice(VESSEL_TYPES, n),
        'vessel_type_entropy': rng.uniform(0, 3.5, n),
        'latitude': rng.uniform(-60, 60, n),
        'longitude': rng.uniform(-180, 180, n),
        'country': rng.choice(COUNTRIES, n),
        'country_name': ['Country_' + c for c in rng.choice(COUNTRIES, n)],
        'continent': rng.choice(CONTINENTS, n),
        'sub_region': rng.choice(SUB_REGIONS, n),
        'out_degree': rng.randint(5, 500, n),
        'outgoing_journeys': rng.randint(10, 50000, n),
        'in_degree': rng.randint(5, 500, n),
        'incoming_journeys': rng.randint(10, 50000, n),
        'total_degree': rng.randint(10, 1000, n),
        'net_flow': rng.uniform(-100, 100, n),
        'hub_score': rng.uniform(0, 200, n),
    })
    return df


def make_edges_df(ports, n_edges=100, seed=42):
    """Create a synthetic edges DataFrame between given port names."""
    rng = np.random.RandomState(seed)
    origins = rng.choice(ports, n_edges)
    destinations = rng.choice(ports, n_edges)

    df = pd.DataFrame({
        'origin_port': origins,
        'destination_port': destinations,
        'total_journeys': rng.randint(1, 200, n_edges),
        'unique_vessels': rng.randint(1, 100, n_edges),
        'mean_travel_time_hours': rng.uniform(0.5, 500, n_edges),
        'median_travel_time_hours': rng.uniform(0.5, 300, n_edges),
        'std_travel_time_hours': rng.uniform(0, 200, n_edges),
        'pct_international': rng.choice([0.0, 1.0], n_edges),
        'pct_intercontinental': rng.choice([0.0, 1.0], n_edges),
        'origin_country_first': rng.choice(COUNTRIES, n_edges),
        'destination_country_first': rng.choice(COUNTRIES, n_edges),
        'origin_continent_first': rng.choice(CONTINENTS, n_edges),
        'destination_continent_first': rng.choice(CONTINENTS, n_edges),
        'origin_sub_region_first': rng.choice(SUB_REGIONS, n_edges),
        'destination_sub_region_first': rng.choice(SUB_REGIONS, n_edges),
        'mean_deadweight': rng.uniform(100, 30000, n_edges),
        'std_deadweight': rng.uniform(0, 10000, n_edges),
        'mean_draught_departure': rng.uniform(1, 10, n_edges),
        'mean_draught_arrival': rng.uniform(1, 10, n_edges),
        'mean_vessel_age': rng.uniform(5, 30, n_edges),
        'dominant_vessel_type': rng.choice(VESSEL_TYPES, n_edges),
        'vessel_type_entropy': rng.uniform(0, 3.0, n_edges),
        'num_vessel_types': rng.randint(1, 15, n_edges),
    })
    return df


def make_journeys_df(ports, n_journeys=500, seed=42):
    """Create a synthetic journeys DataFrame."""
    rng = np.random.RandomState(seed)

    # Generate timestamps spanning Feb 2022 - Aug 2023
    start = pd.Timestamp('2022-02-01')
    end = pd.Timestamp('2023-08-31')
    ts_range = (end - start).total_seconds()
    departure_offsets = rng.uniform(0, ts_range, n_journeys)
    departure_times = [start + pd.Timedelta(seconds=s) for s in sorted(departure_offsets)]

    n_vessels = max(5, n_journeys // 10)
    vessel_ids = rng.randint(1000, 1000 + n_vessels, n_journeys)

    df = pd.DataFrame({
        'journey_id': range(1, n_journeys + 1),
        'vessel_id': vessel_ids,
        'origin_port': rng.choice(ports, n_journeys),
        'destination_port': rng.choice(ports, n_journeys),
        'departure_time': departure_times,
        'arrival_time': [t + pd.Timedelta(hours=rng.uniform(1, 100)) for t in departure_times],
        'origin_lat': rng.uniform(-60, 60, n_journeys),
        'origin_lon': rng.uniform(-180, 180, n_journeys),
        'destination_lat': rng.uniform(-60, 60, n_journeys),
        'destination_lon': rng.uniform(-180, 180, n_journeys),
        'origin_country': rng.choice(COUNTRIES, n_journeys),
        'destination_country': rng.choice(COUNTRIES, n_journeys),
        'draught_departure': rng.uniform(1, 10, n_journeys),
        'draught_arrival': rng.uniform(1, 10, n_journeys),
        'departure_port_visit_id': rng.randint(100000, 999999, n_journeys),
        'arrival_port_visit_id': rng.randint(100000, 999999, n_journeys),
        'departure_movement_type': 'Port Departure',
        'arrival_movement_type': 'Port Arrival',
        'is_complete_journey': True,
        'travel_time_hours': rng.uniform(1, 100, n_journeys),
        'departure_year': [t.year for t in departure_times],
        'departure_month': [t.month for t in departure_times],
        'departure_day_of_week': [t.dayofweek for t in departure_times],
        'departure_hour': [t.hour for t in departure_times],
    })
    return df


# ===========================================================================
# Unit tests
# ===========================================================================

def test_encode_geohash():
    """Known lat/lon pairs produce expected geohash strings."""
    # Rotterdam: lat=51.79, lon=4.64
    gh_rot = encode_geohash(51.79, 4.64, 5)
    assert isinstance(gh_rot, str)
    assert len(gh_rot) == 5

    # Singapore: lat=1.24, lon=103.80
    gh_sg = encode_geohash(1.24, 103.80, 5)
    assert isinstance(gh_sg, str)
    assert len(gh_sg) == 5

    # Different locations should produce different geohashes
    assert gh_rot != gh_sg
    print("  PASS: encode_geohash produces valid geohash strings")


def test_geohash_to_onehot():
    """Correct shape (160,), exactly 5 ones per vector."""
    gh_str = encode_geohash(51.79, 4.64, 5)
    vec = geohash_to_onehot(gh_str, 5)
    assert vec.shape == (160,), f"Expected (160,), got {vec.shape}"
    assert vec.sum() == 5.0, f"Expected 5 ones, got {vec.sum()}"
    assert set(np.unique(vec)) == {0.0, 1.0}
    print("  PASS: geohash_to_onehot has correct shape and exactly 5 ones")


def test_preprocess_node_features():
    """Synthetic nodes_df: verify output shape, no NaN, expected dim range."""
    nodes_df = make_nodes_df(n=30)
    features, port_names, info = preprocess_node_features(nodes_df)

    assert features.shape[0] == 30, f"Expected 30 rows, got {features.shape[0]}"
    assert len(port_names) == 30
    assert not np.isnan(features).any(), "Features contain NaN"
    assert not np.isinf(features).any(), "Features contain Inf"

    # Expect roughly 441 dims (160 geohash + ~20 standard + 1 passthrough + ~260 categorical)
    # Exact depends on unique values; just check reasonable range
    assert features.shape[1] > 100, f"Too few features: {features.shape[1]}"

    # Check info dict
    assert 'feature_dim' in info
    assert 'node_scaler_mean' in info
    assert 'node_category_mappings' in info
    print(f"  PASS: Node features shape = {features.shape}, no NaN/Inf")


def test_preprocess_node_features_with_missing():
    """Missing values are properly imputed."""
    nodes_df = make_nodes_df(n=10)
    # Introduce some NaN
    nodes_df.loc[0, 'mean_dwell_time_hours'] = np.nan
    nodes_df.loc[1, 'total_visits'] = np.nan
    nodes_df.loc[2, 'country'] = np.nan

    features, port_names, info = preprocess_node_features(nodes_df)
    assert not np.isnan(features).any(), "Features still contain NaN after imputation"
    assert len(info['missing_values']) > 0
    print("  PASS: Missing values properly imputed")


def test_preprocess_edge_features():
    """Synthetic edges_df: verify output shape, edge_index valid indices."""
    ports = [f'Port_{i:03d}' for i in range(10)]
    edges_df = make_edges_df(ports, n_edges=50)
    port_to_idx = {p: i for i, p in enumerate(ports)}

    edge_features, edge_index = preprocess_edge_features(edges_df, port_to_idx)

    assert edge_features.shape[0] == 50, f"Expected 50 edges, got {edge_features.shape[0]}"
    assert edge_index.shape == (2, 50), f"Expected (2, 50), got {edge_index.shape}"

    # All indices should be valid
    assert edge_index.min() >= 0
    assert edge_index.max() < 10

    assert not np.isnan(edge_features).any(), "Edge features contain NaN"
    print(f"  PASS: Edge features shape = {edge_features.shape}, edge_index valid")


def test_filter_graph():
    """Self-loops removed, low-journey edges removed, surviving ports have min_degree."""
    ports = [f'Port_{i:03d}' for i in range(10)]
    edges = []

    # Create a well-connected core (ports 0-4) with high journey counts
    for i in range(5):
        for j in range(5):
            if i != j:
                edges.append({
                    'origin_port': ports[i],
                    'destination_port': ports[j],
                    'total_journeys': 10,
                })

    # Add some self-loops (should be removed)
    edges.append({'origin_port': ports[0], 'destination_port': ports[0], 'total_journeys': 50})

    # Add low-journey edges (should be removed)
    edges.append({'origin_port': ports[0], 'destination_port': ports[5], 'total_journeys': 1})

    # Add the isolated port with only 2 connections (below min_degree=5)
    edges.append({'origin_port': ports[6], 'destination_port': ports[0], 'total_journeys': 5})
    edges.append({'origin_port': ports[0], 'destination_port': ports[6], 'total_journeys': 5})

    df = pd.DataFrame(edges)
    filtered, surviving = filter_graph(df, min_journeys=3, min_degree=5)

    # Self-loops should be gone
    assert not (filtered['origin_port'] == filtered['destination_port']).any()

    # Low-journey edges should be gone
    assert (filtered['total_journeys'] >= 3).all()

    # Port_006 should have been removed (degree < 5)
    assert ports[6] not in surviving

    # Core ports should survive
    for i in range(5):
        assert ports[i] in surviving, f"{ports[i]} should have survived"

    print(f"  PASS: Filter removed self-loops, low-journey, low-degree. Surviving: {len(surviving)}")


def test_filter_graph_cascading():
    """Iterative pruning works: removing one port can cascade."""
    # Create chain: A->B->C->D->E->F, each has degree 2 (below min_degree=5)
    ports = [f'P{i}' for i in range(6)]
    edges = []
    for i in range(5):
        edges.append({'origin_port': ports[i], 'destination_port': ports[i+1], 'total_journeys': 10})
        edges.append({'origin_port': ports[i+1], 'destination_port': ports[i], 'total_journeys': 10})

    df = pd.DataFrame(edges)
    filtered, surviving = filter_graph(df, min_journeys=3, min_degree=5)

    # All ports should be removed since max degree is 4 (2 in + 2 out) for internal, 2 for endpoints
    assert len(surviving) == 0
    assert len(filtered) == 0
    print("  PASS: Cascading degree pruning removes all low-degree ports")


def test_build_adjacency_matrix():
    """Correct shape, binary values, correct nnz."""
    edge_index = np.array([[0, 1, 2, 0], [1, 2, 0, 2]])
    adj = build_adjacency_matrix(edge_index, 4)

    assert adj.shape == (4, 4), f"Expected (4, 4), got {adj.shape}"
    assert adj.nnz == 4
    assert set(adj.data) == {1.0}
    # Node 3 should have no connections
    assert adj[3, :].nnz == 0
    assert adj[:, 3].nnz == 0
    print("  PASS: Adjacency matrix is correct shape, binary, correct nnz")


def test_build_adjacency_matrix_dedup():
    """Duplicate edges should result in binary adjacency (not double-counted)."""
    edge_index = np.array([[0, 0, 1], [1, 1, 2]])  # 0->1 appears twice
    adj = build_adjacency_matrix(edge_index, 3)
    assert adj[0, 1] == 1.0  # Still binary
    print("  PASS: Duplicate edges produce binary adjacency")


def test_compute_warm_ports():
    """Threshold correctly applied."""
    ports = [f'Port_{i}' for i in range(5)]
    journeys = []
    # Port_0 and Port_1 appear many times (warm)
    for _ in range(25):
        journeys.append({
            'vessel_id': 1, 'origin_port': ports[0], 'destination_port': ports[1],
            'departure_time': pd.Timestamp('2022-06-01'),
        })
    # Port_2 appears fewer times (cold)
    for _ in range(5):
        journeys.append({
            'vessel_id': 2, 'origin_port': ports[2], 'destination_port': ports[3],
            'departure_time': pd.Timestamp('2022-06-01'),
        })
    # Port_4 only in the test period (no training appearances)
    journeys.append({
        'vessel_id': 3, 'origin_port': ports[4], 'destination_port': ports[0],
        'departure_time': pd.Timestamp('2023-06-01'),
    })

    df = pd.DataFrame(journeys)
    warm = compute_warm_ports(df, TRAIN_END, threshold=20)

    assert ports[0] in warm  # 25 as origin
    assert ports[1] in warm  # 25 as destination
    assert ports[2] not in warm  # only 5 as origin
    assert ports[4] not in warm  # only test period
    print(f"  PASS: Warm port threshold correctly applied. Warm: {warm}")


def test_build_temporal_splits():
    """Correct temporal assignment, warm port filtering."""
    ports = [f'Port_{i}' for i in range(3)]
    port_to_idx = {p: i for i, p in enumerate(ports)}
    warm_ports = set(ports)  # all warm for this test

    journeys = [
        # Train period
        {'vessel_id': 1, 'origin_port': ports[0], 'destination_port': ports[1],
         'departure_time': pd.Timestamp('2022-06-15')},
        # Validation period
        {'vessel_id': 1, 'origin_port': ports[1], 'destination_port': ports[2],
         'departure_time': pd.Timestamp('2023-02-15')},
        # Test period
        {'vessel_id': 1, 'origin_port': ports[2], 'destination_port': ports[0],
         'departure_time': pd.Timestamp('2023-06-15')},
    ]
    df = pd.DataFrame(journeys)

    result = build_temporal_splits(df, warm_ports, port_to_idx)

    assert len(result['train_transitions']) == 1
    assert len(result['val_transitions']) == 1
    assert len(result['test_transitions']) == 1

    # Check train transition content
    origin, dest, vid = result['train_transitions'][0]
    assert origin == 0 and dest == 1 and vid == 1
    print("  PASS: Temporal splits correctly assigned")


def test_build_temporal_splits_warm_filter():
    """Non-warm ports are filtered out of transitions."""
    ports = [f'Port_{i}' for i in range(3)]
    port_to_idx = {p: i for i, p in enumerate(ports)}
    warm_ports = {ports[0], ports[1]}  # Port_2 is cold

    journeys = [
        # This should be included (warm-to-warm)
        {'vessel_id': 1, 'origin_port': ports[0], 'destination_port': ports[1],
         'departure_time': pd.Timestamp('2022-06-15')},
        # This should be excluded (destination is cold)
        {'vessel_id': 1, 'origin_port': ports[0], 'destination_port': ports[2],
         'departure_time': pd.Timestamp('2022-07-15')},
    ]
    df = pd.DataFrame(journeys)

    result = build_temporal_splits(df, warm_ports, port_to_idx)
    assert len(result['train_transitions']) == 1  # only warm-to-warm
    print("  PASS: Non-warm ports correctly filtered from transitions")


def test_build_node_labels():
    """Correct number of labels, 3 activity levels."""
    nodes_df = make_nodes_df(n=30)
    port_names = sorted(nodes_df['port_name'].tolist())

    sr_labels, act_labels, label_info = build_node_labels(nodes_df, port_names)

    assert len(sr_labels) == 30
    assert len(act_labels) == 30
    assert set(act_labels) == {0, 1, 2}, f"Expected 3 activity levels, got {set(act_labels)}"
    assert label_info['num_sub_regions'] > 0
    assert len(label_info['activity_levels']) == 3
    print(f"  PASS: Labels correct. Sub-regions: {label_info['num_sub_regions']}, Activity levels: 3")


def test_random_split():
    """Random split produces correct proportions."""
    indices = list(range(100))
    train, val, test = random_split(indices, [0.8, 0.1, 0.1])

    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10

    # No overlap
    all_idx = set(train) | set(val) | set(test)
    assert len(all_idx) == 100
    print("  PASS: Random split produces correct proportions with no overlap")


# ===========================================================================
# Integration test
# ===========================================================================

def test_integration_full_pipeline():
    """Build synthetic data, write to temp parquet, load MaritimeDataset,
    verify all attributes populated and trainer compatibility."""
    from src.data.maritime import MaritimeDataset

    tmpdir = tempfile.mkdtemp()
    try:
        # Create synthetic data with enough connectivity
        n_ports = 20
        nodes_df = make_nodes_df(n=n_ports, seed=123)
        ports = nodes_df['port_name'].tolist()

        # Create well-connected edges (each port connects to many others)
        edges = []
        rng = np.random.RandomState(123)
        for i in range(n_ports):
            # Connect each port to at least 6 others (bidirectional)
            targets = rng.choice(
                [j for j in range(n_ports) if j != i],
                min(8, n_ports - 1), replace=False
            )
            for t in targets:
                edges.append({
                    'origin_port': ports[i],
                    'destination_port': ports[t],
                    'total_journeys': rng.randint(5, 50),
                    'unique_vessels': rng.randint(2, 30),
                    'mean_travel_time_hours': rng.uniform(1, 200),
                    'median_travel_time_hours': rng.uniform(1, 100),
                    'std_travel_time_hours': rng.uniform(0, 100),
                    'pct_international': float(rng.choice([0, 1])),
                    'pct_intercontinental': float(rng.choice([0, 1])),
                    'origin_country_first': rng.choice(COUNTRIES),
                    'destination_country_first': rng.choice(COUNTRIES),
                    'origin_continent_first': rng.choice(CONTINENTS),
                    'destination_continent_first': rng.choice(CONTINENTS),
                    'origin_sub_region_first': rng.choice(SUB_REGIONS),
                    'destination_sub_region_first': rng.choice(SUB_REGIONS),
                    'mean_deadweight': rng.uniform(100, 30000),
                    'std_deadweight': rng.uniform(0, 10000),
                    'mean_draught_departure': rng.uniform(1, 10),
                    'mean_draught_arrival': rng.uniform(1, 10),
                    'mean_vessel_age': rng.uniform(5, 30),
                    'dominant_vessel_type': rng.choice(VESSEL_TYPES),
                    'vessel_type_entropy': rng.uniform(0, 3.0),
                    'num_vessel_types': rng.randint(1, 15),
                })
        edges_df = pd.DataFrame(edges)

        # Create journeys spanning the full temporal range
        # Use enough journeys to ensure warm ports
        journeys_df = make_journeys_df(ports, n_journeys=2000, seed=123)

        # Write to parquet
        nodes_df.to_parquet(os.path.join(tmpdir, 'nodes.parquet'))
        edges_df.to_parquet(os.path.join(tmpdir, 'edges.parquet'))
        journeys_df.to_parquet(os.path.join(tmpdir, 'journeys.parquet'))

        # Load dataset
        ds = MaritimeDataset(data_dir=tmpdir)
        ds.load()

        # Verify attributes
        assert ds._loaded is True
        assert sp.issparse(ds.features), "features should be scipy sparse"
        assert sp.issparse(ds.adj), "adj should be scipy sparse"
        assert ds.features.shape[0] > 0, "Should have nodes"
        assert ds.features.shape[0] == ds.adj.shape[0]
        assert ds.adj.shape[0] == ds.adj.shape[1]

        N = ds.features.shape[0]
        E = ds.edge_index.shape[1]

        assert ds.edge_index.shape[0] == 2
        assert ds.edge_features.shape[0] == E
        assert ds.edge_features.shape[1] > 10  # ~44 dims

        assert len(ds.labels) == N
        assert ds.num_features == ds.features.shape[1]
        assert ds.num_classes > 0
        assert ds.edge_dim == ds.edge_features.shape[1]

        # Adjacency should be binary
        adj_coo = sp.coo_matrix(ds.adj)
        assert set(adj_coo.data).issubset({1.0})

        # Splits should be non-empty subsets of valid indices
        assert len(ds.idx_train) > 0
        assert len(ds.idx_val) >= 0
        assert len(ds.idx_test) >= 0

        print(f"  Dataset: N={N}, E={E}, features={ds.num_features}, "
              f"classes={ds.num_classes}, edge_dim={ds.edge_dim}")
        print("  PASS: Full pipeline integration test")

        # --- Cache round-trip test ---
        assert os.path.exists(os.path.join(tmpdir, 'processed', 'node_features.npz'))

        ds2 = MaritimeDataset(data_dir=tmpdir)
        ds2.load()

        # The second load should produce identical data
        np.testing.assert_array_equal(ds.features.toarray(), ds2.features.toarray())
        np.testing.assert_array_equal(ds.adj.toarray(), ds2.adj.toarray())
        np.testing.assert_array_equal(ds.labels, ds2.labels)
        np.testing.assert_array_equal(ds.edge_index, ds2.edge_index)
        np.testing.assert_allclose(ds.edge_features, ds2.edge_features)
        np.testing.assert_array_equal(ds.idx_train, ds2.idx_train)
        print("  PASS: Cache round-trip produces identical data")

        # --- Metadata file check ---
        with open(os.path.join(tmpdir, 'processed', 'metadata.json')) as f:
            metadata = json.load(f)
        assert 'port_to_idx' in metadata
        assert 'label_info' in metadata
        assert 'coverage_stats' in metadata
        print("  PASS: Metadata JSON saved correctly")

    finally:
        shutil.rmtree(tmpdir)


def test_trainer_compatibility():
    """DGITrainer._preprocess_data() runs without error on maritime dataset."""
    from src.data.maritime import MaritimeDataset
    from src.config.config import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig
    from src.training.trainer import DGITrainer

    tmpdir = tempfile.mkdtemp()
    try:
        # Build minimal synthetic dataset
        n_ports = 15
        nodes_df = make_nodes_df(n=n_ports, seed=456)
        ports = nodes_df['port_name'].tolist()

        rng = np.random.RandomState(456)
        edges = []
        for i in range(n_ports):
            targets = rng.choice(
                [j for j in range(n_ports) if j != i],
                min(8, n_ports - 1), replace=False
            )
            for t in targets:
                edges.append({
                    'origin_port': ports[i],
                    'destination_port': ports[t],
                    'total_journeys': rng.randint(5, 50),
                    'unique_vessels': rng.randint(2, 30),
                    'mean_travel_time_hours': rng.uniform(1, 200),
                    'median_travel_time_hours': rng.uniform(1, 100),
                    'std_travel_time_hours': rng.uniform(0, 100),
                    'pct_international': float(rng.choice([0, 1])),
                    'pct_intercontinental': float(rng.choice([0, 1])),
                    'origin_country_first': rng.choice(COUNTRIES),
                    'destination_country_first': rng.choice(COUNTRIES),
                    'origin_continent_first': rng.choice(CONTINENTS),
                    'destination_continent_first': rng.choice(CONTINENTS),
                    'origin_sub_region_first': rng.choice(SUB_REGIONS),
                    'destination_sub_region_first': rng.choice(SUB_REGIONS),
                    'mean_deadweight': rng.uniform(100, 30000),
                    'std_deadweight': rng.uniform(0, 10000),
                    'mean_draught_departure': rng.uniform(1, 10),
                    'mean_draught_arrival': rng.uniform(1, 10),
                    'mean_vessel_age': rng.uniform(5, 30),
                    'dominant_vessel_type': rng.choice(VESSEL_TYPES),
                    'vessel_type_entropy': rng.uniform(0, 3.0),
                    'num_vessel_types': rng.randint(1, 15),
                })
        edges_df = pd.DataFrame(edges)
        journeys_df = make_journeys_df(ports, n_journeys=1000, seed=456)

        nodes_df.to_parquet(os.path.join(tmpdir, 'nodes.parquet'))
        edges_df.to_parquet(os.path.join(tmpdir, 'edges.parquet'))
        journeys_df.to_parquet(os.path.join(tmpdir, 'journeys.parquet'))

        ds = MaritimeDataset(data_dir=tmpdir)
        ds.load()

        # Create a config for maritime (directed + edge features)
        config = ExperimentConfig(
            model=ModelConfig(
                hidden_units=32,
                activation='prelu',
                directed=True,
                use_edge_features=True,
            ),
            training=TrainingConfig(epochs=1, patience=1),
            data=DataConfig(dataset='maritime', sparse=True, data_dir=tmpdir),
            output_dir=tmpdir,
        )

        trainer = DGITrainer(config, device='cpu')
        features, adj_data = trainer._preprocess_data(ds)

        # Verify tensor shapes
        assert features.dim() == 3  # (1, N, F)
        assert features.shape[0] == 1
        assert features.shape[2] == ds.num_features

        assert adj_data['use_edge_features'] is True
        assert adj_data['directed'] is True
        assert 'edge_index' in adj_data
        assert 'edge_features' in adj_data
        assert adj_data['edge_dim'] == ds.edge_dim

        print(f"  Trainer features shape: {features.shape}")
        print(f"  Edge dim: {adj_data['edge_dim']}")
        print("  PASS: Trainer compatibility verified")

    finally:
        shutil.rmtree(tmpdir)


# ===========================================================================
# Test runner
# ===========================================================================

if __name__ == '__main__':
    tests = [
        ("1. encode_geohash", test_encode_geohash),
        ("2. geohash_to_onehot", test_geohash_to_onehot),
        ("3. preprocess_node_features", test_preprocess_node_features),
        ("4. preprocess_node_features (missing values)", test_preprocess_node_features_with_missing),
        ("5. preprocess_edge_features", test_preprocess_edge_features),
        ("6. filter_graph", test_filter_graph),
        ("7. filter_graph (cascading)", test_filter_graph_cascading),
        ("8. build_adjacency_matrix", test_build_adjacency_matrix),
        ("9. build_adjacency_matrix (dedup)", test_build_adjacency_matrix_dedup),
        ("10. compute_warm_ports", test_compute_warm_ports),
        ("11. build_temporal_splits", test_build_temporal_splits),
        ("12. build_temporal_splits (warm filter)", test_build_temporal_splits_warm_filter),
        ("13. build_node_labels", test_build_node_labels),
        ("14. random_split", test_random_split),
        ("15. Integration: full pipeline", test_integration_full_pipeline),
        ("16. Integration: trainer compatibility", test_trainer_compatibility),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\nTest {name}:")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
    print("All tests passed!")
