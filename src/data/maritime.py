"""Maritime port-visit network dataset loader.

Loads parquet files, preprocesses node/edge features, constructs the directed
graph, builds temporal splits for downstream evaluation, and caches processed
data for fast reloading.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .base import GraphDataset
from src.preprocessing.maritime import (
    filter_graph,
    sparsify_graph,
    preprocess_node_features,
    preprocess_edge_features,
    build_adjacency_matrix,
    build_node_labels,
    compute_warm_ports,
    build_temporal_splits,
    print_graph_statistics,
    random_split,
    TRAIN_END,
)


class MaritimeDataset(GraphDataset):
    """Maritime port-visit network dataset.

    Dataset characteristics:
    - Nodes: ~4,325 ports worldwide (before filtering)
    - Edges: ~270,272 directed vessel routes (before filtering)
    - Node features: ~441 dims (geographic, vessel, activity, connectivity)
    - Edge features: ~44 dims (travel time, frequency, vessel types)

    Source: AIS data (Feb 2022 - Aug 2023)
    """

    def __init__(
        self,
        data_dir: str = 'data/maritime',
        name: str = 'maritime',
        sparsify_top_k: int = 0,
        exclude_geo_features: bool = False,
        exclude_connectivity_features: bool = False,
    ):
        super().__init__(name, data_dir)
        self._num_classes: Optional[int] = None
        self._num_features: Optional[int] = None
        self.sparsify_top_k = sparsify_top_k
        self.exclude_geo_features = exclude_geo_features
        self.exclude_connectivity_features = exclude_connectivity_features

    def _cache_suffix(self) -> str:
        """Return cache directory suffix based on active preprocessing flags."""
        suffix = ''
        if self.sparsify_top_k > 0:
            suffix += f'_k{self.sparsify_top_k}'
        if self.exclude_geo_features:
            suffix += '_no_geo'
        if self.exclude_connectivity_features:
            suffix += '_no_conn'
        return suffix

    def load(self) -> 'MaritimeDataset':
        """Load maritime dataset from parquet files or cache."""
        data_path = Path(self.data_dir)

        if not data_path.exists():
            raise FileNotFoundError(
                f"Maritime data directory not found: {data_path}. "
                "Please prepare the maritime dataset first."
            )

        cache_path = data_path / f'processed{self._cache_suffix()}'

        # Check for cached processed data
        if (cache_path / 'node_features.npz').exists():
            return self._load_cached(cache_path)

        # Load raw parquet files
        nodes_df = pd.read_parquet(data_path / 'nodes.parquet')
        edges_df = pd.read_parquet(data_path / 'edges.parquet')
        journeys_df = pd.read_parquet(data_path / 'journeys.parquet')

        initial_nodes = nodes_df['port_name'].nunique()
        initial_edges = len(edges_df)

        # Filter graph
        filtered_edges, surviving_ports = filter_graph(edges_df)

        # Optional: sparsify to top-K outgoing edges per node
        if self.sparsify_top_k > 0:
            filtered_edges = sparsify_graph(filtered_edges, self.sparsify_top_k)
            surviving_ports = (
                set(filtered_edges['origin_port'].unique())
                | set(filtered_edges['destination_port'].unique())
            )

        nodes_df = nodes_df[nodes_df['port_name'].isin(surviving_ports)].copy()

        # Build port index mapping (sorted for determinism)
        port_names = sorted(surviving_ports)
        port_to_idx = {name: i for i, name in enumerate(port_names)}

        # Preprocess features
        node_features, port_names_ordered, preproc_info = preprocess_node_features(
            nodes_df,
            exclude_geo_features=self.exclude_geo_features,
            exclude_connectivity_features=self.exclude_connectivity_features,
        )
        edge_features, edge_index = preprocess_edge_features(filtered_edges, port_to_idx)

        # Build adjacency
        num_nodes = len(port_names_ordered)
        adj = build_adjacency_matrix(edge_index, num_nodes)

        # Build labels
        sub_region_labels, activity_labels, label_info = build_node_labels(
            nodes_df, port_names_ordered
        )

        # Temporal splits for downstream evaluation
        # Ensure departure_time is datetime
        if not pd.api.types.is_datetime64_any_dtype(journeys_df['departure_time']):
            journeys_df['departure_time'] = pd.to_datetime(journeys_df['departure_time'])
        warm_ports = compute_warm_ports(journeys_df, TRAIN_END)
        temporal_data = build_temporal_splits(journeys_df, warm_ports, port_to_idx)

        # Build vessel_id -> vessel_type mapping for per-vessel-type evaluation
        if 'ShiptypeLevel3' in journeys_df.columns:
            vt_series = journeys_df.drop_duplicates('vessel_id').set_index('vessel_id')['ShiptypeLevel3']
            # Drop NaN values and ensure string keys for JSON serialization
            vessel_type_map = {
                str(k): v for k, v in vt_series.dropna().items()
            }
        else:
            vessel_type_map = None

        # Random 80/10/10 split among warm ports for classification evaluation
        warm_indices = [port_to_idx[p] for p in warm_ports if p in port_to_idx]
        idx_train, idx_val, idx_test = random_split(warm_indices, [0.8, 0.1, 0.1])

        # Populate dataset attributes
        self.port_to_idx = port_to_idx
        self.features = sp.lil_matrix(node_features)
        self.adj = adj
        self.labels = sub_region_labels
        self.activity_labels = activity_labels
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)
        self._num_features = node_features.shape[1]
        self._num_classes = len(np.unique(sub_region_labels))
        self._loaded = True

        # Store transitions and vessel type map
        for split_name in ['train_transitions', 'val_transitions', 'test_transitions']:
            transitions = temporal_data[split_name]
            setattr(self, split_name, np.array(transitions, dtype=object) if transitions else np.array([]))
        self.vessel_type_map = vessel_type_map

        # Cache processed data
        self._save_cache(
            cache_path, temporal_data, preproc_info, label_info,
            port_to_idx, activity_labels, vessel_type_map,
        )

        # Print statistics
        print_graph_statistics(
            num_nodes, edge_index.shape[1], adj,
            initial_nodes=initial_nodes,
            initial_edges=initial_edges,
        )

        return self

    def _load_cached(self, cache_path: Path) -> 'MaritimeDataset':
        """Load preprocessed data from cache."""
        node_feat = sp.load_npz(cache_path / 'node_features.npz')
        adj = sp.load_npz(cache_path / 'adj.npz')
        edge_features = np.load(cache_path / 'edge_features.npy')
        edge_index = np.load(cache_path / 'edge_index.npy')
        labels = np.load(cache_path / 'labels.npy')
        activity_labels = np.load(cache_path / 'activity_labels.npy')

        # Context manager closes the .npz file handle (Windows cannot delete
        # the cache directory while the handle is open)
        with np.load(cache_path / 'splits.npz') as splits:
            idx_train = splits['idx_train']
            idx_val = splits['idx_val']
            idx_test = splits['idx_test']

        self.features = sp.lil_matrix(node_feat)
        self.adj = adj
        self.labels = labels
        self.activity_labels = activity_labels
        self.edge_index = edge_index
        self.edge_features = edge_features
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self._num_features = node_feat.shape[1]
        self._num_classes = len(np.unique(labels))
        self._loaded = True

        # Load transitions
        for split_name in ['train_transitions', 'val_transitions', 'test_transitions']:
            path = cache_path / f'{split_name}.npy'
            if path.exists():
                setattr(self, split_name, np.load(path, allow_pickle=True))
            else:
                setattr(self, split_name, np.array([]))

        # Load vessel_type_map — migrate cache if file missing
        vt_path = cache_path / 'vessel_type_map.json'
        if vt_path.exists():
            with open(vt_path) as f:
                self.vessel_type_map = json.load(f)
        else:
            # Cache migration: build map from raw parquet (one-time cost).
            # Skipped when journeys.parquet lacks the ShiptypeLevel3 column
            # (e.g. reduced synthetic data) — per-vessel-type metrics are then
            # unavailable but everything else works.
            journeys_path = Path(self.data_dir) / 'journeys.parquet'
            if journeys_path.exists():
                try:
                    jdf = pd.read_parquet(journeys_path, columns=['vessel_id', 'ShiptypeLevel3'])
                except Exception as e:
                    print(f"  Note: vessel_type_map unavailable ({type(e).__name__}); "
                          "per-vessel-type metrics will be skipped.")
                    self.vessel_type_map = None
                else:
                    vt_series = jdf.drop_duplicates('vessel_id').set_index('vessel_id')['ShiptypeLevel3']
                    self.vessel_type_map = {str(k): v for k, v in vt_series.dropna().items()}
                    with open(vt_path, 'w') as f:
                        json.dump(self.vessel_type_map, f)
                    print(f"  Cache migration: wrote vessel_type_map.json ({len(self.vessel_type_map)} vessels)")
            else:
                self.vessel_type_map = None

        # Load port_to_idx from metadata
        metadata_path = cache_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            self.port_to_idx = meta.get('port_to_idx', {})
        else:
            self.port_to_idx = {}

        print(f"Loaded cached maritime dataset from {cache_path}")
        print(f"  Nodes: {node_feat.shape[0]}, Features: {node_feat.shape[1]}")
        print(f"  Edges: {edge_index.shape[1]}, Edge features: {edge_features.shape[1]}")

        return self

    def _save_cache(
        self,
        cache_path: Path,
        temporal_data: dict,
        preproc_info: dict,
        label_info: dict,
        port_to_idx: dict,
        activity_labels: np.ndarray,
        vessel_type_map: Optional[dict] = None,
    ) -> None:
        """Save processed data to cache directory."""
        cache_path.mkdir(parents=True, exist_ok=True)

        # Sparse matrices
        sp.save_npz(cache_path / 'node_features.npz', sp.csr_matrix(self.features))
        sp.save_npz(cache_path / 'adj.npz', sp.csr_matrix(self.adj))

        # Dense arrays
        np.save(cache_path / 'edge_features.npy', self.edge_features)
        np.save(cache_path / 'edge_index.npy', self.edge_index)
        np.save(cache_path / 'labels.npy', self.labels)
        np.save(cache_path / 'activity_labels.npy', activity_labels)

        # Splits
        np.savez(
            cache_path / 'splits.npz',
            idx_train=self.idx_train,
            idx_val=self.idx_val,
            idx_test=self.idx_test,
        )

        # Temporal splits
        for split_name in ['train_transitions', 'val_transitions', 'test_transitions']:
            transitions = temporal_data[split_name]
            if transitions:
                arr = np.array(transitions, dtype=object)
                np.save(cache_path / f'{split_name}.npy', arr, allow_pickle=True)

        # Metadata (JSON-serializable)
        metadata = {
            'port_to_idx': port_to_idx,
            'label_info': label_info,
            'preprocessing_info': preproc_info,
            'coverage_stats': temporal_data['coverage_stats'],
        }
        with open(cache_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

        # Vessel type mapping
        if vessel_type_map is not None:
            with open(cache_path / 'vessel_type_map.json', 'w') as f:
                json.dump(vessel_type_map, f, indent=2)

    @property
    def num_classes(self) -> int:
        if self._num_classes is None:
            raise ValueError("Dataset not loaded or num_classes not set")
        return self._num_classes

    @property
    def num_features(self) -> int:
        if self._num_features is None:
            raise ValueError("Dataset not loaded or num_features not set")
        return self._num_features
