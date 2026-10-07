"""Smoke tests for baseline methods using synthetic data.

All tests use a minimal mock dataset (N=30 nodes, F=10 features, ~50 directed edges)
and verify that get_embeddings() returns a valid (N, D) float32 array.

Run with:
    python tests/test_baselines.py
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Ensure project root is importable
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def make_mock_dataset(N=30, F=10, n_edges=50, seed=0):
    """Build a minimal mock dataset matching MaritimeDataset's interface."""
    rng = np.random.default_rng(seed)

    # Dense features -> lil_matrix (as MaritimeDataset stores them)
    features_dense = rng.random((N, F)).astype(np.float32)
    features = sp.lil_matrix(features_dense)

    # Random directed edge_index (no self-loops)
    src = rng.integers(0, N, size=n_edges)
    dst = rng.integers(0, N, size=n_edges)
    mask = src != dst
    src, dst = src[mask], dst[mask]
    edge_index = np.stack([src, dst], axis=0)  # (2, E)

    # Sparse adjacency matrix
    data = np.ones(len(src))
    adj = sp.csr_matrix((data, (src, dst)), shape=(N, N))

    class MockDataset:
        pass

    ds = MockDataset()
    ds.features = features
    ds.adj = adj
    ds.edge_index = edge_index
    ds.num_nodes = N
    ds.num_features = F

    return ds


class TestRawFeaturesBaseline(unittest.TestCase):
    def test_shape_and_dtype(self):
        from src.baselines.raw_features import RawFeaturesBaseline
        ds = make_mock_dataset()
        emb = RawFeaturesBaseline().get_embeddings(ds)
        self.assertEqual(emb.ndim, 2)
        self.assertEqual(emb.shape[0], ds.num_nodes)
        self.assertEqual(emb.dtype, np.float32)


class TestNode2VecBaseline(unittest.TestCase):
    def test_shape_and_dtype(self):
        try:
            from torch_geometric.nn import Node2Vec
            # Probe whether the random walk backend is available
            import torch
            _test_ei = torch.zeros(2, 2, dtype=torch.long)
            Node2Vec(_test_ei, embedding_dim=4, walk_length=2, context_size=1, num_nodes=3)
        except (ImportError, Exception):
            self.skipTest("torch_geometric Node2Vec backend (pyg-lib/torch-cluster) not available")

        from src.baselines.node2vec import Node2VecBaseline
        ds = make_mock_dataset(N=30, n_edges=60)
        baseline = Node2VecBaseline(
            embedding_dim=16,
            walk_length=5,
            context_size=3,
            walks_per_node=2,
            epochs=1,
        )
        emb = baseline.get_embeddings(ds)
        self.assertEqual(emb.ndim, 2)
        self.assertEqual(emb.shape[0], ds.num_nodes)
        self.assertEqual(emb.dtype, np.float32)


class TestGraphSAGEBaseline(unittest.TestCase):
    def test_shape_and_dtype(self):
        try:
            import torch_geometric  # noqa: F401
        except ImportError:
            self.skipTest("torch_geometric not installed")

        from src.baselines.graphsage import GraphSAGEBaseline
        ds = make_mock_dataset(N=30, n_edges=60)
        baseline = GraphSAGEBaseline(hidden=16, epochs=2)
        emb = baseline.get_embeddings(ds)
        self.assertEqual(emb.ndim, 2)
        self.assertEqual(emb.shape[0], ds.num_nodes)
        self.assertEqual(emb.dtype, np.float32)


class TestGAEBaseline(unittest.TestCase):
    def test_shape_and_dtype(self):
        try:
            import torch_geometric  # noqa: F401
        except ImportError:
            self.skipTest("torch_geometric not installed")

        from src.baselines.gae import GAEBaseline
        ds = make_mock_dataset(N=30, n_edges=60)
        baseline = GAEBaseline(embedding_dim=16, epochs=2)
        emb = baseline.get_embeddings(ds)
        self.assertEqual(emb.ndim, 2)
        self.assertEqual(emb.shape[0], ds.num_nodes)
        self.assertEqual(emb.dtype, np.float32)


class TestHOPEBaseline(unittest.TestCase):
    def test_shape_and_dtype(self):
        from src.baselines.hope import HOPEBaseline
        ds = make_mock_dataset(N=30, n_edges=60)
        baseline = HOPEBaseline(embedding_dim=8)
        emb = baseline.get_embeddings(ds)
        self.assertEqual(emb.ndim, 2)
        self.assertEqual(emb.shape[0], ds.num_nodes)
        self.assertEqual(emb.shape[1], 16)  # 2 * embedding_dim
        self.assertEqual(emb.dtype, np.float32)

    def test_beta_zero_raises_or_returns_zero(self):
        """With beta=0, Katz matrix K = 0; embeddings should still be valid arrays."""
        from src.baselines.hope import HOPEBaseline
        ds = make_mock_dataset(N=20, n_edges=30)
        baseline = HOPEBaseline(embedding_dim=4, beta=0.0)
        emb = baseline.get_embeddings(ds)
        self.assertEqual(emb.shape, (20, 8))
        self.assertTrue(np.isfinite(emb).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
