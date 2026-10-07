"""Planetoid dataset loader (Cora, Citeseer, Pubmed)."""
import pickle as pkl
import sys
from pathlib import Path
from typing import Tuple, Any

import numpy as np
import scipy.sparse as sp
import networkx as nx

from .base import GraphDataset


class PlanetoidDataset(GraphDataset):
    """Planetoid citation network datasets (Cora, Citeseer, Pubmed).

    These datasets were introduced in:
    "Revisiting Semi-Supervised Learning with Graph Embeddings"
    Yang et al., ICML 2016
    """

    VALID_DATASETS = {'cora', 'citeseer', 'pubmed'}

    def __init__(self, name: str, data_dir: str = 'data/planetoid'):
        """
        Args:
            name: Dataset name ('cora', 'citeseer', or 'pubmed')
            data_dir: Directory containing the dataset files
        """
        if name not in self.VALID_DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Must be one of {self.VALID_DATASETS}")
        super().__init__(name, data_dir)

    def load(self) -> 'PlanetoidDataset':
        """Load Planetoid dataset from pickle files."""
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []

        data_path = Path(self.data_dir) / self.name

        for name in names:
            file_path = data_path / f"ind.{self.name}.{name}"
            with open(file_path, 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))
                else:
                    objects.append(pkl.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)

        # Load test indices
        test_idx_path = data_path / f"ind.{self.name}.test.index"
        test_idx_reorder = []
        with open(test_idx_path) as f:
            for line in f:
                test_idx_reorder.append(int(line.strip()))

        test_idx_range = np.sort(test_idx_reorder)

        # Handle citeseer isolated nodes
        if self.name == 'citeseer':
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range - min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range - min(test_idx_range), :] = ty
            ty = ty_extended

        # Combine features and labels
        self.features = sp.vstack((allx, tx)).tolil()
        self.features[test_idx_reorder, :] = self.features[test_idx_range, :]

        self.adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        self.labels = np.vstack((ally, ty))
        self.labels[test_idx_reorder, :] = self.labels[test_idx_range, :]

        # Create splits
        self.idx_test = test_idx_range.tolist()
        self.idx_train = list(range(len(y)))
        self.idx_val = list(range(len(y), len(y) + 500))

        self.graph = graph
        self._loaded = True

        return self

    @property
    def num_classes(self) -> int:
        """Number of classes in the dataset."""
        if not self._loaded:
            self.load()
        return self.labels.shape[1]

    @property
    def num_features(self) -> int:
        """Number of input features per node."""
        if not self._loaded:
            self.load()
        return self.features.shape[1]


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create a mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool_)
