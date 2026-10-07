"""Abstract base class for graph datasets."""
from abc import ABC, abstractmethod
from typing import Tuple, Any, Optional
import torch
import scipy.sparse as sp


class GraphDataset(ABC):
    """Abstract base class for graph datasets.

    All dataset implementations should inherit from this class and implement
    the required abstract methods.
    """

    def __init__(self, name: str, data_dir: str):
        """
        Args:
            name: Dataset name
            data_dir: Directory containing the dataset files
        """
        self.name = name
        self.data_dir = data_dir
        self._loaded = False

        # Data attributes (populated after load())
        self.adj: Optional[sp.spmatrix] = None
        self.features: Optional[sp.spmatrix] = None
        self.labels: Optional[Any] = None
        self.idx_train: Optional[Any] = None
        self.idx_val: Optional[Any] = None
        self.idx_test: Optional[Any] = None
        self.graph: Optional[Any] = None

        # Edge data attributes (populated by datasets with edge features)
        self.edge_index: Optional[Any] = None   # shape (2, E), source-destination pairs
        self.edge_features: Optional[Any] = None  # shape (E, edge_dim)

    @abstractmethod
    def load(self) -> 'GraphDataset':
        """Load raw data files and populate class attributes.

        Returns:
            self for method chaining
        """
        pass

    def get_splits(self) -> Tuple[Any, Any, Any]:
        """Return train/val/test indices."""
        if not self._loaded:
            self.load()
        return self.idx_train, self.idx_val, self.idx_test

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Number of classes in the dataset."""
        pass

    @property
    @abstractmethod
    def num_features(self) -> int:
        """Number of input features per node."""
        pass

    @property
    def edge_dim(self) -> Optional[int]:
        """Number of edge features, or None if no edge features."""
        if self.edge_features is not None:
            return self.edge_features.shape[1]
        return None

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph."""
        if not self._loaded:
            self.load()
        return self.features.shape[0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, nodes={self.num_nodes if self._loaded else '?'})"
