"""DGI training logic."""
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from src.config import ExperimentConfig
from src.models import DGI
from src.preprocessing import (
    preprocess_features,
    normalize_adj,
    preprocess_directed_adj,
    sparse_mx_to_torch_sparse_tensor,
)
from src.data import GraphDataset
from .callbacks import EarlyStopping


class DGITrainer:
    """Deep Graph Infomax trainer.

    Handles the complete training pipeline, including
    - Data preprocessing
    - Model initialization
    - Training loop with early stopping
    - Embedding extraction
    """

    def __init__(self, config: ExperimentConfig, device: Optional[torch.device] = None):
        """
        Args:
            config: Experiment configuration
            device: Device to train on (defaults to CUDA if available)
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model: Optional[DGI] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_history: List[float] = []

    def fit(self, dataset: GraphDataset) -> Tuple[DGI, torch.Tensor]:
        """Train DGI model on dataset.

        Args:
            dataset: Graph dataset to train on

        Returns:
            Tuple of (the trained model, node embeddings)
        """
        # Load and preprocess data
        if not dataset._loaded:
            dataset.load()

        features, adj_data = self._preprocess_data(dataset)

        # Detect edge feature dimensionality
        edge_dim = adj_data.get('edge_dim', None)

        # Initialize model
        self._init_model(dataset.num_features, edge_dim=edge_dim)

        # Setup checkpointing
        checkpoint_dir = Path(self.config.output_dir) / 'checkpoints'
        prefix = getattr(self.config.training, 'checkpoint_prefix', '') or dataset.name
        checkpoint_path = checkpoint_dir / f'{prefix}_best_dgi.pkl'

        early_stopping = EarlyStopping(
            patience=self.config.training.patience,
            checkpoint_path=str(checkpoint_path),
        )

        # Training loop
        nb_nodes = features.shape[1]

        for epoch in range(self.config.training.epochs):
            loss = self._train_epoch(features, adj_data, nb_nodes)
            self.loss_history.append(loss)
            print(f'Epoch {epoch}: Loss = {loss:.4f}')

            if early_stopping(loss, self.model, epoch):
                break

        # Load the best model and extract embeddings
        early_stopping.load_best(self.model)
        embeddings = self._extract_embeddings(features, adj_data)

        return self.model, embeddings

    def _preprocess_data(self, dataset: GraphDataset) -> Tuple[torch.Tensor, dict]:
        """Preprocess dataset features and adjacency matrix."""
        # Preprocess features
        if self.config.data.skip_feature_normalization:
            # Maritime features are already preprocessed (StandardScaled + one-hot);
            # L1 row-normalization is harmful for mixed-sign feature values.
            features = np.array(dataset.features.todense())
        else:
            features, _ = preprocess_features(dataset.features)
        features = torch.FloatTensor(features[np.newaxis]).to(self.device)

        adj_data = {}
        directed = self.config.model.directed
        sparse = self.config.data.sparse

        if directed:
            adj_in_tuple, adj_out_tuple = preprocess_directed_adj(dataset.adj)

            if sparse:
                adj_in_sparse = sparse_mx_to_torch_sparse_tensor(
                    sp.coo_matrix(
                        (adj_in_tuple[1], (adj_in_tuple[0][:, 0], adj_in_tuple[0][:, 1])),
                        shape=adj_in_tuple[2]
                    )
                ).to(self.device)
                adj_out_sparse = sparse_mx_to_torch_sparse_tensor(
                    sp.coo_matrix(
                        (adj_out_tuple[1], (adj_out_tuple[0][:, 0], adj_out_tuple[0][:, 1])),
                        shape=adj_out_tuple[2]
                    )
                ).to(self.device)
                adj_data['adj_in'] = adj_in_sparse
                adj_data['adj_out'] = adj_out_sparse
            else:
                adj_in = sp.coo_matrix(
                    (adj_in_tuple[1], (adj_in_tuple[0][:, 0], adj_in_tuple[0][:, 1])),
                    shape=adj_in_tuple[2]
                ).todense()
                adj_out = sp.coo_matrix(
                    (adj_out_tuple[1], (adj_out_tuple[0][:, 0], adj_out_tuple[0][:, 1])),
                    shape=adj_out_tuple[2]
                ).todense()
                adj_data['adj_in'] = torch.FloatTensor(adj_in[np.newaxis]).to(self.device)
                adj_data['adj_out'] = torch.FloatTensor(adj_out[np.newaxis]).to(self.device)
        else:
            adj = normalize_adj(dataset.adj + sp.eye(dataset.adj.shape[0]))

            if sparse:
                adj_data['adj'] = sparse_mx_to_torch_sparse_tensor(adj).to(self.device)
            else:
                adj_dense = adj.todense()
                adj_data['adj'] = torch.FloatTensor(adj_dense[np.newaxis]).to(self.device)

        adj_data['sparse'] = sparse
        adj_data['directed'] = directed

        # Handle edge features for directed graphs
        use_edge_features = (
            directed
            and self.config.model.use_edge_features
            and dataset.edge_features is not None
        )
        adj_data['use_edge_features'] = use_edge_features

        if use_edge_features:
            adj_data['edge_index'] = torch.LongTensor(dataset.edge_index).to(self.device)
            adj_data['edge_features'] = torch.FloatTensor(dataset.edge_features).to(self.device)
            adj_data['edge_dim'] = dataset.edge_dim

        return features, adj_data

    def _init_model(self, num_features: int, edge_dim: int = None) -> None:
        """Initialize DGI model and optimizer."""
        self.model = DGI(
            num_features,
            self.config.model.hidden_units,
            self.config.model.activation,
            directed=self.config.model.directed,
            edge_dim=edge_dim,
            num_layers=getattr(self.config.model, 'num_layers', 1),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

    def _train_epoch(
        self,
        features: torch.Tensor,
        adj_data: dict,
        nb_nodes: int
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        # Create corrupted features by shuffling
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :].to(self.device)

        # Create labels
        lbl_1 = torch.ones(self.config.training.batch_size, nb_nodes)
        lbl_2 = torch.zeros(self.config.training.batch_size, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)

        # Forward pass
        if adj_data.get('use_edge_features'):
            logits = self.model(
                features, shuf_fts, None,
                adj_data['sparse'], None, None, None,
                edge_index=adj_data['edge_index'],
                edge_features=adj_data['edge_features'],
            )
        elif adj_data['directed']:
            logits = self.model(
                features, shuf_fts, adj_data['adj_in'],
                adj_data['sparse'], None, None, None, adj_data['adj_out']
            )
        else:
            logits = self.model(
                features, shuf_fts, adj_data['adj'],
                adj_data['sparse'], None, None, None
            )

        loss = self.loss_fn(logits, lbl)

        loss.backward()
        edge_clip = self.config.training.edge_grad_clip_norm
        if (
            edge_clip > 0
            and adj_data.get('use_edge_features')
            and hasattr(self.model, 'encoder')
            and hasattr(self.model.encoder, 'edge_mlp')
        ):
            # Per-parameter-group clipping: edge_mlp gets a tighter budget
            nn.utils.clip_grad_norm_(self.model.encoder.edge_mlp.parameters(), edge_clip)
            other_params = [
                p for n, p in self.model.named_parameters()
                if not n.startswith('encoder.edge_mlp')
            ]
            if self.config.training.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(other_params, self.config.training.grad_clip_norm)
        elif self.config.training.grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.grad_clip_norm,
            )
        self.optimizer.step()

        return loss.item()

    def _extract_embeddings(
        self,
        features: torch.Tensor,
        adj_data: dict
    ) -> torch.Tensor:
        """Extract embeddings from the trained model."""
        self.model.eval()

        with torch.no_grad():
            if adj_data.get('use_edge_features'):
                embeds, _ = self.model.embed(
                    features, None,
                    adj_data['sparse'], None,
                    edge_index=adj_data['edge_index'],
                    edge_features=adj_data['edge_features'],
                )
            elif adj_data['directed']:
                embeds, _ = self.model.embed(
                    features, adj_data['adj_in'],
                    adj_data['sparse'], None, adj_data['adj_out']
                )
            else:
                embeds, _ = self.model.embed(
                    features, adj_data['adj'],
                    adj_data['sparse'], None
                )

        return embeds
