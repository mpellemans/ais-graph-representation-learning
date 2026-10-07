"""GraphSAGE unsupervised baseline using PyTorch Geometric."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _SAGEEncoder(nn.Module):
    """Two-layer SAGEConv encoder: F_in -> 512 -> 512."""

    def __init__(self, in_channels: int, hidden: int = 512):
        super().__init__()
        from torch_geometric.nn import SAGEConv
        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class GraphSAGEBaseline:
    """Unsupervised GraphSAGE trained with link-prediction loss.

    For each edge (u, v) in the training set, sample k negative nodes w per u
    and minimize BCE on dot products: embed_u @ embed_v (positive) versus
    embed_u @ embed_w (negative).
    """

    def __init__(
        self,
        hidden: int = 512,
        epochs: int = 100,
        lr: float = 0.001,
        num_neg_samples: int = 5,
    ):
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.num_neg_samples = num_neg_samples

    def get_embeddings(self, dataset, device=None) -> np.ndarray:
        """Train GraphSAGE unsupervised and return node embeddings.

        Args:
            dataset: MaritimeDataset with .features, .edge_index, .num_nodes
            device: torch device

        Returns:
            (N, hidden) float32 numpy array
        """
        if device is None:
            device = torch.device("cpu")

        # Node features
        x = torch.FloatTensor(dataset.features.toarray()).to(device)
        N = dataset.num_nodes
        edge_index = torch.LongTensor(dataset.edge_index).to(device)

        # Use only training-period edges (edge_index already reflects the graph)
        train_src = edge_index[0]
        train_dst = edge_index[1]
        n_edges = edge_index.shape[1]

        model = _SAGEEncoder(x.shape[1], self.hidden).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()

            z = model(x, edge_index)  # (N, hidden)

            # Positive scores
            pos_scores = (z[train_src] * z[train_dst]).sum(dim=1)  # (E,)

            # Negative samples: random nodes for each edge source
            neg_dst = torch.randint(0, N, (n_edges * self.num_neg_samples,), device=device)
            neg_src = train_src.repeat(self.num_neg_samples)
            neg_scores = (z[neg_src] * z[neg_dst]).sum(dim=1)

            pos_labels = torch.ones(n_edges, device=device)
            neg_labels = torch.zeros(n_edges * self.num_neg_samples, device=device)

            loss = loss_fn(
                torch.cat([pos_scores, neg_scores]),
                torch.cat([pos_labels, neg_labels]),
            )
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f"  GraphSAGE epoch {epoch + 1}/{self.epochs}: loss={loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            embeddings = model(x, edge_index).cpu().numpy()

        return embeddings.astype(np.float32)
