"""Graph Autoencoder (GAE) baseline using PyTorch Geometric."""
import numpy as np
import torch
import torch.nn.functional as F


class _GCNEncoder(torch.nn.Module):
    """Two-layer GCN encoder: F_in -> 256 -> 512."""

    def __init__(self, in_channels: int, out_channels: int = 512):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_channels, 256)
        self.conv2 = GCNConv(256, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class GAEBaseline:
    """Graph Autoencoder with GCN encoder and dot-product decoder.

    Treats the directed graph symmetrically (GAE decoder is symmetric
    Z @ Z.T), consistent with the spec's intent as a non-directed baseline.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        epochs: int = 200,
        lr: float = 0.01,
    ):
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.lr = lr

    def get_embeddings(self, dataset, device=None) -> np.ndarray:
        """Train GAE and return node embeddings.

        Args:
            dataset: MaritimeDataset with .features, .edge_index, .num_nodes
            device: torch device

        Returns:
            (N, embedding_dim) float32 numpy array
        """
        from torch_geometric.nn import GAE

        if device is None:
            device = torch.device("cpu")

        x = torch.FloatTensor(dataset.features.toarray()).to(device)
        edge_index = torch.LongTensor(dataset.edge_index).to(device)

        encoder = _GCNEncoder(x.shape[1], self.embedding_dim)
        model = GAE(encoder).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            model.train()
            optimizer.zero_grad()
            z = model.encode(x, edge_index)
            loss = model.recon_loss(z, edge_index)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 50 == 0 or epoch == 0:
                print(f"  GAE epoch {epoch + 1}/{self.epochs}: loss={loss.item():.4f}")

        model.eval()
        with torch.no_grad():
            embeddings = model.encode(x, edge_index).cpu().numpy()

        return embeddings.astype(np.float32)
