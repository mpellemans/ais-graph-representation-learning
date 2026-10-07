"""Node2Vec baseline using PyTorch Geometric."""
import numpy as np
import torch


class Node2VecBaseline:
    """Node2Vec random-walk embeddings via torch_geometric.

    Uses standard DeepWalk parameters (p=1, q=1, walk_length=80,
    context_size=10, walks_per_node=10). The edge_index is passed to PyG's
    Node2Vec as-is, so walks follow the directed edges as given.

    Note: torch_geometric's Node2Vec loader requires the torch-cluster
    package at runtime (see requirements.txt).
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        walk_length: int = 80,
        context_size: int = 10,
        walks_per_node: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        num_negative_samples: int = 1,
        epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 128,
    ):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    def get_embeddings(self, dataset, device=None) -> np.ndarray:
        """Train Node2Vec and return learned embeddings.

        Args:
            dataset: MaritimeDataset with .edge_index (numpy 2×E) and .num_nodes
            device: torch device

        Returns:
            (N, embedding_dim) float32 numpy array
        """
        from torch_geometric.nn import Node2Vec

        if device is None:
            device = torch.device("cpu")

        edge_index = torch.LongTensor(dataset.edge_index).to(device)
        N = dataset.num_nodes

        model = Node2Vec(
            edge_index,
            embedding_dim=self.embedding_dim,
            walk_length=self.walk_length,
            context_size=self.context_size,
            walks_per_node=self.walks_per_node,
            p=self.p,
            q=self.q,
            num_negative_samples=self.num_negative_samples,
            num_nodes=N,
        ).to(device)

        loader = model.loader(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for epoch in range(self.epochs):
            model.train()
            total_loss = 0.0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Node2Vec epoch {epoch + 1}/{self.epochs}: loss={total_loss:.4f}")

        model.eval()
        with torch.no_grad():
            embeddings = model.embedding.weight.detach().cpu().numpy()

        return embeddings.astype(np.float32)
