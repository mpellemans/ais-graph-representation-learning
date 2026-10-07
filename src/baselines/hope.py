"""HOPE (High-Order Proximity preserved Embedding) baseline.

Approximates the Katz index proximity matrix K = (I - beta*A)^{-1} - I
via a dense linear solver, then factorizes it with truncated SVD to produce
source and target embeddings.

Note on dot-product scoring: DotProductLinkPredictor computes embed_i @ embed_j
on the full 512-dim concatenated vector [Phi_s; Phi_t], which mixes Phi_s@Phi_s
and Phi_t@Phi_t terms rather than the intended Phi_s(origin) @ Phi_t(dest) inner
product. MLP and LSTM rankers can learn to recover the correct structure from
the concatenated representation.
"""
import numpy as np
from sklearn.utils.extmath import randomized_svd


class HOPEBaseline:
    """HOPE embeddings via Katz index + truncated SVD.

    Produces (N, 512) float32 embeddings by concatenating source (N, 256)
    and target (N, 256) embedding matrices.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        beta: float = 0.01,
        random_state: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.random_state = random_state

    def get_embeddings(self, dataset, device=None) -> np.ndarray:
        """Compute HOPE embeddings via Katz proximity + SVD.

        Args:
            dataset: MaritimeDataset with .adj (sparse CSR)
            device: ignored (CPU-only numpy computation)

        Returns:
            (N, embedding_dim * 2) float32 numpy array — source+target concat
        """
        # Convert adjacency to dense float64 for BLAS-accelerated solving
        adj = dataset.adj
        N = adj.shape[0]
        A = adj.toarray().astype(np.float64)  # ~49 MB for N=3500

        # Katz index: K = (I - beta*A)^{-1} - I
        I = np.eye(N, dtype=np.float64)
        M = I - self.beta * A
        K = np.linalg.solve(M, I) - I  # BLAS-accelerated, ~seconds for N=3500

        # Truncated SVD: K ≈ U * diag(sigma) * Vt
        U, sigma, Vt = randomized_svd(
            K,
            n_components=self.embedding_dim,
            random_state=self.random_state,
        )

        sqrt_sigma = np.sqrt(np.maximum(sigma, 0))  # guard against tiny negatives

        # Source embeddings: U * sqrt(sigma)   -> (N, embedding_dim)
        phi_s = (U * sqrt_sigma).astype(np.float32)
        # Target embeddings: Vt.T * sqrt(sigma) -> (N, embedding_dim)
        phi_t = (Vt.T * sqrt_sigma).astype(np.float32)

        # Concatenate to (N, 2 * embedding_dim)
        return np.concatenate([phi_s, phi_t], axis=1)
