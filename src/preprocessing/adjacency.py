"""Adjacency matrix preprocessing utilities."""
import numpy as np
import scipy.sparse as sp

from .sparse import sparse_to_tuple


def adj_to_bias(adj, sizes, nhood=1):
    """
    Prepare adjacency matrix by expanding up to a given neighbourhood.
    This will insert loops on every node.
    Finally, the matrix is converted to bias vectors.
    Expected shape: [graph, nodes, nodes]
    """
    nb_graphs = adj.shape[0]
    mt = np.empty(adj.shape)
    for g in range(nb_graphs):
        mt[g] = np.eye(adj.shape[1])
        for _ in range(nhood):
            mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
        for i in range(sizes[g]):
            for j in range(sizes[g]):
                if mt[g][i][j] > 0.0:
                    mt[g][i][j] = 1.0
    return -1e9 * (1.0 - mt)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for the simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def normalize_directed_adj(adj):
    """Normalize adjacency matrix for directed graphs (row normalization by out-degree)."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    # Avoid division by zero
    rowsum[rowsum == 0] = 1
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(adj).tocoo()


def preprocess_directed_adj(adj):
    """
    Preprocessing of adjacency matrix for the directed GCN model.
    Returns both in and out adjacency matrices with self-loops.
    """
    # Add self-loops
    adj_with_self = adj + sp.eye(adj.shape[0])

    # Create out-adjacency (original direction)
    adj_out = normalize_directed_adj(adj_with_self)

    # Create in-adjacency (transpose direction)
    adj_in = normalize_directed_adj(adj_with_self.T)

    return sparse_to_tuple(adj_in), sparse_to_tuple(adj_out)
