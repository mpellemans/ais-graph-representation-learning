"""Feature preprocessing utilities."""
import numpy as np
import scipy.sparse as sp

from .sparse import sparse_to_tuple


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation."""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)


def standardize_data(f, train_mask):
    """Standardize feature matrix and convert to tuple representation."""
    f = f.todense()
    mu = f[train_mask == True, :].mean(axis=0)
    sigma = f[train_mask == True, :].std(axis=0)
    f = f[:, np.squeeze(np.array(sigma > 0))]
    sigma = f[train_mask == True, :].std(axis=0)
    f = (f - mu) / sigma
    return f
