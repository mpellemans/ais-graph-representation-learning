"""PortCity2Vec baseline (Cai et al., 2026).

Faithful re-implementation of the embedding-learning component of PortCity2Vec
("Enhancing the interpretability of port economic modeling via implicit spatial
relationship discovery", Int. J. Applied Earth Obs. & Geoinformation, 2026).

Only the representation-learning half of PortCity2Vec is reproduced here; the
downstream spatial-XGBoost / GeoShapley economic model is out of scope (it
requires city-level socioeconomic indicators we do not have for this network).

Method (paper Section 3.1):
  1. Model the shipping mobility network as a directed *weighted* graph
     G = (V, E, W), where W_{i->j} is the frequency of vessel mobility from
     port i to port j (here: total journeys on the directed edge).
  2. Generate node sequences with a *weighted* biased (Node2Vec) random walk.
     The unnormalized transition probability is
         pi_{i->j} = alpha_pq(t, j) * W_{i->j},
     with the standard second-order bias
         alpha = 1/p  if j == t        (return to the previous node, d_tj = 0)
         alpha = 1    if j ~ t         (j is a neighbor of t,        d_tj = 1)
         alpha = 1/q  otherwise        (                            d_tj = 2).
  3. Learn embeddings from these sequences with Skip-gram (negative sampling).

Differences from our Node2Vec baseline are exactly two controlled factors:
PortCity2Vec (a) weights the walk by mobility frequency W and (b) uses a biased
walk (p, q != 1). Everything else (the filtered graph topology, observation
window, walk length / number of walks, embedding dimension) is held identical so
the comparison in Table 3 isolates the method.

The skip-gram is implemented in pure PyTorch (GPU-capable) to avoid an extra
gensim/torch-cluster dependency on the cluster.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _sample(prob: np.ndarray) -> int:
    """Sample an index from a normalized probability vector (boundary-safe)."""
    idx = int(np.searchsorted(np.cumsum(prob), np.random.random()))
    return min(idx, prob.size - 1)


class PortCity2VecBaseline:
    """Weighted biased-random-walk + Skip-gram embeddings (PortCity2Vec).

    Args:
        embedding_dim: output embedding dimensionality (512 to match the other
            baselines; the paper uses 128 on a 51-city network).
        walk_length: steps per random walk.
        walks_per_node: number of walks started from each node.
        window_size: skip-gram context radius (pairs within +/- window_size).
        p: return parameter (paper regime: p < 1 favors backtracking).
        q: in-out parameter (paper regime: q < 1 favors outward exploration).
        num_negative_samples: negatives per positive skip-gram pair.
        epochs: skip-gram training epochs.
        lr: Adam learning rate.
        batch_size: skip-gram minibatch size (number of positive pairs).
        log_weights: if True, use log1p(W) instead of raw W as edge weights.
            Default False (raw frequency, faithful to the paper's Eq. 1).
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        walk_length: int = 80,
        walks_per_node: int = 10,
        window_size: int = 5,
        p: float = 0.5,
        q: float = 0.5,
        num_negative_samples: int = 5,
        epochs: int = 5,
        lr: float = 0.01,
        batch_size: int = 65536,
        log_weights: bool = False,
    ):
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.window_size = window_size
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.log_weights = log_weights

    # ------------------------------------------------------------------ #
    # Edge weights
    # ------------------------------------------------------------------ #
    def _edge_weights(self, dataset) -> np.ndarray:
        """Return per-edge mobility-frequency weights aligned to edge_index.

        Weight of edge (s, d) is the full-period total_journeys recorded in
        edges.parquet, the directed vessel-mobility frequency W_{s->d}. This is
        the same filtered topology every other baseline uses; only the weights
        are attached.
        """
        edge_index = dataset.edge_index  # (2, E)
        port_to_idx = getattr(dataset, "port_to_idx", None)
        if not port_to_idx:
            raise ValueError(
                "PortCity2Vec needs dataset.port_to_idx to map edges to "
                "total_journeys; cache metadata.json is missing port_to_idx."
            )
        idx_to_port = {v: k for k, v in port_to_idx.items()}

        edges_path = Path(dataset.data_dir) / "edges.parquet"
        edf = pd.read_parquet(
            edges_path, columns=["origin_port", "destination_port", "total_journeys"]
        )
        # Sum journeys per directed pair: robust to duplicate (origin, destination)
        # rows, where a plain dict comprehension would silently keep only the last
        # row; identical to a plain lookup when each pair appears once.
        grouped = edf.groupby(
            ["origin_port", "destination_port"], sort=False
        )["total_journeys"].sum()
        weight_lookup = {(o, d): float(w) for (o, d), w in grouped.items()}

        E = edge_index.shape[1]
        weights = np.empty(E, dtype=np.float64)
        n_missing = 0
        for e in range(E):
            s = int(edge_index[0, e])
            d = int(edge_index[1, e])
            w = weight_lookup.get((idx_to_port[s], idx_to_port[d]))
            if w is None:
                w = 1.0
                n_missing += 1
            weights[e] = w
        if n_missing:
            print(f"  [PortCity2Vec] {n_missing}/{E} edges missing weight; set to 1.0")
            if n_missing > 0.01 * E:
                raise ValueError(
                    f"PortCity2Vec: {n_missing}/{E} edges have no total_journeys "
                    "weight; edges.parquet does not match the cached graph."
                )

        if self.log_weights:
            weights = np.log1p(weights)
        # Guard against non-positive weights (kept strictly positive for sampling)
        weights = np.maximum(weights, 1e-12)
        return weights

    # ------------------------------------------------------------------ #
    # Weighted biased random walk
    # ------------------------------------------------------------------ #
    def _build_adjacency(self, edge_index, edge_weights, num_nodes):
        """Per-node sorted out-neighbor arrays and matching weight arrays."""
        neighbors = [None] * num_nodes
        weights = [None] * num_nodes
        src = edge_index[0]
        dst = edge_index[1]
        order = np.argsort(src, kind="stable")
        src_s = src[order]
        dst_s = dst[order]
        w_s = edge_weights[order]
        # boundaries of each source block
        boundaries = np.searchsorted(src_s, np.arange(num_nodes + 1))
        for v in range(num_nodes):
            lo, hi = boundaries[v], boundaries[v + 1]
            if hi > lo:
                nb = dst_s[lo:hi]
                wv = w_s[lo:hi]
                sort_nb = np.argsort(nb, kind="stable")  # sorted for fast membership
                neighbors[v] = nb[sort_nb]
                weights[v] = wv[sort_nb]
            else:
                neighbors[v] = np.empty(0, dtype=np.int64)
                weights[v] = np.empty(0, dtype=np.float64)
        return neighbors, weights

    def _generate_walks(self, neighbors, weights, num_nodes):
        """Generate weighted, p/q-biased random walks. Returns a list of np.int64."""
        walks = []
        inv_p = 1.0 / self.p
        inv_q = 1.0 / self.q
        mark = np.zeros(num_nodes, dtype=bool)  # scratch buffer for d_tj == 1 test

        start_nodes = np.repeat(np.arange(num_nodes), self.walks_per_node)
        np.random.shuffle(start_nodes)

        for start in start_nodes:
            nbrs = neighbors[start]
            if nbrs.size == 0:
                walks.append(np.array([start], dtype=np.int64))
                continue

            walk = np.empty(self.walk_length, dtype=np.int64)
            walk[0] = start
            # first step: weighted, unbiased
            w = weights[start]
            prob = w / w.sum()
            cur = nbrs[_sample(prob)]
            walk[1] = cur
            prev = start

            length = 2
            for step in range(2, self.walk_length):
                nbrs = neighbors[cur]
                if nbrs.size == 0:
                    break
                w = weights[cur]
                # bias factors alpha_pq
                t_nbrs = neighbors[prev]
                mark[t_nbrs] = True
                alpha = np.where(mark[nbrs], 1.0, inv_q)
                mark[t_nbrs] = False
                alpha[nbrs == prev] = inv_p
                prob = w * alpha
                prob /= prob.sum()
                nxt = nbrs[_sample(prob)]
                walk[step] = nxt
                prev = cur
                cur = nxt
                length = step + 1

            walks.append(walk[:length].copy())

        return walks

    # ------------------------------------------------------------------ #
    # Skip-gram pair generation
    # ------------------------------------------------------------------ #
    def _build_pairs(self, walks):
        """Build (center, context) positive pairs within +/- window_size."""
        centers_parts = []
        contexts_parts = []
        w = self.window_size
        for walk in walks:
            L = walk.size
            if L < 2:
                continue
            for o in range(1, w + 1):
                if o >= L:
                    break
                a = walk[:-o]
                b = walk[o:]
                # both directions (skip-gram context is symmetric)
                centers_parts.append(a)
                contexts_parts.append(b)
                centers_parts.append(b)
                contexts_parts.append(a)
        if not centers_parts:
            raise RuntimeError("PortCity2Vec produced no training pairs (empty walks).")
        centers = np.concatenate(centers_parts)
        contexts = np.concatenate(contexts_parts)
        return centers, contexts

    def _unigram_table(self, walks, num_nodes):
        """Negative-sampling distribution: token frequency ^ 0.75."""
        counts = np.zeros(num_nodes, dtype=np.float64)
        for walk in walks:
            np.add.at(counts, walk, 1.0)
        probs = np.power(counts, 0.75)
        total = probs.sum()
        if total <= 0:
            probs = np.ones(num_nodes, dtype=np.float64)
            total = float(num_nodes)
        return probs / total

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def get_embeddings(self, dataset, device=None) -> np.ndarray:
        """Train PortCity2Vec and return (N, embedding_dim) float32 embeddings."""
        if device is None:
            device = torch.device("cpu")

        edge_index = np.asarray(dataset.edge_index, dtype=np.int64)
        N = dataset.num_nodes

        print("  [PortCity2Vec] Loading mobility-frequency edge weights...")
        edge_weights = self._edge_weights(dataset)
        print(
            f"  [PortCity2Vec] weights: min={edge_weights.min():.3g}, "
            f"median={np.median(edge_weights):.3g}, max={edge_weights.max():.3g}"
        )

        print(
            f"  [PortCity2Vec] Generating biased walks "
            f"(p={self.p}, q={self.q}, len={self.walk_length}, "
            f"per_node={self.walks_per_node})..."
        )
        neighbors, weights = self._build_adjacency(edge_index, edge_weights, N)
        walks = self._generate_walks(neighbors, weights, N)
        total_steps = sum(w.size for w in walks)
        print(f"  [PortCity2Vec] {len(walks):,} walks, {total_steps:,} total nodes visited")

        centers, contexts = self._build_pairs(walks)
        print(f"  [PortCity2Vec] {len(centers):,} skip-gram pairs (window={self.window_size})")
        neg_probs = torch.tensor(
            self._unigram_table(walks, N), dtype=torch.float32, device=device
        )

        # --- Skip-gram with negative sampling ---
        emb_in = torch.nn.Embedding(N, self.embedding_dim).to(device)
        emb_out = torch.nn.Embedding(N, self.embedding_dim).to(device)
        torch.nn.init.uniform_(
            emb_in.weight, -0.5 / self.embedding_dim, 0.5 / self.embedding_dim
        )
        torch.nn.init.zeros_(emb_out.weight)
        optimizer = torch.optim.Adam(
            list(emb_in.parameters()) + list(emb_out.parameters()), lr=self.lr
        )

        centers_t = torch.from_numpy(centers.astype(np.int64))
        contexts_t = torch.from_numpy(contexts.astype(np.int64))
        n_pairs = centers_t.numel()
        k = self.num_negative_samples

        for epoch in range(self.epochs):
            perm = torch.randperm(n_pairs)
            total_loss = 0.0
            n_batches = 0
            for start in range(0, n_pairs, self.batch_size):
                idx = perm[start:start + self.batch_size]
                c = centers_t[idx].to(device)
                o = contexts_t[idx].to(device)
                B = c.numel()

                neg = torch.multinomial(neg_probs, B * k, replacement=True).view(B, k)

                vc = emb_in(c)                      # (B, D)
                vo = emb_out(o)                     # (B, D)
                vneg = emb_out(neg)                 # (B, k, D)

                pos_score = (vc * vo).sum(dim=1)    # (B,)
                pos_loss = -F.logsigmoid(pos_score)
                neg_score = torch.bmm(vneg, vc.unsqueeze(2)).squeeze(2)  # (B, k)
                neg_loss = -F.logsigmoid(-neg_score).sum(dim=1)
                loss = (pos_loss + neg_loss).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_batches += 1

            print(
                f"  [PortCity2Vec] epoch {epoch + 1}/{self.epochs}: "
                f"loss={total_loss / max(n_batches, 1):.4f}"
            )

        emb_in.eval()
        with torch.no_grad():
            embeddings = emb_in.weight.detach().cpu().numpy()
        return embeddings.astype(np.float32)
