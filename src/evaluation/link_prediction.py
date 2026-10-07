"""Link prediction (next-port prediction) evaluation for learned embeddings.

Three evaluation framings:
- Framing A (Dot-Product): Score embed[origin] @ embed[candidates].T, rank
- Framing B (MLP): Train MLP on frozen embeddings to predict the next port
- Framing C (LSTM): Use vessel trajectory sequences to predict the next port

All framings operate on frozen node embeddings from DGI.
"""
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class LinkPredictionResults:
    """Container for link prediction evaluation results."""
    hits_at_1: float
    hits_at_5: float
    hits_at_10: float
    mrr: float
    num_transitions: int
    framing: str  # "dot_product", "mlp", "lstm"
    per_vessel_type: Optional[Dict] = None


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _compute_ranks_from_scores(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute 1-indexed ranks from a score matrix batch.

    Args:
        scores: (B, W) score matrix — higher is better
        targets: (B,) target indices within warm port list

    Returns:
        ranks: (B,) 1-indexed ranks
    """
    sorted_indices = torch.argsort(scores, dim=1, descending=True)
    target_col = targets.unsqueeze(1)
    matches = (sorted_indices == target_col)
    ranks = matches.float().argmax(dim=1) + 1
    return ranks


def compute_ranking_metrics(
    scores: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Compute Hits@K and MRR from score matrix.

    For small tensors only (tests, small eval sets). For large evaluation
    sets, use compute_ranking_metrics_batched instead.

    Args:
        scores: (B, W) score matrix — higher is better
        targets: (B,) target indices within warm port list (0..W-1)

    Returns:
        Dict with hits_at_1, hits_at_5, hits_at_10, mrr
    """
    ranks = _compute_ranks_from_scores(scores, targets)
    return _metrics_from_ranks(ranks)


def _metrics_from_ranks(ranks: torch.Tensor) -> Dict[str, float]:
    """Compute Hits@K and MRR from a 1D tensor of 1-indexed ranks."""
    return {
        "hits_at_1": (ranks <= 1).float().mean().item(),
        "hits_at_5": (ranks <= 5).float().mean().item(),
        "hits_at_10": (ranks <= 10).float().mean().item(),
        "mrr": (1.0 / ranks.float()).mean().item(),
    }


def _metrics_from_ranks_for_group(ranks: torch.Tensor, indices: List[int]) -> Dict[str, float]:
    """Compute metrics for a subset of transitions identified by indices."""
    group_ranks = ranks[indices]
    metrics = _metrics_from_ranks(group_ranks)
    metrics["count"] = len(indices)
    return metrics


def compute_per_vessel_type_metrics(
    ranks: torch.Tensor,
    vessel_types: List[str],
    min_count: int = 100,
) -> Dict[str, Dict]:
    """Compute ranking metrics grouped by vessel type.

    Args:
        ranks: (B,) 1-indexed ranks
        vessel_types: (B,) list of vessel type strings
        min_count: minimum transitions to report a vessel type

    Returns:
        Dict of {vessel_type: {hits_at_1, ..., mrr, count}}
    """
    groups = defaultdict(list)
    for i, vt in enumerate(vessel_types):
        groups[vt].append(i)

    results = {}
    for vt, indices in sorted(groups.items()):
        if len(indices) < min_count:
            continue
        results[vt] = _metrics_from_ranks_for_group(ranks, indices)

    return results


def _score_and_rank_batched(
    score_fn,
    n_items: int,
    targets: torch.Tensor,
    batch_size: int = 10000,
) -> torch.Tensor:
    """Compute ranks by scoring in batches, never materializing full matrix.

    Args:
        score_fn: callable(start, end) -> (batch_size, W) score tensor
        n_items: total number of items to score
        targets: (n_items,) target indices
        batch_size: items per batch

    Returns:
        ranks: (n_items,) 1-indexed ranks
    """
    all_ranks = []
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        batch_scores = score_fn(start, end)  # (B, W)
        batch_targets = targets[start:end]
        batch_ranks = _compute_ranks_from_scores(batch_scores, batch_targets)
        all_ranks.append(batch_ranks)
    return torch.cat(all_ranks, dim=0)


def evaluate_score_matrix(
    score_matrix: np.ndarray,
    test_transitions: np.ndarray,
    warm_port_indices: np.ndarray,
    global_to_warm: Dict[int, int],
    vessel_type_map: Optional[Dict] = None,
    batch_size: int = 10000,
    device: Optional[torch.device] = None,
) -> "LinkPredictionResults":
    """Evaluate dot-product framing using a pre-computed (W, W) score matrix.

    Used by Markov baselines that produce a score matrix directly rather than
    node embeddings.  For each test transition (origin → dest), scores are
    looked up as score_matrix[origin_warm, :].

    Args:
        score_matrix: (W, W) float array — score_matrix[i, j] = score(i→j)
        test_transitions: array of (origin_global, dest_global, vessel_id)
        warm_port_indices: sorted array of warm global port indices
        global_to_warm: dict mapping global port index to warm index
        vessel_type_map: optional {vessel_id_str: vessel_type_str}
        batch_size: number of transitions per scoring batch
        device: torch device (defaults to CPU)

    Returns:
        LinkPredictionResults with framing="dot_product"
    """
    if device is None:
        device = torch.device("cpu")

    score_mat_t = torch.FloatTensor(score_matrix).to(device)  # (W, W)

    origins_warm = torch.LongTensor(
        [global_to_warm[int(t[0])] for t in test_transitions]
    ).to(device)
    targets_warm = torch.LongTensor(
        [global_to_warm[int(t[1])] for t in test_transitions]
    ).to(device)

    vessel_types = None
    if vessel_type_map is not None:
        vessel_types = [vessel_type_map.get(str(t[2]), "Unknown") for t in test_transitions]

    def score_fn(start, end):
        return score_mat_t[origins_warm[start:end]]  # (B, W)

    ranks = _score_and_rank_batched(score_fn, len(test_transitions), targets_warm, batch_size)
    metrics = _metrics_from_ranks(ranks)

    per_vt = None
    if vessel_types is not None:
        per_vt = compute_per_vessel_type_metrics(ranks, vessel_types)

    return LinkPredictionResults(
        hits_at_1=metrics["hits_at_1"],
        hits_at_5=metrics["hits_at_5"],
        hits_at_10=metrics["hits_at_10"],
        mrr=metrics["mrr"],
        num_transitions=len(test_transitions),
        framing="dot_product",
        per_vessel_type=per_vt,
    )


def evaluate_sequence_predictor(
    predictor,
    test_transitions: np.ndarray,
    warm_port_indices: np.ndarray,
    global_to_warm: Dict[int, int],
    max_len: int = 5,
    vessel_type_map: Optional[Dict] = None,
) -> "LinkPredictionResults":
    """Evaluate a sequence predictor on the LSTM framing without training.

    Used by Markov baselines that predict the next port given a context of
    previous ports.  The predictor must implement:
        predict(context_warm: List[int], candidates: List[int]) -> np.ndarray

    Args:
        predictor: object with predict(context_warm, candidates) -> (W,) scores
        test_transitions: array of (origin_global, dest_global, vessel_id)
        warm_port_indices: sorted array of warm global port indices
        global_to_warm: dict mapping global port index to warm index
        max_len: context window length (same as LSTM max_len)
        vessel_type_map: optional {vessel_id_str: vessel_type_str}

    Returns:
        LinkPredictionResults with framing="lstm"
    """
    W = len(warm_port_indices)
    candidates = list(range(W))

    # Reset fallback counters if predictor supports them (HigherOrderMarkovBaseline)
    if hasattr(predictor, "reset_fallback_counts"):
        predictor.reset_fallback_counts()

    test_seqs = build_sequences(test_transitions, max_len)
    if not test_seqs:
        return LinkPredictionResults(
            hits_at_1=0.0, hits_at_5=0.0, hits_at_10=0.0, mrr=0.0,
            num_transitions=0, framing="lstm",
        )

    all_ranks = []
    all_vessel_types = []
    n_skipped = 0

    for i, (context_global, target_global, vessel_id) in enumerate(test_seqs):
        target_warm = global_to_warm.get(int(target_global))
        if target_warm is None:
            n_skipped += 1
            continue

        context_warm = [global_to_warm[p] for p in context_global if p in global_to_warm]

        scores = predictor.predict(context_warm, candidates)  # (W,) float32

        # Rank: number of candidates scoring strictly higher + 1
        rank = int(np.sum(scores > scores[target_warm])) + 1
        all_ranks.append(rank)

        if vessel_type_map is not None:
            all_vessel_types.append(vessel_type_map.get(str(vessel_id), "Unknown"))

        if (i + 1) % 200_000 == 0:
            print(f"    Processed {i + 1:,}/{len(test_seqs):,} sequences...")

    if not all_ranks:
        return LinkPredictionResults(
            hits_at_1=0.0, hits_at_5=0.0, hits_at_10=0.0, mrr=0.0,
            num_transitions=0, framing="lstm",
        )

    if n_skipped > 0:
        print(f"    Warning: skipped {n_skipped} sequences (target not in warm ports)")

    # Print fallback distribution for higher-order Markov models
    if hasattr(predictor, "fallback_counts") and predictor.fallback_counts:
        fc = predictor.fallback_counts
        total_preds = sum(fc.values())
        if total_preds > 0:
            parts = []
            for k in sorted(fc.keys(), reverse=True):
                if k == 0:
                    parts.append(f"unresolved={fc[k]/total_preds:.1%}")
                else:
                    parts.append(f"k={k}: {fc[k]/total_preds:.1%}")
            print(f"    Fallback distribution: {', '.join(parts)}")

    ranks_t = torch.LongTensor(all_ranks)
    metrics = _metrics_from_ranks(ranks_t)

    per_vt = None
    if vessel_type_map is not None and all_vessel_types:
        per_vt = compute_per_vessel_type_metrics(ranks_t, all_vessel_types)

    return LinkPredictionResults(
        hits_at_1=metrics["hits_at_1"],
        hits_at_5=metrics["hits_at_5"],
        hits_at_10=metrics["hits_at_10"],
        mrr=metrics["mrr"],
        num_transitions=len(all_ranks),
        framing="lstm",
        per_vessel_type=per_vt,
    )


def load_transitions(cache_path: str, split: str) -> np.ndarray:
    """Load transition numpy array from cache.

    Args:
        cache_path: path to processed cache directory
        split: one of 'train', 'val', 'test'

    Returns:
        numpy array of (origin_idx, dest_idx, vessel_id) tuples
    """
    path = Path(cache_path) / f"{split}_transitions.npy"
    if path.exists():
        return np.load(path, allow_pickle=True)
    return np.array([])


def _build_warm_port_mapping(
    train_transitions: np.ndarray,
    val_transitions: np.ndarray,
    test_transitions: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Build warm port index list and global-to-warm mapping.

    Args:
        train/val/test_transitions: arrays of (origin, dest, vessel_id) tuples

    Returns:
        warm_ports: sorted array of unique global port indices
        global_to_warm: dict mapping global index to warm-port-list index
    """
    all_ports = set()
    for transitions in [train_transitions, val_transitions, test_transitions]:
        if len(transitions) == 0:
            continue
        for t in transitions:
            all_ports.add(int(t[0]))
            all_ports.add(int(t[1]))

    warm_ports = np.array(sorted(all_ports))
    global_to_warm = {int(g): w for w, g in enumerate(warm_ports)}
    return warm_ports, global_to_warm


# ---------------------------------------------------------------------------
# Framing A: Dot-Product Link Predictor
# ---------------------------------------------------------------------------

class DotProductLinkPredictor:
    """Score transitions by dot product between origin and destination embeddings."""

    def evaluate(
        self,
        embeddings: np.ndarray,
        test_transitions: np.ndarray,
        warm_port_indices: np.ndarray,
        global_to_warm: Dict[int, int],
        vessel_type_map: Optional[Dict] = None,
        batch_size: int = 10000,
        device: Optional[torch.device] = None,
        normalize: bool = True,
        exclude_self: bool = True,
    ) -> LinkPredictionResults:
        """Evaluate dot-product link prediction.

        Scores and ranks are computed in batches to avoid OOM on large
        transition sets (~1.2M transitions x 3,451 candidates).

        Args:
            normalize: if True, L2-normalize embeddings before scoring, so the
                score is cosine similarity rather than the raw inner product.
                Removes the embedding-norm confound (a few high-norm hub ports
                otherwise capture the top rank for most origins).
            exclude_self: if True, mask the origin port out of its own candidate
                ranking. Self-transitions are removed from the data, so the true
                destination is never the origin; leaving the origin in the
                candidate set penalizes embeddings whose nearest neighbor is
                themselves.

        Note: the defaults (True, True) are the fair retrieval setting the
        paper describes and reports. Passing normalize=False, exclude_self=False
        reproduces the raw inner product over all warm candidates.
        """
        if device is None:
            device = torch.device("cpu")

        embed_t = torch.FloatTensor(embeddings).to(device)
        if normalize:
            embed_t = embed_t / embed_t.norm(dim=1, keepdim=True).clamp_min(1e-12)
        candidate_embeds = embed_t[warm_port_indices]  # (W, D)

        origins = np.array([int(t[0]) for t in test_transitions])
        dests = np.array([int(t[1]) for t in test_transitions])
        targets_warm = torch.LongTensor([global_to_warm[d] for d in dests]).to(device)

        # Each origin's own column in the (B, W) candidate matrix, for masking.
        origins_warm = None
        if exclude_self:
            origins_warm = torch.LongTensor(
                [global_to_warm[int(o)] for o in origins]
            ).to(device)

        vessel_types = None
        if vessel_type_map is not None:
            vessel_types = [vessel_type_map.get(str(t[2]), "Unknown") for t in test_transitions]

        def score_fn(start, end):
            batch_origins = torch.LongTensor(origins[start:end]).to(device)
            origin_embeds = embed_t[batch_origins]
            scores = origin_embeds @ candidate_embeds.T  # (B, W)
            if exclude_self:
                rows = torch.arange(end - start, device=device)
                scores[rows, origins_warm[start:end]] = float("-inf")
            return scores

        ranks = _score_and_rank_batched(score_fn, len(test_transitions), targets_warm, batch_size)
        metrics = _metrics_from_ranks(ranks)

        per_vt = None
        if vessel_types is not None:
            per_vt = compute_per_vessel_type_metrics(ranks, vessel_types)

        return LinkPredictionResults(
            hits_at_1=metrics["hits_at_1"],
            hits_at_5=metrics["hits_at_5"],
            hits_at_10=metrics["hits_at_10"],
            mrr=metrics["mrr"],
            num_transitions=len(test_transitions),
            framing="dot_product",
            per_vessel_type=per_vt,
        )


# ---------------------------------------------------------------------------
# Framing B: MLP Link Predictor
# ---------------------------------------------------------------------------

class _NextPortMLP(nn.Module):
    """MLP: embed(origin) -> next port logits."""

    def __init__(self, embed_dim: int, num_warm_ports: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden, num_warm_ports),
        )

    def forward(self, x):
        return self.net(x)


def _eval_model_batched(model, embed_t, origins, targets, batch_size=10000):
    """Score with a model in batches, return ranks (never materializes full matrix)."""
    n = len(origins)

    def score_fn(start, end):
        batch_embeds = embed_t[origins[start:end]]
        return model(batch_embeds)

    with torch.no_grad():
        return _score_and_rank_batched(score_fn, n, targets, batch_size)


class MLPLinkPredictor:
    """Train an MLP on frozen embeddings to predict the next port."""

    def train_and_evaluate(
        self,
        embeddings: np.ndarray,
        train_transitions: np.ndarray,
        val_transitions: np.ndarray,
        test_transitions: np.ndarray,
        warm_port_indices: np.ndarray,
        global_to_warm: Dict[int, int],
        vessel_type_map: Optional[Dict] = None,
        n_runs: int = 5,
        epochs: int = 100,
        lr: float = 0.001,
        patience: int = 10,
        batch_size: int = 4096,
        device: Optional[torch.device] = None,
    ) -> LinkPredictionResults:
        """Train MLP and evaluate, return mean results over n_runs."""
        if device is None:
            device = torch.device("cpu")

        embed_t = torch.FloatTensor(embeddings).to(device)
        W = len(warm_port_indices)
        D = embeddings.shape[1]

        def _make_tensors(transitions):
            origins = torch.LongTensor([int(t[0]) for t in transitions])
            targets = torch.LongTensor([global_to_warm[int(t[1])] for t in transitions])
            return origins.to(device), targets.to(device)

        train_origins, train_targets = _make_tensors(train_transitions)
        val_origins, val_targets = _make_tensors(val_transitions)
        test_origins, test_targets = _make_tensors(test_transitions)

        vessel_types = None
        if vessel_type_map is not None:
            vessel_types = [vessel_type_map.get(str(t[2]), "Unknown") for t in test_transitions]

        all_metrics = []

        for run in range(n_runs):
            model = _NextPortMLP(D, W).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()

            best_val_mrr = 0.0
            patience_counter = 0
            best_state = None

            for epoch in range(epochs):
                # Train
                model.train()
                perm = torch.randperm(len(train_origins), device=device)

                for start in range(0, len(perm), batch_size):
                    idx = perm[start:start + batch_size]
                    origin_embeds = embed_t[train_origins[idx]]
                    logits = model(origin_embeds)
                    loss = loss_fn(logits, train_targets[idx])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Validate (batched ranking)
                model.eval()
                val_ranks = _eval_model_batched(model, embed_t, val_origins, val_targets, batch_size)
                val_mrr = (1.0 / val_ranks.float()).mean().item()

                if val_mrr > best_val_mrr:
                    best_val_mrr = val_mrr
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Load best model and evaluate on test (batched)
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            test_ranks = _eval_model_batched(model, embed_t, test_origins, test_targets, batch_size)
            run_metrics = _metrics_from_ranks(test_ranks)

            all_metrics.append(run_metrics)
            print(f"  MLP run {run + 1}/{n_runs}: "
                  f"Hits@1={run_metrics['hits_at_1']:.4f}, "
                  f"MRR={run_metrics['mrr']:.4f} "
                  f"(val MRR={best_val_mrr:.4f})")

        # Average across runs
        mean_metrics = {
            k: np.mean([m[k] for m in all_metrics])
            for k in ["hits_at_1", "hits_at_5", "hits_at_10", "mrr"]
        }

        # Per-vessel-type from last run's test ranks
        per_vt = None
        if vessel_types is not None:
            per_vt = compute_per_vessel_type_metrics(test_ranks, vessel_types)

        return LinkPredictionResults(
            hits_at_1=mean_metrics["hits_at_1"],
            hits_at_5=mean_metrics["hits_at_5"],
            hits_at_10=mean_metrics["hits_at_10"],
            mrr=mean_metrics["mrr"],
            num_transitions=len(test_transitions),
            framing="mlp",
            per_vessel_type=per_vt,
        )


# ---------------------------------------------------------------------------
# Framing C: LSTM Link Predictor
# ---------------------------------------------------------------------------

class _NextPortLSTM(nn.Module):
    """LSTM: sequence of port embeddings -> next port logits."""

    def __init__(self, embed_dim: int, num_warm_ports: int, hidden: int = 256, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden, num_warm_ports)

    def forward(self, x):
        # x: (B, seq_len, D)
        _, (h_n, _) = self.lstm(x)  # h_n: (num_layers, B, hidden)
        out = self.fc(h_n[-1])  # (B, num_warm_ports)
        return out


def build_sequences(
    transitions: np.ndarray,
    max_len: int = 5,
) -> List[Tuple[List[int], int, object]]:
    """Build sliding-window sequences from transitions.

    Groups transitions by vessel_id (already sorted by departure_time),
    then creates sliding windows of length max_len with stride 1.
    Each sequence predicts the next port after the window.

    Args:
        transitions: array of (origin_idx, dest_idx, vessel_id) tuples
        max_len: context window length

    Returns:
        List of (context_port_indices, target_port_idx, vessel_id) tuples.
        context_port_indices is a list of global port indices forming the
        trajectory context, and target_port_idx is the next port to predict.
    """
    if len(transitions) == 0:
        return []

    # Group by vessel_id, preserving order
    vessel_trajectories = defaultdict(list)
    for t in transitions:
        origin, dest, vessel_id = int(t[0]), int(t[1]), t[2]
        vessel_trajectories[vessel_id].append((origin, dest))

    sequences = []
    for vessel_id, traj in vessel_trajectories.items():
        # Build port sequence: origin of first, then all destinations
        ports = [traj[0][0]] + [dest for _, dest in traj]

        # Sliding window
        if len(ports) < 2:
            continue

        for i in range(len(ports) - 1):
            start = max(0, i - max_len + 1)
            context = ports[start:i + 1]
            target = ports[i + 1]
            sequences.append((context, target, vessel_id))

    return sequences


def _eval_lstm_batched(model, embed_t, ctx_indices, targets, batch_size=4096):
    """Score LSTM model in batches, return ranks."""
    n = len(ctx_indices)

    def score_fn(start, end):
        batch_embeds = embed_t[ctx_indices[start:end]]
        return model(batch_embeds)

    with torch.no_grad():
        return _score_and_rank_batched(score_fn, n, targets, batch_size)


class LSTMLinkPredictor:
    """Train an LSTM on vessel trajectory sequences to predict the next port."""

    def train_and_evaluate(
        self,
        embeddings: np.ndarray,
        train_transitions: np.ndarray,
        val_transitions: np.ndarray,
        test_transitions: np.ndarray,
        warm_port_indices: np.ndarray,
        global_to_warm: Dict[int, int],
        vessel_type_map: Optional[Dict] = None,
        n_runs: int = 5,
        max_len: int = 5,
        epochs: int = 50,
        lr: float = 0.001,
        patience: int = 10,
        batch_size: int = 4096,
        device: Optional[torch.device] = None,
    ) -> LinkPredictionResults:
        """Train LSTM and evaluate, return mean results over n_runs."""
        if device is None:
            device = torch.device("cpu")

        embed_t = torch.FloatTensor(embeddings).to(device)
        W = len(warm_port_indices)
        D = embeddings.shape[1]

        # Build sequences
        train_seqs = build_sequences(train_transitions, max_len)
        val_seqs = build_sequences(val_transitions, max_len)
        test_seqs = build_sequences(test_transitions, max_len)

        if not train_seqs or not test_seqs:
            print("  Warning: no sequences could be built. Skipping LSTM evaluation.")
            return LinkPredictionResults(
                hits_at_1=0.0, hits_at_5=0.0, hits_at_10=0.0, mrr=0.0,
                num_transitions=0, framing="lstm",
            )

        def _pad_and_batch(seqs):
            """Pad sequences and create batched tensors."""
            contexts = [s[0] for s in seqs]
            targets_warm = [global_to_warm[s[1]] for s in seqs]
            lengths = [len(c) for c in contexts]
            max_seq_len = max(lengths)

            padded = np.zeros((len(seqs), max_seq_len), dtype=np.int64)
            for i, (ctx, l) in enumerate(zip(contexts, lengths)):
                padded[i, max_seq_len - l:] = ctx  # right-align

            return (
                torch.LongTensor(padded).to(device),
                torch.LongTensor(targets_warm).to(device),
            )

        train_ctx, train_tgt = _pad_and_batch(train_seqs)
        val_ctx, val_tgt = _pad_and_batch(val_seqs)
        test_ctx, test_tgt = _pad_and_batch(test_seqs)

        # Vessel types for test sequences
        vessel_types = None
        if vessel_type_map is not None:
            vessel_types = [
                vessel_type_map.get(str(s[2]), "Unknown") for s in test_seqs
            ]

        all_metrics = []

        for run in range(n_runs):
            model = _NextPortLSTM(D, W).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.CrossEntropyLoss()

            best_val_mrr = 0.0
            patience_counter = 0
            best_state = None

            n_train = len(train_ctx)

            for epoch in range(epochs):
                # Train
                model.train()
                perm = torch.randperm(n_train, device=device)

                for start in range(0, n_train, batch_size):
                    idx = perm[start:start + batch_size]
                    batch_embeds = embed_t[train_ctx[idx]]
                    logits = model(batch_embeds)
                    loss = loss_fn(logits, train_tgt[idx])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Validate (batched ranking)
                model.eval()
                val_ranks = _eval_lstm_batched(model, embed_t, val_ctx, val_tgt, batch_size)
                val_mrr = (1.0 / val_ranks.float()).mean().item()

                if val_mrr > best_val_mrr:
                    best_val_mrr = val_mrr
                    patience_counter = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            # Load best and evaluate on test (batched)
            if best_state is not None:
                model.load_state_dict(best_state)
            model.eval()
            test_ranks = _eval_lstm_batched(model, embed_t, test_ctx, test_tgt, batch_size)
            run_metrics = _metrics_from_ranks(test_ranks)

            all_metrics.append(run_metrics)
            print(f"  LSTM run {run + 1}/{n_runs}: "
                  f"Hits@1={run_metrics['hits_at_1']:.4f}, "
                  f"MRR={run_metrics['mrr']:.4f} "
                  f"(val MRR={best_val_mrr:.4f})")

        # Average across runs
        mean_metrics = {
            k: np.mean([m[k] for m in all_metrics])
            for k in ["hits_at_1", "hits_at_5", "hits_at_10", "mrr"]
        }

        # Per-vessel-type from last run
        per_vt = None
        if vessel_types is not None:
            per_vt = compute_per_vessel_type_metrics(test_ranks, vessel_types)

        return LinkPredictionResults(
            hits_at_1=mean_metrics["hits_at_1"],
            hits_at_5=mean_metrics["hits_at_5"],
            hits_at_10=mean_metrics["hits_at_10"],
            mrr=mean_metrics["mrr"],
            num_transitions=len(test_seqs),
            framing="lstm",
            per_vessel_type=per_vt,
        )
