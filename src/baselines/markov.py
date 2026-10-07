"""Higher-Order Markov Chain baseline for next-port prediction.

Two classes:
- MarkovBaseline (order=1): builds a W x W transition count matrix;
  score_matrix() is used directly for dot-product evaluation.
- HigherOrderMarkovBaseline (order=n): builds n-gram count dicts for orders 1...n;
  predict(context, candidates) uses Stupid Backoff for sequence evaluation;
  score_matrix() returns the order-1 count matrix (used for dot-product).
"""
from collections import defaultdict
from typing import Dict, List, Optional
import numpy as np


class MarkovBaseline:
    """First-order Markov chain: W x W transition count matrix over warm ports.

    Fit on training transitions and used directly as a dot-product score matrix
    (row i = scores for all destinations given origin i).  Also supports
    sequence-based prediction (predict uses the last context port as origin).
    """

    def __init__(self):
        self._score_mat: Optional[np.ndarray] = None
        self._W: Optional[int] = None
        self._global_to_warm: Optional[Dict[int, int]] = None

    def fit(self, train_transitions: np.ndarray, warm_port_idx: np.ndarray) -> None:
        """Build transition count matrix.

        Args:
            train_transitions: array of (origin_global, dest_global, vessel_id)
            warm_port_idx: sorted array of warm global port indices
        """
        global_to_warm = {int(g): w for w, g in enumerate(warm_port_idx)}
        self._global_to_warm = global_to_warm
        W = len(warm_port_idx)
        self._W = W

        mat = np.zeros((W, W), dtype=np.float32)
        for t in train_transitions:
            src_w = global_to_warm.get(int(t[0]))
            dst_w = global_to_warm.get(int(t[1]))
            if src_w is not None and dst_w is not None:
                mat[src_w, dst_w] += 1.0

        self._score_mat = mat

    def score_matrix(self) -> np.ndarray:
        """Return (W, W) transition count matrix (ranking-invariant, no normalization)."""
        assert self._score_mat is not None, "Call fit() first"
        return self._score_mat

    def predict(self, context_warm: List[int], candidates: List[int]) -> np.ndarray:
        """Score candidates given a context (uses last port as origin).

        Args:
            context_warm: list of warm port indices (last = most recent)
            candidates: list of warm port indices to score (typically a list(range(W)))

        Returns:
            scores: (len(candidates),) float32 — higher is better
        """
        assert self._score_mat is not None, "Call fit() first"
        if not context_warm:
            return np.zeros(len(candidates), dtype=np.float32)
        last = context_warm[-1]
        # Select the requested candidates from the score row. For the standard
        # full-range candidate list this equals the whole row.
        return self._score_mat[last][np.asarray(candidates, dtype=np.int64)]


class HigherOrderMarkovBaseline:
    """n-gram Markov chain with Stupid Backoff for next-port prediction.

    Builds count dictionaries for orders 1..n from training trajectories.
    At prediction time, tries the highest order first and falls back to
    lower orders with a 0.4 discount multiplier per backoff step.

    Also exposes score_matrix() (order-1 counts) for the dot-product framing.
    """

    def __init__(self, order: int = 2):
        assert order >= 2, "Use MarkovBaseline for order=1"
        self.order = order
        # counts[k][(p_{t-k}, ..., p_{t-1})] -> {p_t: count}
        self.counts: Dict[int, Dict] = {}
        self._W: Optional[int] = None
        self._global_to_warm: Optional[Dict[int, int]] = None
        # fallback_counts[k] = number of predictions resolved at order k
        self.fallback_counts: Dict[int, int] = {}

    def fit(self, train_transitions: np.ndarray, warm_port_idx: np.ndarray) -> None:
        """Build n-gram count dictionaries from training trajectories.

        Args:
            train_transitions: array of (origin_global, dest_global, vessel_id)
            warm_port_idx: sorted array of warm global port indices
        """
        global_to_warm = {int(g): w for w, g in enumerate(warm_port_idx)}
        self._global_to_warm = global_to_warm
        self._W = len(warm_port_idx)

        self.counts = {k: defaultdict(lambda: defaultdict(int)) for k in range(1, self.order + 1)}

        # Build per-vessel trajectories. NOTE: this assumes the transition array
        # is time-sorted per vessel and contiguous (dest_i == origin_{i+1}), which
        # holds for the cached temporal splits produced by the preprocessing
        # pipeline. Unsorted or gapped input would silently yield wrong n-grams.
        vessel_trajectories: Dict = defaultdict(list)
        for t in train_transitions:
            origin, dest, vessel_id = int(t[0]), int(t[1]), t[2]
            vessel_trajectories[vessel_id].append((origin, dest))

        for vessel_id, traj in vessel_trajectories.items():
            # Port sequence: origin of the first step + all destinations.
            # Non-warm ports are removed and the sequence is re-joined, so an
            # n-gram may span a removed port (deliberate modeling choice, kept
            # for consistency with the reported results).
            ports_global = [traj[0][0]] + [dst for _, dst in traj]
            ports_warm = [global_to_warm[p] for p in ports_global if p in global_to_warm]

            if len(ports_warm) < 2:
                continue

            for i in range(1, len(ports_warm)):
                target = ports_warm[i]
                for k in range(1, self.order + 1):
                    if i >= k:
                        ctx = tuple(ports_warm[i - k:i])
                        self.counts[k][ctx][target] += 1

    def reset_fallback_counts(self) -> None:
        """Reset fallback counters. Call before each evaluation pass."""
        self.fallback_counts = {k: 0 for k in range(1, self.order + 1)}
        self.fallback_counts[0] = 0  # unresolved (zero context or no counts)

    def score_matrix(self) -> np.ndarray:
        """Return (W, W) order-1 transition count matrix for dot-product evaluation."""
        assert self._W is not None, "Call fit() first"
        W = self._W
        mat = np.zeros((W, W), dtype=np.float32)
        for ctx, dst_counts in self.counts[1].items():
            src_w = ctx[0]
            for dst_w, cnt in dst_counts.items():
                mat[src_w, dst_w] = cnt
        return mat

    def predict(self, context_warm: List[int], candidates: List[int]) -> np.ndarray:
        """Score candidates using Stupid Backoff over n-grams.

        Tries highest order first; backs off to lower orders multiplying by
        0.4 per step.  Returns the scores from the first order that has counts
        for the given context, scaled by the accumulated discount.

        Args:
            context_warm: list of warm port indices (last = most recent)
            candidates: list of warm port indices to score (typically list(range(W)))

        Returns:
            scores: (len(candidates),) float32 — higher is better (0 if no match)
        """
        n_cands = len(candidates)
        discount = 1.0

        for k in range(self.order, 0, -1):
            if len(context_warm) >= k:
                ctx = tuple(context_warm[-k:])
                counts_for_ctx = self.counts.get(k, {}).get(ctx)
                if counts_for_ctx:
                    # Score in full warm-index space, then select the requested
                    # candidates (equal to the whole vector for the standard
                    # full-range candidate list).
                    scores_full = np.zeros(self._W, dtype=np.float32)
                    total = sum(counts_for_ctx.values())
                    for dst_w, cnt in counts_for_ctx.items():
                        scores_full[dst_w] = discount * cnt / total
                    if self.fallback_counts:
                        self.fallback_counts[k] = self.fallback_counts.get(k, 0) + 1
                    return scores_full[np.asarray(candidates, dtype=np.int64)]
            discount *= 0.4

        if self.fallback_counts:
            self.fallback_counts[0] = self.fallback_counts.get(0, 0) + 1
        return np.zeros(n_cands, dtype=np.float32)
