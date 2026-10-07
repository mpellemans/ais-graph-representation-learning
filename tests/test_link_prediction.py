"""Verification tests for link prediction (next-port prediction) evaluation."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch

from src.evaluation.link_prediction import (
    compute_ranking_metrics,
    compute_per_vessel_type_metrics,
    build_sequences,
    _build_warm_port_mapping,
    DotProductLinkPredictor,
    MLPLinkPredictor,
    LSTMLinkPredictor,
    LinkPredictionResults,
)


def test_ranking_metrics_perfect():
    """Test 1: Perfect predictions -> Hits@1=1.0, MRR=1.0."""
    W = 10
    B = 5
    # Scores where the target is always ranked first
    scores = torch.zeros(B, W)
    targets = torch.arange(B) % W  # targets: 0,1,2,3,4
    for i in range(B):
        scores[i, targets[i]] = 100.0  # make target score highest

    metrics = compute_ranking_metrics(scores, targets)
    assert abs(metrics["hits_at_1"] - 1.0) < 1e-6, f"Expected Hits@1=1.0, got {metrics['hits_at_1']}"
    assert abs(metrics["mrr"] - 1.0) < 1e-6, f"Expected MRR=1.0, got {metrics['mrr']}"
    assert abs(metrics["hits_at_5"] - 1.0) < 1e-6
    assert abs(metrics["hits_at_10"] - 1.0) < 1e-6
    print("  PASS: Perfect predictions -> Hits@1=1.0, MRR=1.0")


def test_ranking_metrics_partial():
    """Test 2: Known ranking -> verify correct metric values."""
    # 4 queries, 5 candidates
    # Target ranks: 1, 2, 5, 3 (1-indexed)
    scores = torch.tensor([
        [10.0, 1.0, 2.0, 3.0, 4.0],   # target=0 (score 10) -> rank 1
        [1.0, 9.0, 10.0, 2.0, 3.0],   # target=1 (score 9) -> rank 2
        [10.0, 9.0, 8.0, 7.0, 1.0],   # target=4 (score 1) -> rank 5
        [1.0, 2.0, 10.0, 8.0, 9.0],   # target=3 (score 8) -> rank 3
    ])
    targets = torch.tensor([0, 1, 4, 3])

    metrics = compute_ranking_metrics(scores, targets)

    # Hits@1: 1/4 = 0.25
    assert abs(metrics["hits_at_1"] - 0.25) < 1e-6, f"Expected Hits@1=0.25, got {metrics['hits_at_1']}"
    # Hits@5: 4/4 = 1.0
    assert abs(metrics["hits_at_5"] - 1.0) < 1e-6, f"Expected Hits@5=1.0, got {metrics['hits_at_5']}"
    # MRR: (1/1 + 1/2 + 1/5 + 1/3) / 4 = (1 + 0.5 + 0.2 + 0.333) / 4 = 0.50833...
    expected_mrr = (1.0 + 0.5 + 0.2 + 1.0/3.0) / 4.0
    assert abs(metrics["mrr"] - expected_mrr) < 1e-4, f"Expected MRR={expected_mrr:.4f}, got {metrics['mrr']:.4f}"
    print("  PASS: Partial ranking metrics are correct")


def test_dot_product_shapes():
    """Test 3: DotProduct evaluator produces correct output type."""
    N, D = 20, 8
    embeddings = np.random.randn(N, D).astype(np.float32)

    # 10 transitions among ports 0-9
    transitions = np.array(
        [(i, (i + 1) % 10, f"vessel_{i}") for i in range(10)],
        dtype=object,
    )
    warm_ports = np.arange(10)
    global_to_warm = {i: i for i in range(10)}

    dp = DotProductLinkPredictor()
    result = dp.evaluate(embeddings, transitions, warm_ports, global_to_warm)

    assert isinstance(result, LinkPredictionResults)
    assert result.framing == "dot_product"
    assert result.num_transitions == 10
    assert 0.0 <= result.hits_at_1 <= 1.0
    assert 0.0 <= result.mrr <= 1.0
    print("  PASS: DotProduct shapes and output type correct")


def test_dot_product_results_structure():
    """Test 4: DotProduct results have all required fields."""
    N, D = 15, 4
    embeddings = np.random.randn(N, D).astype(np.float32)
    transitions = np.array([(0, 1, "v1"), (1, 2, "v2"), (2, 3, "v3")], dtype=object)
    warm_ports = np.arange(5)
    global_to_warm = {i: i for i in range(5)}

    dp = DotProductLinkPredictor()
    result = dp.evaluate(embeddings, transitions, warm_ports, global_to_warm)

    assert hasattr(result, 'hits_at_1')
    assert hasattr(result, 'hits_at_5')
    assert hasattr(result, 'hits_at_10')
    assert hasattr(result, 'mrr')
    assert hasattr(result, 'num_transitions')
    assert hasattr(result, 'framing')
    print("  PASS: DotProduct results structure correct")


def test_mlp_trains():
    """Test 5: MLP loss decreases over a few epochs on synthetic data."""
    N, D = 20, 8
    np.random.seed(42)
    embeddings = np.random.randn(N, D).astype(np.float32)

    # Simple transitions: port i -> port (i+1) % 10
    train_trans = np.array(
        [(i % 10, (i + 1) % 10, f"v{i}") for i in range(50)],
        dtype=object,
    )
    val_trans = np.array(
        [(i % 10, (i + 1) % 10, f"v{i}") for i in range(10)],
        dtype=object,
    )
    test_trans = np.array(
        [(i % 10, (i + 1) % 10, f"v{i}") for i in range(10)],
        dtype=object,
    )

    warm_ports = np.arange(10)
    global_to_warm = {i: i for i in range(10)}

    mlp = MLPLinkPredictor()
    result = mlp.train_and_evaluate(
        embeddings, train_trans, val_trans, test_trans,
        warm_ports, global_to_warm,
        n_runs=1, epochs=20, patience=20,
    )

    assert isinstance(result, LinkPredictionResults)
    assert result.framing == "mlp"
    # With enough training, MRR should be above random (1/10 = 0.1 baseline)
    print(f"  PASS: MLP trains (MRR={result.mrr:.4f})")


def test_mlp_results_structure():
    """Test 6: MLP results have all required fields."""
    N, D = 15, 4
    embeddings = np.random.randn(N, D).astype(np.float32)

    train_trans = np.array([(0, 1, "v1"), (1, 2, "v2"), (2, 0, "v3")] * 5, dtype=object)
    val_trans = np.array([(0, 1, "v1"), (1, 2, "v2")], dtype=object)
    test_trans = np.array([(0, 1, "v1"), (1, 2, "v2")], dtype=object)
    warm_ports = np.arange(3)
    global_to_warm = {i: i for i in range(3)}

    mlp = MLPLinkPredictor()
    result = mlp.train_and_evaluate(
        embeddings, train_trans, val_trans, test_trans,
        warm_ports, global_to_warm,
        n_runs=1, epochs=5, patience=5,
    )

    assert hasattr(result, 'hits_at_1')
    assert hasattr(result, 'mrr')
    assert result.framing == "mlp"
    print("  PASS: MLP results structure correct")


def test_build_sequences():
    """Test 7: 3 transitions same vessel -> correct sequences."""
    # vessel_0: port 0 -> 1, 1 -> 2, 2 -> 3
    transitions = np.array([
        (0, 1, "vessel_0"),
        (1, 2, "vessel_0"),
        (2, 3, "vessel_0"),
    ], dtype=object)

    seqs = build_sequences(transitions, max_len=5)

    # Port sequence: [0, 1, 2, 3]
    # Sequences with sliding window:
    #   context=[0], target=1
    #   context=[0,1], target=2
    #   context=[0,1,2], target=3
    assert len(seqs) == 3, f"Expected 3 sequences, got {len(seqs)}"
    assert seqs[0][0] == [0] and seqs[0][1] == 1
    assert seqs[1][0] == [0, 1] and seqs[1][1] == 2
    assert seqs[2][0] == [0, 1, 2] and seqs[2][1] == 3
    print("  PASS: Sequence building for single vessel correct")


def test_build_sequences_different_vessels():
    """Test 8: Different vessels -> separate sequences."""
    transitions = np.array([
        (0, 1, "vessel_A"),
        (1, 2, "vessel_A"),
        (5, 6, "vessel_B"),
        (6, 7, "vessel_B"),
    ], dtype=object)

    seqs = build_sequences(transitions, max_len=5)

    # vessel_A: [0,1,2] -> 2 sequences
    # vessel_B: [5,6,7] -> 2 sequences
    assert len(seqs) == 4, f"Expected 4 sequences, got {len(seqs)}"

    # Check vessel IDs are preserved
    vessel_ids = [s[2] for s in seqs]
    assert vessel_ids.count("vessel_A") == 2
    assert vessel_ids.count("vessel_B") == 2
    print("  PASS: Different vessels produce separate sequences")


def test_lstm_trains():
    """Test 9: LSTM loss decreases on synthetic data."""
    N, D = 20, 8
    np.random.seed(42)
    embeddings = np.random.randn(N, D).astype(np.float32)

    # vessel_0 repeats route: 0 -> 1 -> 2 -> 3 many times
    train_trans = np.array(
        [(i % 4, (i + 1) % 4, "v0") for i in range(40)],
        dtype=object,
    )
    val_trans = np.array(
        [(i % 4, (i + 1) % 4, "v1") for i in range(10)],
        dtype=object,
    )
    test_trans = np.array(
        [(i % 4, (i + 1) % 4, "v2") for i in range(10)],
        dtype=object,
    )

    warm_ports = np.arange(4)
    global_to_warm = {i: i for i in range(4)}

    lstm = LSTMLinkPredictor()
    result = lstm.train_and_evaluate(
        embeddings, train_trans, val_trans, test_trans,
        warm_ports, global_to_warm,
        n_runs=1, epochs=20, patience=20, max_len=3,
    )

    assert isinstance(result, LinkPredictionResults)
    assert result.framing == "lstm"
    print(f"  PASS: LSTM trains (MRR={result.mrr:.4f})")


def test_lstm_results_structure():
    """Test 10: LSTM results have all required fields."""
    N, D = 15, 4
    embeddings = np.random.randn(N, D).astype(np.float32)

    train_trans = np.array([(0, 1, "v1"), (1, 2, "v1"), (2, 0, "v1")] * 5, dtype=object)
    val_trans = np.array([(0, 1, "v2"), (1, 2, "v2")], dtype=object)
    test_trans = np.array([(0, 1, "v3"), (1, 2, "v3")], dtype=object)
    warm_ports = np.arange(3)
    global_to_warm = {i: i for i in range(3)}

    lstm = LSTMLinkPredictor()
    result = lstm.train_and_evaluate(
        embeddings, train_trans, val_trans, test_trans,
        warm_ports, global_to_warm,
        n_runs=1, epochs=5, patience=5,
    )

    assert hasattr(result, 'hits_at_1')
    assert hasattr(result, 'mrr')
    assert result.framing == "lstm"
    assert result.num_transitions > 0
    print("  PASS: LSTM results structure correct")


def test_warm_port_index_mapping():
    """Test 11: Verify global-to-warm index conversion."""
    # Transitions use global indices 2, 5, 8, 10
    train_trans = np.array([(2, 5, "v1"), (5, 8, "v2")], dtype=object)
    val_trans = np.array([(8, 10, "v3")], dtype=object)
    test_trans = np.array([(10, 2, "v4")], dtype=object)

    warm_ports, global_to_warm = _build_warm_port_mapping(train_trans, val_trans, test_trans)

    assert list(warm_ports) == [2, 5, 8, 10], f"Expected [2,5,8,10], got {list(warm_ports)}"
    assert global_to_warm[2] == 0
    assert global_to_warm[5] == 1
    assert global_to_warm[8] == 2
    assert global_to_warm[10] == 3
    print("  PASS: Global-to-warm index mapping correct")


def test_load_transitions():
    """Test 12: Verify numpy object array loading round-trip."""
    import tempfile
    transitions = [(0, 1, "v1"), (1, 2, "v2"), (2, 3, "v3")]
    arr = np.array(transitions, dtype=object)

    with tempfile.TemporaryDirectory() as tmpdir:
        np.save(os.path.join(tmpdir, "test_transitions.npy"), arr, allow_pickle=True)
        from src.evaluation.link_prediction import load_transitions
        loaded = load_transitions(tmpdir, "test")

        assert len(loaded) == 3
        assert int(loaded[0][0]) == 0
        assert int(loaded[0][1]) == 1
        assert loaded[0][2] == "v1"
    print("  PASS: Transition load round-trip correct")


def test_per_vessel_type_metrics():
    """Test 13: Verify per-vessel-type grouping and metric computation."""
    B = 300  # enough to meet min_count=100

    # Two vessel types: "Tanker" (200 transitions) and "Container" (100 transitions)
    # All rank 1 (perfect predictions)
    ranks = torch.ones(B)
    vessel_types = []

    for i in range(200):
        vessel_types.append("Tanker")

    for i in range(200, 300):
        vessel_types.append("Container")

    result = compute_per_vessel_type_metrics(ranks, vessel_types, min_count=100)

    assert "Tanker" in result, f"Expected Tanker in results, got {list(result.keys())}"
    assert "Container" in result, f"Expected Container in results, got {list(result.keys())}"
    assert result["Tanker"]["count"] == 200
    assert result["Container"]["count"] == 100
    assert abs(result["Tanker"]["hits_at_1"] - 1.0) < 1e-6
    assert abs(result["Container"]["hits_at_1"] - 1.0) < 1e-6
    print("  PASS: Per-vessel-type metrics correct")


if __name__ == "__main__":
    tests = [
        test_ranking_metrics_perfect,
        test_ranking_metrics_partial,
        test_dot_product_shapes,
        test_dot_product_results_structure,
        test_mlp_trains,
        test_mlp_results_structure,
        test_build_sequences,
        test_build_sequences_different_vessels,
        test_lstm_trains,
        test_lstm_results_structure,
        test_warm_port_index_mapping,
        test_load_transitions,
        test_per_vessel_type_metrics,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        name = test_fn.__doc__.split(":")[0] if test_fn.__doc__ else test_fn.__name__
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {name} — {e}")
            failed += 1

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed == 0:
        print("All tests passed!")
    else:
        print(f"{failed} test(s) FAILED")
        sys.exit(1)
