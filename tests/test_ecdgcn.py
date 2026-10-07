"""Verification tests for EdgeConditionedDGCN integration."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import numpy as np


def test_layer_output_shape():
    """Test 1: EdgeConditionedDGCN produces the correct output shape (1, N, out_ft)."""
    from src.layers.ecdgcn import EdgeConditionedDGCN

    N, in_ft, out_ft, edge_ft = 5, 10, 16, 4
    layer = EdgeConditionedDGCN(in_ft, out_ft, 'prelu', edge_ft)

    # Triangle graph: 0->1, 1->2, 2->0
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_features = torch.randn(3, edge_ft)
    seq = torch.randn(1, N, in_ft)

    out = layer(seq, edge_index, edge_features)
    assert out.shape == (1, N, out_ft), f"Expected (1, {N}, {out_ft}), got {out.shape}"
    print("  PASS: Output shape is correct")


def test_gradient_flow():
    """Test 2: Gradients flow through both node features and edge features."""
    from src.layers.ecdgcn import EdgeConditionedDGCN

    N, in_ft, out_ft, edge_ft = 5, 8, 16, 4
    layer = EdgeConditionedDGCN(in_ft, out_ft, 'prelu', edge_ft)

    # Use a graph where some nodes have in-degree > 1 so weight normalization
    # doesn't collapse to raw_w/raw_w=1 (which zeroes edge_features gradients)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 0, 4, 2]], dtype=torch.long)
    edge_features = torch.randn(5, edge_ft, requires_grad=True)
    seq = torch.randn(1, N, in_ft, requires_grad=True)

    out = layer(seq, edge_index, edge_features)
    loss = out.sum()
    loss.backward()

    assert seq.grad is not None and seq.grad.abs().sum() > 0, "No gradient on seq"
    assert edge_features.grad is not None and edge_features.grad.abs().sum() > 0, "No gradient on edge_features"
    print("  PASS: Gradients flow through seq and edge_features")


def test_dgi_undirected_backward_compat():
    """Test 3: DGI with directed=False, edge_dim=None still works."""
    from src.models.dgi import DGI

    n_in, n_h = 10, 16
    model = DGI(n_in, n_h, 'prelu', directed=False)

    assert not model.use_edge_features
    assert isinstance(model.encoder, nn.Module)

    # Forward pass (undirected, sparse=False)
    N = 5
    adj = torch.eye(N).unsqueeze(0)
    seq1 = torch.randn(1, N, n_in)
    seq2 = torch.randn(1, N, n_in)
    logits = model(seq1, seq2, adj, False, None, None, None)
    assert logits.shape == (1, 2 * N), f"Expected (1, {2*N}), got {logits.shape}"
    print("  PASS: Undirected DGI backward-compatible")


def test_dgi_directed_no_edge_backward_compat():
    """Test 4: DGI with directed=True, edge_dim=None uses DirectedGCN."""
    from src.models.dgi import DGI
    from src.layers.dgcn import DirectedGCN

    n_in, n_h = 10, 16
    model = DGI(n_in, n_h, 'prelu', directed=True)

    assert not model.use_edge_features
    assert isinstance(model.encoder, DirectedGCN)

    # Forward pass (directed, sparse=False)
    N = 5
    adj_in = torch.eye(N).unsqueeze(0)
    adj_out = torch.eye(N).unsqueeze(0)
    seq1 = torch.randn(1, N, n_in)
    seq2 = torch.randn(1, N, n_in)
    logits = model(seq1, seq2, adj_in, False, None, None, None, adj_out)
    assert logits.shape == (1, 2 * N), f"Expected (1, {2*N}), got {logits.shape}"
    print("  PASS: Directed DGI (no edge features) backward-compatible")


def test_dgi_with_edge_features():
    """Test 5: DGI with directed=True and edge_dim selects EdgeConditionedDGCN."""
    from src.models.dgi import DGI
    from src.layers.ecdgcn import EdgeConditionedDGCN

    n_in, n_h, edge_ft = 10, 16, 4
    model = DGI(n_in, n_h, 'prelu', directed=True, edge_dim=edge_ft)

    assert model.use_edge_features
    assert isinstance(model.encoder, EdgeConditionedDGCN)

    N = 5
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 4]], dtype=torch.long)
    edge_features = torch.randn(4, edge_ft)
    seq1 = torch.randn(1, N, n_in)
    seq2 = torch.randn(1, N, n_in)

    logits = model(seq1, seq2, None, False, None, None, None,
                   edge_index=edge_index, edge_features=edge_features)
    assert logits.shape == (1, 2 * N), f"Expected (1, {2*N}), got {logits.shape}"

    # Test embed
    embeds, summary = model.embed(seq1, None, False, None,
                                  edge_index=edge_index, edge_features=edge_features)
    assert embeds.shape == (1, N, n_h), f"Expected (1, {N}, {n_h}), got {embeds.shape}"
    print("  PASS: DGI with edge features works correctly")


def test_edge_features_not_corrupted():
    """Test 6: Edge features tensor is not modified during forward pass."""
    from src.models.dgi import DGI

    n_in, n_h, edge_ft = 10, 16, 4
    model = DGI(n_in, n_h, 'prelu', directed=True, edge_dim=edge_ft)

    N = 5
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_features = torch.randn(3, edge_ft)
    ef_original = edge_features.clone()
    seq1 = torch.randn(1, N, n_in)
    seq2 = torch.randn(1, N, n_in)

    model(seq1, seq2, None, False, None, None, None,
          edge_index=edge_index, edge_features=edge_features)

    assert torch.equal(edge_features, ef_original), "Edge features were modified during forward pass!"
    print("  PASS: Edge features not corrupted during forward")


def test_dgi_training_loss_decreases():
    """Test 7: DGI with edge features can train and loss decreases."""
    from src.models.dgi import DGI

    torch.manual_seed(42)
    N, n_in, n_h, edge_ft = 20, 10, 32, 4
    model = DGI(n_in, n_h, 'prelu', directed=True, edge_dim=edge_ft)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    # Create a small random graph
    num_edges = 40
    edge_index = torch.stack([
        torch.randint(0, N, (num_edges,)),
        torch.randint(0, N, (num_edges,)),
    ])
    edge_features = torch.randn(num_edges, edge_ft)
    features = torch.randn(1, N, n_in)

    losses = []
    for _ in range(30):
        model.train()
        optimizer.zero_grad()

        idx = np.random.permutation(N)
        shuf_fts = features[:, idx, :]

        lbl = torch.cat([torch.ones(1, N), torch.zeros(1, N)], 1)
        logits = model(features, shuf_fts, None, False, None, None, None,
                       edge_index=edge_index, edge_features=edge_features)
        loss = loss_fn(logits, lbl)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease from start to end (compare mean of first 5 vs. last 5)
    early_loss = np.mean(losses[:5])
    late_loss = np.mean(losses[-5:])
    assert late_loss < early_loss, f"Loss did not decrease: {early_loss:.4f} -> {late_loss:.4f}"
    print(f"  PASS: Training loss decreased ({early_loss:.4f} -> {late_loss:.4f})")


def test_config_use_edge_features_default():
    """Test 8: ModelConfig.use_edge_features defaults to False."""
    from src.config.config import ModelConfig

    cfg = ModelConfig()
    assert cfg.use_edge_features is False
    print("  PASS: ModelConfig.use_edge_features defaults to False")


def test_graph_dataset_edge_attributes():
    """Test 9: GraphDataset has edge_index, edge_features, edge_dim."""
    from src.data.base import GraphDataset

    # Can't instantiate ABC directly, check that the attributes exist
    # by checking the __init__ signature effect through a concrete subclass
    class DummyDataset(GraphDataset):
        def load(self):
            self._loaded = True
            return self
        @property
        def num_classes(self):
            return 2
        @property
        def num_features(self):
            return 10

    ds = DummyDataset('test', '.')
    assert ds.edge_index is None
    assert ds.edge_features is None
    assert ds.edge_dim is None

    # Set edge features and check the edge_dim property
    ds.edge_features = np.random.randn(10, 4)
    assert ds.edge_dim == 4
    print("  PASS: GraphDataset edge attributes work correctly")


if __name__ == '__main__':
    tests = [
        ("1. Layer output shape", test_layer_output_shape),
        ("2. Gradient flow", test_gradient_flow),
        ("3. Undirected DGI backward compat", test_dgi_undirected_backward_compat),
        ("4. Directed DGI (no edge feat) backward compat", test_dgi_directed_no_edge_backward_compat),
        ("5. DGI with edge features", test_dgi_with_edge_features),
        ("6. Edge features not corrupted", test_edge_features_not_corrupted),
        ("7. Training loss decreases", test_dgi_training_loss_decreases),
        ("8. Config default", test_config_use_edge_features_default),
        ("9. GraphDataset edge attributes", test_graph_dataset_edge_attributes),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        print(f"\nTest {name}:")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
    print("All tests passed!")
