"""Train DGI on the maritime graph and save embeddings + visualizations.

Usage:
    python experiments/maritime/run_dgi_training.py --config configs/experiments/maritime.yaml
    python experiments/maritime/run_dgi_training.py --config configs/experiments/maritime.yaml --seed 42 --epochs 10
    python experiments/maritime/run_dgi_training.py --config configs/experiments/maritime.yaml --undirected
    python experiments/maritime/run_dgi_training.py --config configs/experiments/maritime.yaml --no-edge-features

Outputs (in results/maritime/embeddings/{encoder}_seed{seed}/):
    embeddings.npy          - Node embeddings (N, D)
    loss_curve.npy          - Per-epoch loss values
    training_results.json   - Config, timing, data stats, embedding shape
    *_loss_curve.pdf        - Training loss curve
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import add_common_args, setup_experiment, get_data_suffix

from src.training import DGITrainer
from src.utils import visualizations


def main():
    parser = argparse.ArgumentParser(description="Maritime DGI training")
    add_common_args(parser)
    args = parser.parse_args()

    print("=" * 60)
    print("Maritime DGI Training")
    print("=" * 60)

    config, dataset, device, encoder_name, output_dir = setup_experiment(args)

    # Set the checkpoint prefix so ablation runs don't overwrite each other
    data_suffix = get_data_suffix(config)
    config.training.checkpoint_prefix = f"{encoder_name}{data_suffix}"

    # Paths
    embeddings_path = output_dir / "embeddings.npy"
    loss_curve_path = output_dir / "loss_curve.npy"
    results_path = output_dir / "training_results.json"

    # Ensure checkpoint dir exists
    checkpoint_dir = Path(config.output_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Train DGI
    print(f"\nTraining DGI ({encoder_name})...")
    trainer = DGITrainer(config, device)
    t0 = time.time()
    model, embeddings_tensor = trainer.fit(dataset)
    elapsed = time.time() - t0

    embeddings_np = embeddings_tensor.detach().cpu().numpy()[0]  # (1, N, D) -> (N, D)
    loss_history = trainer.loss_history
    best_epoch = int(np.argmin(loss_history))

    print(f"Training complete: {elapsed:.1f}s, best epoch {best_epoch}")
    print(f"Embedding shape: {embeddings_np.shape}")

    # Save embeddings and loss curve
    np.save(embeddings_path, embeddings_np.astype(np.float32))
    np.save(loss_curve_path, np.array(loss_history, dtype=np.float32))

    # Loss curve
    if loss_history:
        visualizations.plot_loss_curve(
            loss_history, 'Training Loss', 'BCE with logits', encoder_name, output_dir=str(output_dir)
        )

    # t-SNE colored by sub-region (uses integer labels, always available)
    _meta_path = Path(config.data.data_dir) / "processed" / "metadata.json"
    _label_info = {}
    if _meta_path.exists():
        with open(_meta_path) as _f:
            _label_info = json.load(_f).get("label_info", {})
    sub_region_names = _label_info.get("sub_regions", [f"Class_{i}" for i in range(dataset.num_classes)])
    label_dict = {i: name for i, name in enumerate(sub_region_names)}

    print("  Generating sub-region t-SNE...")
    tsne_df = visualizations.compute_tsne_embeddings(embeddings_np, dataset.labels, label_dict)
    visualizations.plot_learned_tsne_embeddings(
        tsne_df, f"maritime_{encoder_name}{data_suffix}", output_dir=str(output_dir)
    )

    # Save training results JSON
    results_dict = {
        "config": {
            "encoder": encoder_name,
            "seed": config.seed,
            "hidden_units": config.model.hidden_units,
            "directed": config.model.directed,
            "use_edge_features": config.model.use_edge_features,
            "epochs_configured": config.training.epochs,
            "learning_rate": config.training.learning_rate,
            "patience": config.training.patience,
        },
        "timing": {
            "training_seconds": round(elapsed, 2),
            "best_epoch": best_epoch,
            "total_epochs_run": len(loss_history),
        },
        "data": {
            "num_nodes": dataset.num_nodes,
            "num_features": dataset.num_features,
            "num_edges": dataset.edge_index.shape[1] if dataset.edge_index is not None else 0,
        },
        "embedding_shape": list(embeddings_np.shape),
    }

    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
