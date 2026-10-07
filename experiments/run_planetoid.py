"""Planetoid benchmark runner: DGI training and evaluation on Cora/Citeseer/Pubmed.

Reproduces the DGI replication study (training, 50-run linear probe,
t-SNE visualizations, confusion matrix).

Usage:
    python experiments/run_planetoid.py --config configs/experiments/cora.yaml
    python experiments/run_planetoid.py --config configs/experiments/cora.yaml --epochs 500
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn import metrics

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data import PlanetoidDataset, label_dicts
from src.training import DGITrainer
from src.evaluation import LinearProbeEvaluator
from src.utils import visualizations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Deep Graph Infomax training and evaluation on Planetoid benchmarks"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset name from config"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs from config"
    )
    parser.add_argument(
        "--directed",
        action="store_true",
        default=None,
        help="Override directed mode from config"
    )
    return parser.parse_args()


def get_dataset(config):
    """Load appropriate dataset based on config."""
    dataset_name = config.data.dataset

    if dataset_name in {'cora', 'citeseer', 'pubmed'}:
        return PlanetoidDataset(dataset_name, config.data.data_dir)
    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. This runner supports the Planetoid "
            "benchmarks (cora, citeseer, pubmed); maritime experiments use the "
            "entry points in experiments/maritime/."
        )


def main():
    args = parse_args()

    # Load configuration
    cli_overrides = {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'directed': args.directed,
    }
    config = load_config(args.config, cli_overrides)

    print(f"Running DGI experiment:")
    print(f"  Dataset: {config.data.dataset}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Directed: {config.model.directed}")
    print(f"  Hidden units: {config.model.hidden_units}")
    print(f"  Output dir: {config.output_dir}")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Create output directories
    output_dir = Path(config.output_dir)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = get_dataset(config)
    dataset.load()
    print(f"  Loaded {dataset.name}: {dataset.num_nodes} nodes, {dataset.num_features} features, {dataset.num_classes} classes")

    # Visualize raw features
    from src.preprocessing import preprocess_features
    features_dense, _ = preprocess_features(dataset.features)

    label_dict = label_dicts.get(dataset.name, {})
    if label_dict:
        embeddings_raw = visualizations.compute_tsne_embeddings(
            np.array(features_dense), dataset.labels, label_dict
        )
        visualizations.plot_raw_tsne_embeddings(embeddings_raw, dataset.name)

    # Train DGI
    trainer = DGITrainer(config, device)
    model, embeddings = trainer.fit(dataset)

    # Plot loss curve
    visualizations.plot_loss_curve(
        trainer.loss_history,
        'Training Loss',
        'BCE with logits',
        dataset.name
    )

    # Prepare labels as tensors
    labels = torch.FloatTensor(dataset.labels[np.newaxis]).to(device)
    idx_train = torch.LongTensor(dataset.idx_train).to(device)
    idx_test = torch.LongTensor(dataset.idx_test).to(device)

    # Evaluate with the linear probe
    evaluator = LinearProbeEvaluator(
        embedding_dim=config.model.hidden_units,
        num_classes=dataset.num_classes,
        n_runs=50,
        device=device,
    )
    results = evaluator.evaluate(embeddings, labels, idx_train, idx_test)

    print(f"\nResults:")
    print(f"  Average accuracy: {results.mean_accuracy:.2f}%")
    print(f"  Standard deviation: {results.std_accuracy:.2f}%")

    # Visualize learned embeddings
    embeddings_np = embeddings.detach().cpu().numpy()[0]
    labels_np = dataset.labels

    if label_dict:
        embeddings_learned = visualizations.compute_tsne_embeddings(
            embeddings_np, labels_np, label_dict
        )

        # Silhouette score
        silhouette = metrics.silhouette_score(
            np.array(embeddings_learned[['tsne_1', 'tsne_2']]),
            embeddings_learned['label']
        )
        print(f"  Silhouette coefficient: {silhouette:.3f}")

        visualizations.plot_learned_tsne_embeddings(embeddings_learned, dataset.name)

        # Classification report
        label_names = list(label_dict.values())
        print(f"\nClassification report for {dataset.name}:")
        visualizations.print_classification_report(
            results.labels, results.predictions, label_names
        )

        # Confusion matrix
        visualizations.plot_confusion_matrix(
            results.labels,
            results.predictions,
            label_names,
            trainer.loss_history,
            dataset.name,
            normalize=False
        )

        # TSNE comparison
        visualizations.plot_tsne_comparison(
            embeddings_raw, embeddings_learned, dataset.name
        )

    print("\nExperiment complete!")
    print(f"  Checkpoints: {output_dir / 'checkpoints'}")
    print(f"  Plots: {output_dir / 'plots'}")


if __name__ == '__main__':
    main()
