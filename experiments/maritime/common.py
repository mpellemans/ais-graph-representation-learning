"""Shared utilities for maritime experiment scripts.

Provides the common setup (seed, device, config loading, dataset loading, path
construction) so that each experiment script reduces to ~3 lines of boilerplate.
"""
import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
import torch

# Ensure project root is importable
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.data import MaritimeDataset


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_encoder_name(config):
    """Derive encoder name from config flags.

    Multi-layer DirectedGCN runs (num_layers > 1) get a '_{L}layer' suffix so
    they write to distinct output dirs/checkpoints and never collide with the
    canonical single-layer 'dgcn' run.
    """
    if not config.model.directed:
        base = "gcn"
    elif not config.model.use_edge_features:
        base = "dgcn"
    else:
        base = "ecdgcn"

    num_layers = getattr(config.model, 'num_layers', 1)
    if num_layers > 1:
        base = f"{base}_{num_layers}layer"
    return base


def get_data_suffix(config):
    """Return a directory suffix encoding active data preprocessing flags.

    Ensures ablation runs never collide with the base run output directory.
    Examples: '' (base), '_no_geo', '_k10', '_k20', '_k10_no_geo'
    """
    suffix = ''
    if getattr(config.data, 'sparsify_top_k', 0) > 0:
        suffix += f'_k{config.data.sparsify_top_k}'
    if getattr(config.data, 'exclude_geo_features', False):
        suffix += '_no_geo'
    if getattr(config.data, 'exclude_connectivity_features', False):
        suffix += '_no_conn'
    return suffix


def get_device():
    """Return cuda device if available, else cpu."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_output_dir(config, encoder_name, seed):
    """Build output directory: results/maritime/embeddings/{encoder}{data_suffix}_seed{seed}/."""
    data_suffix = get_data_suffix(config)
    base_dir = Path(config.output_dir) / "embeddings" / f"{encoder_name}{data_suffix}_seed{seed}"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def load_maritime_dataset(config):
    """Load and return the MaritimeDataset."""
    dataset = MaritimeDataset(
        config.data.data_dir,
        sparsify_top_k=getattr(config.data, 'sparsify_top_k', 0),
        exclude_geo_features=getattr(config.data, 'exclude_geo_features', False),
        exclude_connectivity_features=getattr(config.data, 'exclude_connectivity_features', False),
    )
    dataset.load()
    return dataset


def load_embeddings(path):
    """Load embeddings from a .npy file."""
    return np.load(path)


def load_label_info(data_dir):
    """Load label metadata from the cached metadata.json."""
    metadata_path = Path(data_dir) / "processed" / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata.get("label_info", {})
    return {}


def add_common_args(parser):
    """Add common CLI arguments shared across all maritime scripts."""
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to maritime YAML config file"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed (default: from config)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--no-edge-features", action="store_true",
        help="Disable edge features (DirectedGCN ablation)"
    )
    parser.add_argument(
        "--undirected", action="store_true",
        help="Disable directed mode (GCN ablation)"
    )
    return parser


def setup_experiment(args):
    """Common experiment setup from parsed CLI args.

    Loads config, applies overrides, sets seed, gets device, builds paths,
    and loads the dataset.

    Args:
        args: Parsed argparse namespace with config, seed, epochs,
              no_edge_features, undirected fields.

    Returns:
        Tuple of (config, dataset, device, encoder_name, output_dir)
    """
    config = load_config(args.config)

    # Apply CLI overrides
    if args.seed is not None:
        config.seed = args.seed
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.no_edge_features:
        config.model.use_edge_features = False
    if args.undirected:
        config.model.directed = False
        config.model.use_edge_features = False

    seed = config.seed
    encoder_name = get_encoder_name(config)
    device = get_device()

    set_seed(seed)

    output_dir = build_output_dir(config, encoder_name, seed)
    dataset = load_maritime_dataset(config)

    print(f"  Encoder:    {encoder_name}")
    print(f"  Seed:       {seed}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  Device:     {device} ({gpu_name})")
    else:
        print(f"  Device:     {device} (no CUDA)")
    print(f"  Nodes:      {dataset.num_nodes}")
    print(f"  Features:   {dataset.num_features}")
    print(f"  Classes:    {dataset.num_classes}")
    print(f"  Output:     {output_dir}")

    return config, dataset, device, encoder_name, output_dir
