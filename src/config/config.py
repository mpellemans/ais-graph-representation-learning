"""Configuration management using dataclasses and YAML."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    hidden_units: int = 512
    activation: str = 'prelu'
    directed: bool = False
    use_edge_features: bool = False
    num_layers: int = 1  # Number of stacked encoder layers (DirectedGCN); 1 = original single-layer encoder


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 300
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    patience: int = 20
    batch_size: int = 1
    grad_clip_norm: float = 0.0  # Max gradient norm (0.0 = disabled)
    edge_grad_clip_norm: float = 0.0  # Per-group clip for edge_mlp (ECDGCN, 0.0 = disabled)
    checkpoint_prefix: str = ""  # Override checkpoint filename stem (default: dataset name)


@dataclass
class DataConfig:
    """Data loading configuration."""
    dataset: str = 'cora'
    sparse: bool = True
    data_dir: str = 'data/planetoid'
    skip_feature_normalization: bool = False  # Skip L1 row-norm (for pre-normalized features)
    sparsify_top_k: int = 0  # Keep top-K outgoing edges per node (0 = disabled)
    exclude_geo_features: bool = False  # Exclude geohash+country+continent+sub_region one-hots
    exclude_connectivity_features: bool = False  # Exclude degree/connectivity/activity-count node features


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: str = 'results'
    seed: int = 42

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'ExperimentConfig':
        """Create config from the dictionary."""
        model_cfg = ModelConfig(**config_dict.get('model', {}))
        training_cfg = TrainingConfig(**config_dict.get('training', {}))
        data_cfg = DataConfig(**config_dict.get('data', {}))

        return cls(
            model=model_cfg,
            training=training_cfg,
            data=data_cfg,
            output_dir=config_dict.get('output_dir', 'results'),
            seed=config_dict.get('seed', 42),
        )


def load_config(config_path: str, cli_overrides: Optional[dict] = None) -> ExperimentConfig:
    """
    Load configuration from the YAML file with optional CLI overrides.

    Args:
        config_path: Path to YAML config file
        cli_overrides: Dictionary of CLI argument overrides

    Returns:
        ExperimentConfig instance
    """
    config_path = Path(config_path)

    # Load base config if specified in the experiment config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Handle _base_ inheritance
    if '_base_' in config_dict:
        base_path = config_path.parent / config_dict.pop('_base_')
        with open(base_path, 'r') as f:
            base_dict = yaml.safe_load(f)
        # Merge: experiment overrides base
        config_dict = _deep_merge(base_dict, config_dict)

    # Apply CLI overrides (None means "not provided"; falsy values like 0 or
    # False are valid overrides and must not be dropped)
    if cli_overrides:
        if cli_overrides.get('dataset') is not None:
            config_dict.setdefault('data', {})['dataset'] = cli_overrides['dataset']
        if cli_overrides.get('epochs') is not None:
            config_dict.setdefault('training', {})['epochs'] = cli_overrides['epochs']
        if cli_overrides.get('directed') is not None:
            config_dict.setdefault('model', {})['directed'] = cli_overrides['directed']

    return ExperimentConfig.from_dict(config_dict)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
