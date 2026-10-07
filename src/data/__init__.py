from .base import GraphDataset
from .planetoid import PlanetoidDataset
from .maritime import MaritimeDataset
from .label_maps import (
    label_dict_cora,
    label_dict_citeseer,
    label_dict_pubmed,
    label_dicts,
)

__all__ = [
    'GraphDataset',
    'PlanetoidDataset',
    'MaritimeDataset',
    'label_dict_cora',
    'label_dict_citeseer',
    'label_dict_pubmed',
    'label_dicts',
]
