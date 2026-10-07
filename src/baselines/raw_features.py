"""Raw features baseline: use node features directly as embeddings."""
import numpy as np


class RawFeaturesBaseline:
    """Return raw node features as embeddings without any training."""

    def get_embeddings(self, dataset, device=None) -> np.ndarray:
        """Return dense node feature matrix as float32.

        Args:
            dataset: MaritimeDataset with .features (sp.lil_matrix)
            device: ignored (no computation)

        Returns:
            (N, F) float32 numpy array
        """
        return dataset.features.toarray().astype(np.float32)
