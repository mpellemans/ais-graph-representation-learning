"""Linear probe evaluation for learned embeddings."""
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from src.models import LogReg


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    accuracies: List[float]
    mean_accuracy: float
    std_accuracy: float
    predictions: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None


class LinearProbeEvaluator:
    """Linear probe evaluation for graph embeddings.

    Trains a logistic regression classifier on frozen embeddings
    to evaluate the quality of learned representations.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        n_runs: int = 50,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        epochs: int = 100,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            embedding_dim: Dimension of input embeddings
            num_classes: Number of output classes
            n_runs: Number of evaluation runs (for mean/std)
            lr: Learning rate for logistic regression
            weight_decay: L2 regularization weight
            epochs: Training epochs per run
            device: Device to run on
        """
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.n_runs = n_runs
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def evaluate(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        idx_train: torch.Tensor,
        idx_test: torch.Tensor,
    ) -> EvaluationResults:
        """Evaluate embeddings using the linear probe.

        Args:
            embeddings: Node embeddings [1, num_nodes, embedding_dim]
            labels: One-hot labels [1, num_nodes, num_classes]
            idx_train: Training node indices
            idx_test: Test node indices

        Returns:
            EvaluationResults with accuracies and predictions
        """
        # Prepare data
        train_embs = embeddings[0, idx_train]
        test_embs = embeddings[0, idx_test]
        train_lbls = torch.argmax(labels[0, idx_train], dim=1)
        test_lbls = torch.argmax(labels[0, idx_test], dim=1)

        accuracies = []
        loss_fn = nn.CrossEntropyLoss()
        final_preds = None

        for run in range(self.n_runs):
            # Initialize fresh classifier
            classifier = LogReg(self.embedding_dim, self.num_classes).to(self.device)
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )

            # Train classifier
            for _ in range(self.epochs):
                classifier.train()
                optimizer.zero_grad()

                logits = classifier(train_embs)
                loss = loss_fn(logits, train_lbls)

                loss.backward()
                optimizer.step()

            # Evaluate
            classifier.eval()
            with torch.no_grad():
                logits = classifier(test_embs)
                preds = torch.argmax(logits, dim=1)
                acc = (preds == test_lbls).float().mean().item() * 100
                accuracies.append(acc)

                # Save final predictions
                if run == self.n_runs - 1:
                    final_preds = preds.cpu().numpy()

            print(f'Run {run + 1}/{self.n_runs}: {acc:.2f}%')

        accs_tensor = torch.tensor(accuracies)

        return EvaluationResults(
            accuracies=accuracies,
            mean_accuracy=accs_tensor.mean().item(),
            std_accuracy=accs_tensor.std().item(),
            predictions=final_preds,
            labels=test_lbls.cpu().numpy(),
        )
