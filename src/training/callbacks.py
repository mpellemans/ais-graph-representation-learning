"""Training callbacks for early stopping and checkpointing."""
from pathlib import Path
from typing import Optional
import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping callback with model checkpointing.

    Monitors a metric (typically loss) and stops training when no improvement
    is seen for a specified number of epochs.
    """

    def __init__(
        self,
        patience: int = 20,
        checkpoint_path: Optional[str] = None,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            checkpoint_path: Path to save best model weights
            mode: 'min' for loss (lower is better), 'max' for accuracy
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score: Optional[float] = None
        self.best_epoch = 0
        self.should_stop = False

    def __call__(self, score: float, model: nn.Module, epoch: int) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value (loss or accuracy)
            model: Model to checkpoint
            epoch: Current epoch number

        Returns:
            True if training should stop, False otherwise
        """
        is_improvement = self._is_improvement(score)

        if is_improvement:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self._save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.verbose:
                    print(f'Early stopping! Best epoch: {self.best_epoch}')
                self.should_stop = True

        return self.should_stop

    def _is_improvement(self, score: float) -> bool:
        """Check if the score is an improvement over best."""
        if self.best_score is None:
            return True

        if self.mode == 'min':
            return score < self.best_score
        else:
            return score > self.best_score

    def _save_checkpoint(self, model: nn.Module) -> None:
        """Save model checkpoint."""
        if self.checkpoint_path:
            Path(self.checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)

    def load_best(self, model: nn.Module) -> nn.Module:
        """Load the best checkpoint into the model."""
        if self.checkpoint_path and Path(self.checkpoint_path).exists():
            model.load_state_dict(torch.load(self.checkpoint_path))
            if self.verbose:
                print(f'Loaded best model from epoch {self.best_epoch}')
        return model
