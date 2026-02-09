from abc import ABC, abstractmethod
import numpy as np


class Strategy(ABC):
    """Base class for model update strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_sample_weights(self, y: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Compute sample weights for this batch."""
        pass

    def initialize(self, y: np.ndarray, A: np.ndarray):
        """Optional hook for warm-up initialization."""
        return

    def fit(self, model, X: np.ndarray, y: np.ndarray,
            A: np.ndarray = None, sample_weight: np.ndarray = None):
        """Update model with optional custom weights."""
        if sample_weight is None:
            sample_weight = self.get_sample_weights(y, A)
        model.fit(X, y, sample_weight=sample_weight)


class BaselineStrategy(Strategy):
    """No reweighting - standard training."""

    def __init__(self):
        super().__init__("base")

    def get_sample_weights(self, y: np.ndarray, A: np.ndarray) -> np.ndarray:
        return np.ones_like(y, dtype=float)
