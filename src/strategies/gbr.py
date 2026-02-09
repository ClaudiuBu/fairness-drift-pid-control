import numpy as np
from src.strategies.base import Strategy


def _compute_group_weights(y: np.ndarray, A: np.ndarray) -> dict:
    """Compute group-based weights from labels and sensitive attribute."""
    count_A0 = np.sum(A == 0)
    count_A1 = np.sum(A == 1)

    if count_A0 == 0 or count_A1 == 0:
        return {0: 1.0, 1: 1.0}

    rate0 = y[A == 0].mean() if count_A0 > 0 else 0.0
    rate1 = y[A == 1].mean() if count_A1 > 0 else 0.0

    if np.isclose(rate0, rate1):
        return {0: 1.0, 1: 1.0}

    if rate0 < rate1:
        return {0: (rate1 / max(rate0, 1e-6)), 1: 1.0}
    return {0: 1.0, 1: (rate0 / max(rate1, 1e-6))}


class StaticGBRStrategy(Strategy):
    """Static Group-Based Reweighting (computed once at initialization)."""

    def __init__(self, weights_map: dict = None, name: str = "static"):
        super().__init__(name)
        self.weights_map = weights_map

    def initialize(self, y: np.ndarray, A: np.ndarray):
        if self.weights_map is None:
            self.weights_map = _compute_group_weights(y, A)

    def get_sample_weights(self, y: np.ndarray, A: np.ndarray) -> np.ndarray:
        return np.array([self.weights_map[a] for a in A], dtype=float)


class SlidingGBRStrategy(Strategy):
    """Adaptive Group-Based Reweighting (computed from sliding window history)."""

    def __init__(self, window_size: int = 1000):
        super().__init__("sliding")
        self.history_y = None
        self.history_A = None
        self.window_size = window_size

    def initialize(self, y: np.ndarray, A: np.ndarray):
        """Set initial history."""
        self.history_y = y.copy()
        self.history_A = A.copy()

    def _calculate_weights_from_history(self) -> dict:
        """Compute weights from past data only."""
        return _compute_group_weights(self.history_y, self.history_A)

    def get_sample_weights(self, y: np.ndarray, A: np.ndarray) -> np.ndarray:
        weights_map = self._calculate_weights_from_history()
        return np.array([weights_map[a] for a in A], dtype=float)

    def update_history(self, y: np.ndarray, A: np.ndarray):
        """Add current batch to history and trim if needed."""
        self.history_y = np.concatenate([self.history_y, y])
        self.history_A = np.concatenate([self.history_A, A])

        if len(self.history_y) > self.window_size:
            self.history_y = self.history_y[-self.window_size:]
            self.history_A = self.history_A[-self.window_size:]