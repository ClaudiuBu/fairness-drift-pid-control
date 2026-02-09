import numpy as np
from src.strategies.base import Strategy


class PIDControlStrategy(Strategy):
    """PID-controlled fairness via sample reweighting."""

    def __init__(self, pid_controller, u_clip: float = 3.0):
        super().__init__("pid")
        self.pid = pid_controller
        self.u_clip = u_clip
        self.last_u = 0.0

    def compute_control_action(self, fairness_metric: float) -> float:
        """Compute control signal from current fairness violation."""
        u = self.pid.step(fairness_metric)
        self.last_u = float(np.clip(u, -self.u_clip, self.u_clip))
        return self.last_u

    def get_sample_weights(self, y: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Apply exponential weight mapping based on control signal."""
        scale = float(np.exp(-self.last_u))
        w = np.ones_like(A, dtype=float)
        w[A == 0] *= scale
        w[A == 1] *= (1.0 / scale)
        return w