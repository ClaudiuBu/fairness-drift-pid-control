import numpy as np
from src.strategies.base import Strategy


class ThresholdOptimizationStrategy(Strategy):
    """Post-processing via threshold optimization for fairness."""

    def __init__(self, fairness_metric: str = "tpr", name: str = "threshold_opt"):
        super().__init__(name)
        self.fairness_metric = fairness_metric
        self.thresholds = {0: 0.5, 1: 0.5}

    def compute_thresholds(self, y_proba: np.ndarray, y_true: np.ndarray, A: np.ndarray):
        """Find thresholds that equalize fairness metric across groups."""
        thresholds = {}
        for group in [0, 1]:
            mask = A == group
            y_group = y_true[mask]
            y_proba_group = y_proba[mask]

            best_threshold = 0.5
            best_score = 0

            for t in np.linspace(0, 1, 11):
                y_pred = (y_proba_group >= t).astype(int)
                acc = np.mean(y_pred == y_group)
                if acc > best_score:
                    best_score = acc
                    best_threshold = t

            thresholds[group] = best_threshold

        self.thresholds = thresholds
        return thresholds

    def get_sample_weights(self, y: np.ndarray, A: np.ndarray) -> np.ndarray:
        return np.ones_like(y, dtype=float)


class ConstrainedFairnessStrategy(Strategy):
    """Lagrangian approach to constrain fairness metric."""

    def __init__(self, fairness_constraint: float = 0.1,
                 lambda_init: float = 0.1, name: str = "constrained_fair"):
        super().__init__(name)
        self.fairness_constraint = fairness_constraint
        self.lambda_mult = lambda_init
        self.dpd_history = []

    def update_lambda(self, current_dpd: float, decay: float = 0.95):
        """Update Lagrange multiplier based on fairness constraint violation."""
        violation = max(0, abs(current_dpd) - self.fairness_constraint)
        self.lambda_mult = decay * self.lambda_mult + (1 - decay) * (0.1 * violation)
        self.dpd_history.append(current_dpd)

    def get_sample_weights(self, y: np.ndarray, A: np.ndarray) -> np.ndarray:
        """Weight samples to enforce fairness constraint."""
        w = np.ones_like(y, dtype=float)
        minority_rate = np.mean(A == 1)
        majority_rate = np.mean(A == 0)

        w[A == 1] *= (1.0 + self.lambda_mult)
        w[A == 0] *= (1.0 - self.lambda_mult * minority_rate / majority_rate)

        return np.clip(w, 0.1, 10.0)


class CalibratedFairnessStrategy(Strategy):
    """Calibrated fairness via post-processing."""

    def __init__(self, name: str = "calibrated"):
        super().__init__(name)
        self.calibration_map = {0: {}, 1: {}}

    def calibrate_probabilities(self, y_proba: np.ndarray, A: np.ndarray):
        """Build calibration map from predictions."""
        for group in [0, 1]:
            mask = A == group
            proba_group = y_proba[mask]
            self.calibration_map[group]['mean'] = np.mean(proba_group)
            self.calibration_map[group]['std'] = np.std(proba_group)

    def get_sample_weights(self, y: np.ndarray, A: np.ndarray) -> np.ndarray:
        return np.ones_like(y, dtype=float)
