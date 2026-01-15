import numpy as np
from sklearn.linear_model import SGDClassifier

class OnlineModel:
    """Incremental (streaming) binary classifier.

    Uses SGDClassifier with log-loss and updates via partial_fit.
    This is a closer match to an 'online' learning setup than repeated batch fits.
    """
    def __init__(self, random_state: int = 42):
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            learning_rate="optimal",
            random_state=random_state
        )
        self._initialized = False
        self._classes = np.array([0, 1], dtype=int)

    def fit(self, X, y, sample_weight=None):
        # One-step online update
        if not self._initialized:
            self.model.partial_fit(X, y, classes=self._classes, sample_weight=sample_weight)
            self._initialized = True
        else:
            self.model.partial_fit(X, y, sample_weight=sample_weight)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        # Probability of positive class
        return self.model.predict_proba(X)[:, 1]
