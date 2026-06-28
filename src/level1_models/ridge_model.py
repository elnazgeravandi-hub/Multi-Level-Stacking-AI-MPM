import numpy as np
from sklearn.linear_model import Ridge


class RidgeLevel1:
    """
    Ridge regression Level-1 meta-learner.

    This model receives out-of-fold Level-0 predictions and produces
    an intermediate probability-like score for the Level-2 meta-learner.
    """

    def __init__(self, alpha=10.0, random_state=42):
        self.alpha = alpha
        self.random_state = random_state
        self.model = Ridge(
            alpha=alpha,
            random_state=random_state,
        )

    def fit(self, X_train, y_train):
        """Train the Ridge meta-learner."""
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        """Return clipped probability-like predictions."""
        proba = self.model.predict(X)
        return np.clip(proba, 0.0, 1.0)

    def predict(self, X, threshold=0.5):
        """Return binary predictions using a probability threshold."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
