import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


class SVRLevel0:
    """
    Support Vector Regression Level-0 base learner.

    SVR is used as a regression-style base learner and its outputs are
    clipped to the [0, 1] interval to provide probability-like scores.
    """

    def __init__(self):
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "svr",
                    SVR(
                        kernel="linear",
                        C=0.01,
                        epsilon=0.2,
                    ),
                ),
            ]
        )

    def fit(self, X_train, y_train):
        """Train the SVR model."""
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        """Return probability-like predictions for the positive class."""
        proba = self.model.predict(X)
        return np.clip(proba, 0.0, 1.0)

    def predict(self, X, threshold=0.5):
        """Return binary predictions using a probability threshold."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
