import numpy as np
from sklearn.linear_model import Ridge


class RidgeLevel1:
    """
    Level-1 meta-model using Ridge Regression.
    Input: stacked predictions of Level-0 models.
    Output: meta-prediction (continuous, probability-like).
    """

    def __init__(self, alpha=1.0, random_state=42):
        self.model = Ridge(alpha=alpha, random_state=random_state)

    def fit(self, X_meta, y):
        self.model.fit(X_meta, y)

    def predict_proba(self, X_meta):
        preds = self.model.predict(X_meta)
        return np.clip(preds, 0.0, 1.0)

    def predict(self, X_meta, threshold=0.5):
        proba = self.predict_proba(X_meta)
        return (proba >= threshold).astype(int)
