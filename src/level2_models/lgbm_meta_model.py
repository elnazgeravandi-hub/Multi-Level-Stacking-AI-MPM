import numpy as np
from lightgbm import LGBMRegressor


class LGBMMetaModel:
    """
    LightGBM Level-2 meta-learner.

    This model receives Level-1 Ridge outputs and produces the final
    probability-like prospectivity score.
    """

    def __init__(self, random_state=42):
        self.model = LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.8,
            reg_lambda=4.0,
            objective="regression",
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )

    def fit(self, X_train, y_train):
        """Train the Level-2 LightGBM meta-learner."""
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
