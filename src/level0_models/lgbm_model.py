import numpy as np
from lightgbm import LGBMRegressor


class LGBMLevel0:
    """
    LightGBM Level-0 base learner.

    The model returns probability-like outputs for the positive class.
    """

    def __init__(self, random_state=42):
        self.model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=20,
            min_child_samples=50,
            subsample=0.7,
            subsample_freq=5,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=2.0,
            objective="regression",
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )

    def fit(self, X_train, y_train):
        """Train the LightGBM model."""
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

    def feature_importance(self, feature_names):
        """Return feature importance values as a dictionary."""
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))
