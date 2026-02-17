import numpy as np
from lightgbm import LGBMClassifier


class LGBMMetaModel:
    """
    Final Level-2 meta-model using LightGBM.
    Input: stacked predictions from Level-1.
    Output: final probability prediction in [0,1].
    """

    def __init__(self, random_state=42):
        self.model = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=24,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=2.0,
            objective="binary",
            n_jobs=-1,
            random_state=random_state
        )

    def fit(self, Z_train, y):
        """Train Level-2 LightGBM classifier."""
        self.model.fit(Z_train, y)

    def predict_proba(self, Z):
        """Return probability predictions for the positive class."""
        return self.model.predict_proba(Z)[:, 1]

    def predict(self, Z, threshold=0.5):
        """Return binary predictions based on threshold."""
        proba = self.predict_proba(Z)
        return (proba >= threshold).astype(int)

    def feature_importance(self, feature_names):
        """Return feature importance as a dict."""
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))
