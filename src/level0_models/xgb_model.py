import numpy as np
from xgboost import XGBRegressor


class XGBLevel0:
    """
    XGBoost Level-0 base learner.

    The model is implemented with a binary logistic objective and returns
    probability-like outputs for the positive class.
    """

    def __init__(self, random_state=42):
        self.model = XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            reg_alpha=2.0,
            objective="binary:logistic",
            n_jobs=-1,
            random_state=random_state,
        )

    def fit(self, X_train, y_train):
        """Train the XGBoost model."""
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
