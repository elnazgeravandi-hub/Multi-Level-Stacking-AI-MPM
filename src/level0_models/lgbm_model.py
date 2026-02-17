from lightgbm import LGBMClassifier


class LGBMLevel0:
    """
    LightGBM model for Level-0 stacking.
    Same logic as RF and XGB:
        - classifier
        - predict_proba in [0,1]
        - thresholding in utils / pipeline
    """

    def __init__(self, random_state=42):
        self.model = LGBMClassifier(
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
            objective="binary",
            n_jobs=-1,
            random_state=random_state
        )

    def fit(self, X_train, y_train):
        """Train the LightGBM classifier."""
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        """Return probability predictions for the positive class."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        """Return binary predictions based on probability and threshold."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def feature_importance(self, feature_names):
        """Return feature importance as a dict."""
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))
