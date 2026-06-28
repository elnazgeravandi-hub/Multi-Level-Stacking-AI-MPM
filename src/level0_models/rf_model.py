from sklearn.ensemble import RandomForestClassifier


class RFLevel0:
    """
    Random Forest Level-0 base learner.

    This wrapper provides a unified interface for the stacking pipeline:
    - fit()
    - predict_proba()
    - predict()
    - feature_importance()
    """

    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=5,
            n_jobs=-1,
            oob_score=True,
            random_state=random_state,
        )

    def fit(self, X_train, y_train):
        """Train the Random Forest model."""
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        """Return probability predictions for the positive class."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        """Return binary predictions using a probability threshold."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def feature_importance(self, feature_names):
        """Return feature importance values as a dictionary."""
        importances = self.model.feature_importances_
        return dict(zip(feature_names, importances))
