from sklearn.neural_network import MLPClassifier


class MLPLevel0:
    """
    MLP neural network for Level-0 stacking.
    Unified interface with RF, XGB, LGBM, SVR:
        - fit()
        - predict_proba()
        - predict()
        - feature_importance()
    """

    def __init__(self, random_state=42):
        self.model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=random_state
        )

    def fit(self, X_train, y_train):
        """Train the MLP classifier."""
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        """Return probability predictions for the positive class."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        """Return binary predictions based on probability and threshold."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def feature_importance(self, feature_names):
        """
        MLP has no native feature importance.
        Return zeros to keep interface consistent.
        """
        return {f: 0.0 for f in feature_names}
