from sklearn.neural_network import MLPClassifier


class MLPLevel0:
    """
    Multi-Layer Perceptron Level-0 base learner.

    This wrapper provides a unified interface for the stacking pipeline:
    - fit()
    - predict_proba()
    - predict()
    """

    def __init__(self, random_state=42):
        self.model = MLPClassifier(
            hidden_layer_sizes=(30,),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=300,
            random_state=random_state,
        )

    def fit(self, X_train, y_train):
        """Train the MLP model."""
        self.model.fit(X_train, y_train)
        return self

    def predict_proba(self, X):
        """Return probability predictions for the positive class."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        """Return binary predictions using a probability threshold."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
