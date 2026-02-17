import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


class SVRLevel0:
    """
    SVR Level-0 model.
    Unified logic with RF, XGB, LGBM:
        - fit()
        - predict_proba()  -> returns clipped [0,1] scores
        - predict()        -> threshold-based classification
    """

    def __init__(self):
        # Default SVR parameters (you can tune later)
        self.model = SVR(kernel="linear", C=0.01, epsilon=0.1)
        self.scaler = StandardScaler()

    def fit(self, X_train, y_train):
        """
        Fit scaler + SVR model.
        """
        X_train_sc = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_sc, y_train)

    def predict_proba(self, X):
        """
        Return probability-like scores in [0,1].
        SVR outputs continuous values → clip to [0,1].
        """
        X_sc = self.scaler.transform(X)
        preds = self.model.predict(X_sc)
        preds_clip = np.clip(preds, 0.0, 1.0)
        return preds_clip

    def predict(self, X, threshold=0.5):
        """
        Convert clipped scores to binary predictions.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def feature_importance(self, feature_names):
        """
        SVR has no native feature importance.
        We return zeros to keep the interface consistent.
        """
        return {f: 0.0 for f in feature_names}
