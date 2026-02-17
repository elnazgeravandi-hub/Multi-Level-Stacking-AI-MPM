from xgboost import XGBClassifier


class XGBLevel0:
    """
    XGBoost model for Level-0 stacking.
    Handles:
        - model creation
        - training
        - prediction
    """

    def __init__(self, random_state=42):
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            reg_alpha=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state
        )

    def fit(self, X_train, y_train):
        """Train the XGB model."""
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        """Return probability predictions."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X, threshold=0.5):
        """Return binary predictions."""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def feature_importance_gain(self, feature_names):
        """
        Return gain-based feature importance as a dict.
        """
        booster = self.model.get_booster()
        gain_dict = booster.get_score(importance_type="gain")

        # Map f0 -> feature_names[0]
        mapped = {}
        for k, v in gain_dict.items():
            idx = int(k.replace("f", ""))
            mapped[feature_names[idx]] = float(v)

        # Ensure all features appear
        for f in feature_names:
            if f not in mapped:
                mapped[f] = 0.0

        return mapped
