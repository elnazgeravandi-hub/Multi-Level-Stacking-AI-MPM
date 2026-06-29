from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor


def make_blocks(x, y, block_size=5000):
    x_block = (x // block_size).astype(int)
    y_block = (y // block_size).astype(int)
    return x_block * 100000 + y_block


def get_multistack_models(seed=42):
    return {
        "RF": RandomForestClassifier(
            n_estimators=400,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=seed,
        ),
        "LightGBM_reg": LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            num_leaves=24,
            min_child_samples=40,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=2.0,
            random_state=seed,
            verbosity=-1,
        ),
        "XGB_reg": XGBRegressor(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            reg_alpha=2.0,
            objective="binary:logistic",
            n_jobs=-1,
            random_state=seed,
        ),
        "SVR_rbf": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svr", SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")),
            ]
        ),
        "MLP_reg": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(40, 20),
                        alpha=0.001,
                        max_iter=2000,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=30,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def predict_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]

    pred = model.predict(X)
    return np.clip(pred, 0.0, 1.0)


def run_stacking(
    train_path,
    test_path,
    feature_cols,
    output_dir,
    group_name="G2",
    block_size=5000,
    n_splits=5,
    train_ratio=0.7,
    threshold=0.5,
    random_state=42,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    id_cols = ["pointid", "x", "y"]
    target_col = "Label"

    if feature_cols is None:
        excluded = id_cols + [target_col, "block_id", "spatial_block_id"]
        feature_cols = [c for c in train_df.columns if c not in excluded]

    missing_train = [
        c for c in feature_cols + id_cols + [target_col] if c not in train_df.columns
    ]
    missing_test = [c for c in feature_cols + id_cols if c not in test_df.columns]

    print("Missing in TRAIN:", missing_train)
    print("Missing in TEST :", missing_test)

    if missing_train:
        raise ValueError(f"Missing columns in TRAIN: {missing_train}")

    if missing_test:
        raise ValueError(f"Missing columns in TEST: {missing_test}")

    # Use preserved spatial_block_id when available.
    # This keeps spatial block validation reproducible after anonymizing x/y
    # into a relative local coordinate system.
    if "spatial_block_id" in train_df.columns:
        train_df["block_id"] = train_df["spatial_block_id"].astype(int)
    elif "block_id" not in train_df.columns:
        train_df["block_id"] = make_blocks(
            train_df["x"].values,
            train_df["y"].values,
            block_size=block_size,
        )

    if "spatial_block_id" in test_df.columns:
        test_df["block_id"] = test_df["spatial_block_id"].astype(int)
    elif "block_id" not in test_df.columns:
        test_df["block_id"] = make_blocks(
            test_df["x"].values,
            test_df["y"].values,
            block_size=block_size,
        )

    unique_blocks = train_df["block_id"].unique()

    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_blocks)

    n_train_blocks = int(len(unique_blocks) * train_ratio)

    train_blocks = unique_blocks[:n_train_blocks]
    val_blocks = unique_blocks[n_train_blocks:]

    train_mask = train_df["block_id"].isin(train_blocks)
    val_mask = train_df["block_id"].isin(val_blocks)

    print("Blocks total:", len(unique_blocks))
    print("Blocks train:", len(train_blocks))
    print("Blocks val :", len(val_blocks))

    X_train = train_df.loc[train_mask, feature_cols].values
    y_train = train_df.loc[train_mask, target_col].astype(int).values
    g_train = train_df.loc[train_mask, "block_id"].values

    X_val = train_df.loc[val_mask, feature_cols].values
    y_val = train_df.loc[val_mask, target_col].astype(int).values

    X_test = test_df[feature_cols].values

    print("Spatial train:", X_train.shape, y_train.shape)
    print("Spatial val :", X_val.shape, y_val.shape)
    print("Class dist (val):", dict(pd.Series(y_val).value_counts()))

    seed = random_state

    base_models = get_multistack_models(seed=seed)
    base_names = list(base_models.keys())

    print("Base models:", base_names)

    effective_splits = min(n_splits, len(np.unique(g_train)))

    if effective_splits < 2:
        raise ValueError("At least two unique training spatial blocks are required.")

    gkf = GroupKFold(n_splits=effective_splits)

    z_train_oof = np.zeros((X_train.shape[0], len(base_names)), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(
        gkf.split(X_train, y_train, groups=g_train),
        1,
    ):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_va = X_train[va_idx]

        for j, name in enumerate(base_names):
            model = clone(base_models[name])
            model.fit(X_tr, y_tr)
            z_train_oof[va_idx, j] = predict_score(model, X_va)

        print(f"Fold {fold} done.")

    print("Level-0 OOF matrix:", z_train_oof.shape)

    # ---------------------------------------------------------------------
    # Level-1 meta-learner: Ridge Regression
    # ---------------------------------------------------------------------
    level1_ridge = Ridge(
        alpha=10.0,
        random_state=seed,
    )

    level1_ridge.fit(z_train_oof, y_train)

    z_train_level1 = np.clip(
        level1_ridge.predict(z_train_oof),
        0.0,
        1.0,
    ).reshape(-1, 1)

    print("Level-1 Ridge output matrix:", z_train_level1.shape)

    # ---------------------------------------------------------------------
    # Level-2 meta-learner: LightGBM
    # ---------------------------------------------------------------------
    level2_lgbm = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.7,
        colsample_bytree=0.8,
        reg_lambda=4.0,
        objective="regression",
        n_jobs=-1,
        random_state=seed,
        verbosity=-1,
    )

    level2_lgbm.fit(z_train_level1, y_train)

    z_val = np.zeros((X_val.shape[0], len(base_names)), dtype=float)
    z_test = np.zeros((X_test.shape[0], len(base_names)), dtype=float)

    for j, name in enumerate(base_names):
        model = clone(base_models[name])
        model.fit(X_train, y_train)

        z_val[:, j] = predict_score(model, X_val)
        z_test[:, j] = predict_score(model, X_test)

    z_val_level1 = np.clip(
        level1_ridge.predict(z_val),
        0.0,
        1.0,
    ).reshape(-1, 1)

    z_test_level1 = np.clip(
        level1_ridge.predict(z_test),
        0.0,
        1.0,
    ).reshape(-1, 1)

    val_prob = np.clip(level2_lgbm.predict(z_val_level1), 0.0, 1.0)
    test_prob = np.clip(level2_lgbm.predict(z_test_level1), 0.0, 1.0)

    val_pred = (val_prob >= threshold).astype(int)

    validation_predictions = pd.DataFrame(
        {
            "y_true": y_val,
            "y_prob": val_prob,
            "y_pred_thr_0_5": val_pred,
        }
    )

    val_pred_path = output_dir / f"MultiStack_{group_name}_validation_predictions.csv"
    validation_predictions.to_csv(val_pred_path, index=False)

    print(f"Saved validation predictions: {val_pred_path}")

    val_auc = roc_auc_score(y_val, val_prob)
    accuracy = accuracy_score(y_val, val_pred)
    precision = precision_score(y_val, val_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_val, val_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_val, val_pred, pos_label=1, zero_division=0)

    print(f"\nValidation AUC (MultiStack {group_name} - spatial):", round(val_auc, 6))
    print("Accuracy:", round(accuracy, 4))
    print("Precision (deposit=1):", round(precision, 4))
    print("Recall (deposit=1):", round(recall, 4))
    print("F1 (deposit=1):", round(f1, 4))

    cm = confusion_matrix(y_val, val_pred)

    print("\nConfusion matrix (thr=0.5):")
    print(cm)

    report = classification_report(y_val, val_pred, zero_division=0)

    print("\nClassification report (thr=0.5):")
    print(report)

    metrics_df = pd.DataFrame(
        [
            {
                "Model": "MultiStack(Level0->Ridge->LightGBM)",
                "Group": group_name,
                "Split": "SpatialBlock",
                "block_size": block_size,
                "random_state": random_state,
                "n_train": len(y_train),
                "n_val": len(y_val),
                "n_blocks_total": len(unique_blocks),
                "n_blocks_train": len(train_blocks),
                "n_blocks_val": len(val_blocks),
                "threshold": threshold,
                "AUC": val_auc,
                "Accuracy": accuracy,
                "Precision_pos1": precision,
                "Recall_pos1": recall,
                "F1_pos1": f1,
            }
        ]
    )

    metrics_path = output_dir / f"MultiStack_{group_name}_spatial_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print("\nSaved metrics:", metrics_path)

    report_path = output_dir / f"MultiStack_{group_name}_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    print("Saved report:", report_path)

    fpr, tpr, _ = roc_curve(y_val, val_prob)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {val_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC – Multi-level Stacking {group_name} (Spatial Block Split)")
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = output_dir / f"ROC_MultiStack_{group_name}_spatial.png"
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print("ROC saved:", roc_path)

    test_out = test_df.copy()

    test_out["Multi_Prob"] = test_prob
    test_out["Multi_Class"] = (test_prob >= threshold).astype(int)

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = [1, 2, 3, 4, 5]

    test_out["Multi_Class5"] = pd.cut(
        test_out["Multi_Prob"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(int)

    full_test_path = output_dir / f"MultiStack_{group_name}_Test_FullOutput_spatial.csv"
    test_out.to_csv(full_test_path, index=False, encoding="utf-8-sig")

    arc_test_path = output_dir / f"MultiStack_{group_name}_For_ArcMap_spatial.csv"
    test_out[
        ["pointid", "x", "y", "Multi_Prob", "Multi_Class", "Multi_Class5"]
    ].to_csv(
        arc_test_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nSaved outputs:")
    print("- Full:", full_test_path)
    print("- Arc :", arc_test_path)
    print(
        "Arc shape:",
        test_out[
            ["pointid", "x", "y", "Multi_Prob", "Multi_Class", "Multi_Class5"]
        ].shape,
    )

    return {
        "val_auc": val_auc,
        "metrics_path": str(metrics_path),
        "report_path": str(report_path),
        "roc_path": str(roc_path),
        "full_test_path": str(full_test_path),
        "arc_test_path": str(arc_test_path),
    }
