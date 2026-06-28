import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold

from models_level0 import get_level0_models
from level1_models.ridge_model import RidgeLevel1
from level2_models.lgbm_meta_model import LGBMMetaModel


def make_blocks(x, y, block_size=5000):
    """
    Create spatial block identifiers from x and y coordinates.
    The released x/y coordinates are anonymized relative coordinates.
    """
    x_block = (x // block_size).astype(int)
    y_block = (y // block_size).astype(int)
    return x_block * 100000 + y_block


def clone_level0_models(level0_models):
    """
    Create fresh Level-0 model instances for cross-validation and refitting.
    """
    cloned = {}

    for name, model in level0_models.items():
        if name == "RF":
            from level0_models.rf_model import RFLevel0

            cloned[name] = RFLevel0()
        elif name == "XGB":
            from level0_models.xgb_model import XGBLevel0

            cloned[name] = XGBLevel0()
        elif name == "LGBM":
            from level0_models.lgbm_model import LGBMLevel0

            cloned[name] = LGBMLevel0()
        elif name == "SVR":
            from level0_models.svr_model import SVRLevel0

            cloned[name] = SVRLevel0()
        elif name == "MLP":
            from level0_models.mlp_model import MLPLevel0

            cloned[name] = MLPLevel0()
        else:
            raise ValueError(f"Unknown Level-0 model name: {name}")

    return cloned


def _check_required_columns(train_df, test_df, feature_cols, id_cols, target_col):
    missing_train = [
        c for c in feature_cols + id_cols + [target_col]
        if c not in train_df.columns
    ]
    missing_test = [
        c for c in feature_cols + id_cols
        if c not in test_df.columns
    ]

    if missing_train:
        raise ValueError(f"Missing columns in TRAIN data: {missing_train}")

    if missing_test:
        raise ValueError(f"Missing columns in TEST data: {missing_test}")


def _save_metrics(
    output_dir,
    group_name,
    block_size,
    random_state,
    y_train,
    y_val,
    unique_blocks,
    train_blocks,
    val_blocks,
    threshold,
    val_auc,
    accuracy,
    precision,
    recall,
    f1,
):
    metrics_df = pd.DataFrame(
        [
            {
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

    metrics_path = os.path.join(output_dir, f"Stack_{group_name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    return metrics_path


def _save_roc_curve(output_dir, group_name, y_val, val_prob, val_auc):
    fpr, tpr, _ = roc_curve(y_val, val_prob)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {val_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - Stacking {group_name} Spatial Block Split")
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = os.path.join(output_dir, f"ROC_Stack_{group_name}.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    return roc_path


def _save_test_outputs(output_dir, group_name, test_df, test_prob, threshold):
    test_out = test_df.copy()
    test_out["Stack_Prob"] = test_prob
    test_out["Stack_Class"] = (test_prob >= threshold).astype(int)

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = [1, 2, 3, 4, 5]

    test_out["Stack_Class5"] = pd.cut(
        test_out["Stack_Prob"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(int)

    full_test_path = os.path.join(
        output_dir,
        f"Stack_{group_name}_Test_FullOutput_spatial.csv",
    )
    test_out.to_csv(full_test_path, index=False, encoding="utf-8-sig")

    arc_test_path = os.path.join(
        output_dir,
        f"Stack_{group_name}_For_ArcMap_spatial.csv",
    )
    arc_out = test_out[
        ["pointid", "x", "y", "Stack_Prob", "Stack_Class", "Stack_Class5"]
    ].copy()
    arc_out.to_csv(arc_test_path, index=False, encoding="utf-8-sig")

    return full_test_path, arc_test_path


def run_stacking(
    train_path,
    test_path,
    feature_cols,
    output_dir,
    group_name="G1",
    block_size=5000,
    n_splits=5,
    train_ratio=0.7,
    threshold=0.5,
    random_state=42,
):
    """
    Run the multi-level stacking workflow.

    Level 0:
        RF, XGBoost, LightGBM, SVR, and MLP

    Level 1:
        Ridge regression meta-learner

    Level 2:
        LightGBM meta-learner
    """
    os.makedirs(output_dir, exist_ok=True)

    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)

    id_cols = ["pointid", "x", "y"]
    target_col = "Label"

    _check_required_columns(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        id_cols=id_cols,
        target_col=target_col,
    )

    train_df["block_id"] = make_blocks(
        train_df["x"].values,
        train_df["y"].values,
        block_size=block_size,
    )
    test_df["block_id"] = make_blocks(
        test_df["x"].values,
        test_df["y"].values,
        block_size=block_size,
    )

    unique_blocks = train_df["block_id"].unique()

    if len(unique_blocks) < 2:
        raise ValueError("At least two spatial blocks are required.")

    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_blocks)

    n_train_blocks = int(len(unique_blocks) * train_ratio)

    if n_train_blocks < 1:
        n_train_blocks = 1

    if n_train_blocks >= len(unique_blocks):
        n_train_blocks = len(unique_blocks) - 1

    train_blocks = unique_blocks[:n_train_blocks]
    val_blocks = unique_blocks[n_train_blocks:]

    train_mask = train_df["block_id"].isin(train_blocks)
    val_mask = train_df["block_id"].isin(val_blocks)

    X_train = train_df.loc[train_mask, feature_cols].values
    y_train = train_df.loc[train_mask, target_col].astype(int).values
    groups_train = train_df.loc[train_mask, "block_id"].values

    X_val = train_df.loc[val_mask, feature_cols].values
    y_val = train_df.loc[val_mask, target_col].astype(int).values

    X_test = test_df[feature_cols].values

    print(f"[{group_name}] Spatial train:", X_train.shape, y_train.shape)
    print(f"[{group_name}] Spatial validation:", X_val.shape, y_val.shape)
    print(f"[{group_name}] Validation class distribution:")
    print(pd.Series(y_val).value_counts())

    base_level0 = get_level0_models(random_state=random_state)
    base_names = list(base_level0.keys())

    print("Level-0 models:", base_names)

    n_unique_train_groups = len(np.unique(groups_train))
    effective_splits = min(n_splits, n_unique_train_groups)

    if effective_splits < 2:
        raise ValueError(
            "At least two unique training spatial blocks are required for GroupKFold."
        )

    group_kfold = GroupKFold(n_splits=effective_splits)

    z0_oof = np.zeros((X_train.shape[0], len(base_names)), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(
        group_kfold.split(X_train, y_train, groups=groups_train),
        start=1,
    ):
        X_tr = X_train[tr_idx]
        y_tr = y_train[tr_idx]
        X_va = X_train[va_idx]

        level0_models = clone_level0_models(base_level0)

        for model_index, model_name in enumerate(base_names):
            model = level0_models[model_name]
            model.fit(X_tr, y_tr)
            z0_oof[va_idx, model_index] = model.predict_proba(X_va)

        print(f"OOF fold {fold}/{effective_splits} completed.")

    print("Level-0 OOF matrix:", z0_oof.shape)

    level1 = RidgeLevel1(alpha=10.0, random_state=random_state)
    level1.fit(z0_oof, y_train)
    z1_train = level1.predict_proba(z0_oof).reshape(-1, 1)

    level0_models_full = clone_level0_models(base_level0)

    z0_val = np.zeros((X_val.shape[0], len(base_names)), dtype=float)
    z0_test = np.zeros((X_test.shape[0], len(base_names)), dtype=float)

    for model_index, model_name in enumerate(base_names):
        model = level0_models_full[model_name]
        model.fit(X_train, y_train)
        z0_val[:, model_index] = model.predict_proba(X_val)
        z0_test[:, model_index] = model.predict_proba(X_test)

    z1_val = level1.predict_proba(z0_val).reshape(-1, 1)
    z1_test = level1.predict_proba(z0_test).reshape(-1, 1)

    level2 = LGBMMetaModel(random_state=random_state)
    level2.fit(z1_train, y_train)

    val_prob = level2.predict_proba(z1_val)
    test_prob = level2.predict_proba(z1_test)

    val_pred = (val_prob >= threshold).astype(int)

    val_auc = roc_auc_score(y_val, val_prob)
    accuracy = accuracy_score(y_val, val_pred)
    precision = precision_score(
        y_val,
        val_pred,
        pos_label=1,
        zero_division=0,
    )
    recall = recall_score(
        y_val,
        val_pred,
        pos_label=1,
        zero_division=0,
    )
    f1 = f1_score(
        y_val,
        val_pred,
        pos_label=1,
        zero_division=0,
    )

    print(f"\n[{group_name}] Validation AUC:", round(val_auc, 6))
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))

    confusion = confusion_matrix(y_val, val_pred)
    report = classification_report(y_val, val_pred, zero_division=0)

    print("\nConfusion matrix:")
    print(confusion)

    print("\nClassification report:")
    print(report)

    metrics_path = _save_metrics(
        output_dir=output_dir,
        group_name=group_name,
        block_size=block_size,
        random_state=random_state,
        y_train=y_train,
        y_val=y_val,
        unique_blocks=unique_blocks,
        train_blocks=train_blocks,
        val_blocks=val_blocks,
        threshold=threshold,
        val_auc=val_auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
    )

    report_path = os.path.join(
        output_dir,
        f"Stack_{group_name}_classification_report.txt",
    )
    with open(report_path, "w", encoding="utf-8") as file:
        file.write(report)

    roc_path = _save_roc_curve(
        output_dir=output_dir,
        group_name=group_name,
        y_val=y_val,
        val_prob=val_prob,
        val_auc=val_auc,
    )

    full_test_path, arc_test_path = _save_test_outputs(
        output_dir=output_dir,
        group_name=group_name,
        test_df=test_df,
        test_prob=test_prob,
        threshold=threshold,
    )

    print("\nSaved outputs:")
    print("Metrics:", metrics_path)
    print("Classification report:", report_path)
    print("ROC curve:", roc_path)
    print("Full prediction CSV:", full_test_path)
    print("GIS-ready prediction CSV:", arc_test_path)

    return {
        "val_auc": val_auc,
        "metrics_path": metrics_path,
        "roc_path": roc_path,
        "full_test_path": full_test_path,
        "arc_test_path": arc_test_path,
    }
