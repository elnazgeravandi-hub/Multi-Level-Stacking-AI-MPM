import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)


# -----------------------------
# Spatial blocks & split
# -----------------------------
def make_blocks(x, y, block_size=5000):
    """
    Make spatial blocks from UTM coordinates.
    block_size in meters (e.g., 5000 = 5 km).
    """
    x_block = (x // block_size).astype(int)
    y_block = (y // block_size).astype(int)
    return x_block * 100000 + y_block


def spatial_train_val_split(df, feature_cols, target_col, block_size=5000,
                            train_ratio=0.7, random_state=42):
    """
    Perform spatial block-based train/validation split.
    Returns:
        X_train, y_train, X_val, y_val, info_dict
    """

    df = df.copy()
    df["block_id"] = make_blocks(df["x"].values, df["y"].values, block_size)

    unique_blocks = df["block_id"].unique()
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_blocks)

    n_train_blocks = int(len(unique_blocks) * train_ratio)
    train_blocks = unique_blocks[:n_train_blocks]
    val_blocks = unique_blocks[n_train_blocks:]

    train_mask = df["block_id"].isin(train_blocks)
    val_mask = df["block_id"].isin(val_blocks)

    X_train = df.loc[train_mask, feature_cols].values
    y_train = df.loc[train_mask, target_col].astype(int).values

    X_val = df.loc[val_mask, feature_cols].values
    y_val = df.loc[val_mask, target_col].astype(int).values

    info = {
        "block_size": block_size,
        "random_state": random_state,
        "n_blocks_total": len(unique_blocks),
        "n_blocks_train": len(train_blocks),
        "n_blocks_val": len(val_blocks),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "class_dist_val": dict(pd.Series(y_val).value_counts())
    }

    return X_train, y_train, X_val, y_val, info


# -----------------------------
# Metrics & evaluation
# -----------------------------
def evaluate_classifier(y_true, y_proba, threshold=0.5, positive_label=1):
    """
    Compute classification metrics and return as dict.
    """

    y_pred = (y_proba >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_proba)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, zero_division=0)

    metrics = {
        "AUC": auc,
        "Accuracy": acc,
        "Precision_pos1": prec,
        "Recall_pos1": rec,
        "F1_pos1": f1,
        "ConfusionMatrix": cm,
        "ClassificationReport": cr
    }

    return metrics


def save_metrics_csv(metrics_dict, extra_info, output_path):
    """
    Save metrics + extra info as a single-row CSV.
    """

    row = {}
    row.update(extra_info)
    row.update({k: v for k, v in metrics_dict.items() if k not in ["ConfusionMatrix", "ClassificationReport"]})

    df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


# -----------------------------
# ROC curve
# -----------------------------
def plot_and_save_roc(y_true, y_proba, auc_value, title, output_path):
    """
    Plot ROC curve and save as PNG.
    """

    fpr, tpr, _ = roc_curve(y_true, y_proba)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


# -----------------------------
# Feature importance
# -----------------------------
def save_feature_importance(feature_importance_dict, output_csv, output_png, title="Feature Importance"):
    """
    Save feature importance as CSV and bar plot.
    """

    fi_df = pd.DataFrame({
        "Feature": list(feature_importance_dict.keys()),
        "Importance": list(feature_importance_dict.values())
    }).sort_values("Importance", ascending=False)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fi_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(7, 4))
    plt.barh(fi_df["Feature"], fi_df["Importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()

    plt.savefig(output_png, dpi=300)
    plt.close()


# -----------------------------
# Test predictions & 5-class bins
# -----------------------------
def bin_probabilities_to_5classes(proba_array):
    """
    Convert probabilities [0,1] to 5 ordinal classes (1..5).
    Bins: [0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
    """

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = [1, 2, 3, 4, 5]

    s = pd.Series(proba_array)
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(int)


def save_test_outputs(test_df, proba, binary_pred, output_dir, model_name="RF", group_name="G2", suffix="spatial"):
    """
    Save full test outputs + ArcMap-ready subset.
    """

    os.makedirs(output_dir, exist_ok=True)

    out = test_df.copy()
    out[f"{model_name}_Prob"] = proba
    out[f"{model_name}_Class"] = binary_pred
    out[f"{model_name}_Class5"] = bin_probabilities_to_5classes(proba)

    full_path = os.path.join(output_dir, f"{model_name}_{group_name}_Test_FullOutput_{suffix}.csv")
    out.to_csv(full_path, index=False, encoding="utf-8-sig")

    arc_cols = ["pointid", "x", "y",
                f"{model_name}_Prob",
                f"{model_name}_Class",
                f"{model_name}_Class5"]
    arc_out = out[arc_cols].copy()
    arc_path = os.path.join(output_dir, f"{model_name}_{group_name}_For_ArcMap_{suffix}.csv")
    arc_out.to_csv(arc_path, index=False, encoding="utf-8-sig")

    return full_path, arc_path
