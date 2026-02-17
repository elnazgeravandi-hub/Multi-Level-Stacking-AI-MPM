import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)

from models_level0 import get_level0_models
from level1_models.ridge_level1 import RidgeLevel1
from level2_models.lgbm_meta_model import LGBMMetaModel


def make_blocks(x, y, block_size=5000):
    x_block = (x // block_size).astype(int)
    y_block = (y // block_size).astype(int)
    return x_block * 100000 + y_block


def clone_level0_models(level0_models):
    """
    Create fresh instances of Level-0 models for each fold/refit step.
    Assumes each model can be re-created from its class and __dict__.
    """
    cloned = {}
    for name, mdl in level0_models.items():
        params = mdl.__dict__.copy()
        # remove attributes that are created after fit (to avoid errors)
        for k in list(params.keys()):
            if k.endswith("_") or k in ("model", "scaler"):
                params.pop(k, None)
        cloned[name] = mdl.__class__(**params)
    return cloned


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
    random_state=42
):
    os.makedirs(output_dir, exist_ok=True)

    # ----------------- 1) Read data -----------------
    train_df = pd.read_excel(train_path)
    test_df  = pd.read_excel(test_path)

    id_cols = ["pointid", "x", "y"]
    target_col = "Label"

    # sanity check
    missing_train = [c for c in feature_cols + id_cols + [target_col] if c not in train_df.columns]
    missing_test  = [c for c in feature_cols + id_cols if c not in test_df.columns]
    if missing_train:
        raise ValueError(f"Missing columns in TRAIN: {missing_train}")
    if missing_test:
        raise ValueError(f"Missing columns in TEST: {missing_test}")

    # ----------------- 2) Spatial blocks -----------------
    train_df["block_id"] = make_blocks(train_df["x"].values, train_df["y"].values, block_size)
    test_df["block_id"]  = make_blocks(test_df["x"].values,  test_df["y"].values,  block_size)

    unique_blocks = train_df["block_id"].unique()
    rng = np.random.RandomState(random_state)
    rng.shuffle(unique_blocks)

    n_train_blocks = int(len(unique_blocks) * train_ratio)
    train_blocks = unique_blocks[:n_train_blocks]
    val_blocks   = unique_blocks[n_train_blocks:]

    train_mask = train_df["block_id"].isin(train_blocks)
    val_mask   = train_df["block_id"].isin(val_blocks)

    X_train = train_df.loc[train_mask, feature_cols].values
    y_train = train_df.loc[train_mask, target_col].astype(int).values
    g_train = train_df.loc[train_mask, "block_id"].values

    X_val   = train_df.loc[val_mask, feature_cols].values
    y_val   = train_df.loc[val_mask, target_col].astype(int).values

    X_test  = test_df[feature_cols].values

    print(f"[{group_name}] Spatial train:", X_train.shape, y_train.shape)
    print(f"[{group_name}] Spatial val  :", X_val.shape, y_val.shape)
    print(f"[{group_name}] Class dist (val):", dict(pd.Series(y_val).value_counts()))

    # ----------------- 3) Level-0 models -----------------
    base_level0 = get_level0_models(random_state=random_state)
    base_names = list(base_level0.keys())
    print("Level-0 models:", base_names)

    # ----------------- 4) OOF for Level-0 -----------------
    gkf = GroupKFold(n_splits=n_splits)
    Z0_oof = np.zeros((X_train.shape[0], len(base_names)), dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_train, y_train, groups=g_train), 1):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_va       = X_train[va_idx]

        # fresh models for this fold
        level0_models = clone_level0_models(base_level0)

        for j, name in enumerate(base_names):
            mdl = level0_models[name]
            mdl.fit(X_tr, y_tr)
            Z0_oof[va_idx, j] = mdl.predict_proba(X_va)

        print(f"Fold {fold} done.")

    print("Level-0 OOF matrix:", Z0_oof.shape)

    # ----------------- 5) Level-1 (Ridge) -----------------
    level1 = RidgeLevel1(alpha=1.0, random_state=random_state)
    level1.fit(Z0_oof, y_train)

    Z1_train = level1.predict_proba(Z0_oof).reshape(-1, 1)

    # ----------------- 6) Refit Level-0 on full train و ساخت Z0_val/Z0_test -----------------
    level0_models_full = clone_level0_models(base_level0)
    Z0_val  = np.zeros((X_val.shape[0], len(base_names)), dtype=float)
    Z0_test = np.zeros((X_test.shape[0], len(base_names)), dtype=float)

    for j, name in enumerate(base_names):
        mdl = level0_models_full[name]
        mdl.fit(X_train, y_train)
        Z0_val[:, j]  = mdl.predict_proba(X_val)
        Z0_test[:, j] = mdl.predict_proba(X_test)

    Z1_val  = level1.predict_proba(Z0_val).reshape(-1, 1)
    Z1_test = level1.predict_proba(Z0_test).reshape(-1, 1)

    # ----------------- 7) Level-2 (LGBM Meta) -----------------
    level2 = LGBMMetaModel(random_state=random_state)
    level2.fit(Z1_train, y_train)

    val_prob  = level2.predict_proba(Z1_val)
    test_prob = level2.predict_proba(Z1_test)

    # ----------------- 8) Metrics روی VAL -----------------
    val_pred = (val_prob >= threshold).astype(int)

    val_auc = roc_auc_score(y_val, val_prob)
    acc  = accuracy_score(y_val, val_pred)
    prec = precision_score(y_val, val_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_val, val_pred, pos_label=1, zero_division=0)
    f1   = f1_score(y_val, val_pred, pos_label=1, zero_division=0)

    print(f"\n[{group_name}] Validation AUC:", round(val_auc, 6))
    print("Accuracy:", round(acc, 4))
    print("Precision (deposit=1):", round(prec, 4))
    print("Recall (deposit=1):", round(rec, 4))
    print("F1 (deposit=1):", round(f1, 4))

    cm = confusion_matrix(y_val, val_pred)
    print("\nConfusion matrix:")
    print(cm)

    rep = classification_report(y_val, val_pred, zero_division=0)
    print("\nClassification report:")
    print(rep)

    # ----------------- 9) Save metrics & report -----------------
    metrics_df = pd.DataFrame([{
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
        "Accuracy": acc,
        "Precision_pos1": prec,
        "Recall_pos1": rec,
        "F1_pos1": f1
    }])

    metrics_path = os.path.join(output_dir, f"Stack_{group_name}_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    rep_path = os.path.join(output_dir, f"Stack_{group_name}_classification_report.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(rep)

    print("\nSaved metrics:", metrics_path)
    print("Saved report :", rep_path)

    # ----------------- 10) ROC curve -----------------
    fpr, tpr, _ = roc_curve(y_val, val_prob)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {val_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC – Stacking {group_name} (Spatial Block Split)")
    plt.legend(loc="lower right")
    plt.tight_layout()

    roc_path = os.path.join(output_dir, f"ROC_Stack_{group_name}.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    print("ROC saved:", roc_path)

    # ----------------- 11) TEST outputs (Full + ArcMap) -----------------
    test_out = test_df.copy()
    test_out["Stack_Prob"]  = test_prob
    test_out["Stack_Class"] = (test_prob >= threshold).astype(int)

    bins   = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = [1, 2, 3, 4, 5]
    test_out["Stack_Class5"] = pd.cut(
        test_out["Stack_Prob"], bins=bins, labels=labels, include_lowest=True
    ).astype(int)

    full_path = os.path.join(output_dir, f"Stack_{group_name}_Test_FullOutput_spatial.csv")
    test_out.to_csv(full_path, index=False, encoding="utf-8-sig")

    arc_out = test_out[["pointid", "x", "y", "Stack_Prob", "Stack_Class", "Stack_Class5"]].copy()
    arc_path = os.path.join(output_dir, f"Stack_{group_name}_For_ArcMap_spatial.csv")
    arc_out.to_csv(arc_path, index=False, encoding="utf-8-sig")

    print("\nSaved TEST outputs:")
    print("- Full:", full_path)
    print("- Arc :", arc_path)
    print("Arc shape:", arc_out.shape)

    return {
        "val_auc": val_auc,
        "metrics_path": metrics_path,
        "roc_path": roc_path,
        "full_test_path": full_path,
        "arc_test_path": arc_path
    }
