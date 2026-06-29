import argparse
from pathlib import Path

import pandas as pd

from stacking_pipeline import run_stacking


G2_FEATURES = [
    "F1_clr",
    "F4_clr",
    "Fe_oxide",
    "Phylic",
    "argillic",
    "Microdiorite_Micromonzonite",
    "Granodiorite",
    "NS",
    "Fault_Densi",
    "NWSE",
]


def get_repo_root():
    return Path(__file__).resolve().parents[1]


def get_config(group_name):
    group_name = group_name.upper()
    repo_root = get_repo_root()

    if group_name not in ["G1", "G2"]:
        raise ValueError("Unknown group name. Use G1 or G2.")

    data_dir = repo_root / "data" / group_name
    train_path = data_dir / "Train_data.xlsx"
    test_path = data_dir / "testnew.xlsx"
    output_dir = repo_root / "results" / group_name

    if group_name == "G1":
        feature_cols = None
    else:
        feature_cols = G2_FEATURES

    return train_path, test_path, feature_cols, output_dir


def infer_feature_cols_if_needed(train_path, feature_cols):
    if feature_cols is not None:
        return feature_cols

    train_df = pd.read_excel(train_path)

    excluded_cols = ["pointid", "x", "y", "Label", "block_id"]
    inferred_features = [c for c in train_df.columns if c not in excluded_cols]

    print(f"[G1] Auto-detected {len(inferred_features)} predictor variables.")
    print("[G1] Feature list:", inferred_features)

    return inferred_features


def check_input_files(train_path, test_path):
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"Prediction/test file not found: {test_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run the multi-level stacking workflow for G1 or G2 datasets."
    )

    parser.add_argument(
        "--group",
        type=str,
        required=True,
        choices=["G1", "G2"],
        help="Select modeling scenario: G1 or G2.",
    )

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--block_size", type=int, default=5000)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    train_path, test_path, feature_cols, output_dir = get_config(args.group)

    check_input_files(train_path, test_path)

    feature_cols = infer_feature_cols_if_needed(train_path, feature_cols)

    print("\n==============================")
    print(f"Running multi-level stacking for {args.group}")
    print("==============================")
    print("Training file:", train_path)
    print("Prediction/test file:", test_path)
    print("Output directory:", output_dir)
    print("Number of predictor variables:", len(feature_cols))
    print("==============================\n")

    results = run_stacking(
        train_path=str(train_path),
        test_path=str(test_path),
        feature_cols=feature_cols,
        output_dir=str(output_dir),
        group_name=args.group,
        block_size=args.block_size,
        n_splits=args.n_splits,
        train_ratio=args.train_ratio,
        threshold=args.threshold,
        random_state=args.seed,
    )

    print("\n===== FINISHED =====")
    print("Validation AUC:", results["val_auc"])
    print("Metrics saved:", results["metrics_path"])
    print("ROC curve saved:", results["roc_path"])
    print("Full prediction CSV:", results["full_test_path"])
    print("GIS-ready prediction CSV:", results["arc_test_path"])


if __name__ == "__main__":
    main()
