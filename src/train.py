import os
import argparse
import pandas as pd

from stacking_pipeline import run_stacking


def get_config(group_name: str):
    """
    Returns dataset paths, feature list, and output directory
    based on the selected group (G1 or G2).
    """
    if group_name == "G1":
        base_path = r"F:\Ahar-Arasbaran\GIS\MultiLevelStack\G1_All_Data"
        feature_cols = None  # Automatically inferred from dataset
        output_dir = os.path.join(base_path, "output_Stack_G1_spatial")

    elif group_name == "G2":
        base_path = r"F:\Ahar-Arasbaran\GIS\MultiLevelStack\G2_Above_0_6"
        feature_cols = [
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
        output_dir = os.path.join(base_path, "output_Stack_G2_spatial")

    else:
        raise ValueError(f"Unknown group name: {group_name}")

    train_path = os.path.join(base_path, "Train_data.xlsx")
    test_path = os.path.join(base_path, "testnew.xlsx")

    return train_path, test_path, feature_cols, output_dir


def infer_feature_cols_if_needed(train_path, feature_cols):
    """
    If feature_cols is None (G1), automatically infer features
    by excluding ID columns, target, and block_id.
    """
    if feature_cols is not None:
        return feature_cols

    df = pd.read_excel(train_path)
    id_cols = ["pointid", "x", "y"]
    target_col = "Label"
    exclude_cols = id_cols + [target_col, "block_id"]

    inferred = [c for c in df.columns if c not in exclude_cols]

    print(f"[G1] Auto-detected {len(inferred)} features.")
    print("[G1] Feature list:", inferred)

    return inferred


def main():
    parser = argparse.ArgumentParser(description="Run multi-level stacking for G1 or G2.")

    parser.add_argument(
        "--group",
        type=str,
        required=True,
        choices=["G1", "G2"],
        help="Select dataset group: G1 (all layers) or G2 (Above_0_6)."
    )

    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for final probability.")

    parser.add_argument("--block_size", type=int, default=5000,
                        help="Spatial block size in meters.")

    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of GroupKFold splits for OOF.")

    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Fraction of spatial blocks used for training.")

    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")

    args = parser.parse_args()

    # Load configuration
    train_path, test_path, feature_cols, output_dir = get_config(args.group)

    # Auto-detect features for G1
    feature_cols = infer_feature_cols_if_needed(train_path, feature_cols)

    print("\n==============================")
    print(f" Running Stacking for {args.group} ")
    print("==============================")
    print("Train file :", train_path)
    print("Test file  :", test_path)
    print("Output dir :", output_dir)
    print("Features   :", len(feature_cols))
    print("Threshold  :", args.threshold)
    print("Block size :", args.block_size)
    print("Splits     :", args.n_splits)
    print("Train ratio:", args.train_ratio)
    print("Seed       :", args.seed)
    print("==============================\n")

    # Run the full stacking pipeline
    results = run_stacking(
        train_path=train_path,
        test_path=test_path,
        feature_cols=feature_cols,
        output_dir=output_dir,
        group_name=args.group,
        block_size=args.block_size,
        n_splits=args.n_splits,
        train_ratio=args.train_ratio,
        threshold=args.threshold,
        random_state=args.seed
    )

    print("\n===== FINISHED =====")
    print("Validation AUC :", results["val_auc"])
    print("Metrics saved  :", results["metrics_path"])
    print("ROC saved      :", results["roc_path"])
    print("Test full CSV  :", results["full_test_path"])
    print("Test Arc CSV   :", results["arc_test_path"])


if __name__ == "__main__":
    main()
