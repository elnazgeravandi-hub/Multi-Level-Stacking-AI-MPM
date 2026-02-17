import pandas as pd
import os
import json


def load_settings(settings_path):
    """Load global settings from JSON file."""
    with open(settings_path, "r") as f:
        return json.load(f)


def load_layer_config(config_path):
    """Load layer list (G1 or G2) from JSON file."""
    with open(config_path, "r") as f:
        data = json.load(f)
    return data["layers"]


def load_train_test(train_path, test_path):
    """Load train and test Excel files."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)

    return train_df, test_df
