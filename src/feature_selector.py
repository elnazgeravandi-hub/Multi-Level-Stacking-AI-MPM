import os
from .data_loader import load_layer_config


def get_feature_list(group_name, config_dir):
    """
    Return the list of feature names for the selected group (G1 or G2).

    Parameters
    ----------
    group_name : str
        Either "G1" or "G2".
    config_dir : str
        Path to the config folder containing JSON files.

    Returns
    -------
    list
        List of feature names.
    """

    group_name = group_name.upper()

    if group_name not in ["G1", "G2"]:
        raise ValueError("Invalid group name. Use 'G1' or 'G2'.")

    json_file = "layers_G1.json" if group_name == "G1" else "layers_G2.json"
    config_path = os.path.join(config_dir, json_file)

    return load_layer_config(config_path)
