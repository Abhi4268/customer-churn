import os
import pandas as pd

def load_data(filename: str, data_dir: str = "../data") -> pd.DataFrame:
    """
    Load a CSV file from the data directory.
    """
    # Anchor relative to src/data_loader.py
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, data_dir, filename)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)