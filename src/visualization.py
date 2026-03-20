import os
import pandas as pd

def load_data(filename: str, data_dir: str = "../data") -> pd.DataFrame:
    """
    Load a CSV file from the data directory.

    Parameters
    ----------
    filename : str
        Name of the CSV file (e.g., 'churn_data.csv').
    data_dir : str, optional
        Path to the data directory. Defaults to '../data' assuming
        this file lives inside src/.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame.
    """
    file_path = os.path.join(data_dir, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    return df


def preview_data(df: pd.DataFrame, n: int = 5) -> None:
    """
    Print basic info and head of the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to preview.
    n : int, optional
        Number of rows to display. Default is 5.
    """
    print("\n--- Data Info ---")
    print(df.info())
    print("\n--- First Rows ---")
    print(df.head(n))