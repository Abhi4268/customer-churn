import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def split_features_target(df: pd.DataFrame, target_col: str = "Churn", drop_cols: list = ["id"]):
    """
    Split DataFrame into features (X) and target (y).
    """
    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col].map({"Yes": 1, "No": 0})  # binary encoding
    return X, y


def build_preprocessor(X: pd.DataFrame):
    """
    Build preprocessing pipeline: OneHot for categorical, scaling for numeric.
    """
    cat_cols = X.select_dtypes(include=["object", "string"]).columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols)
        ]
    )
    return preprocessor


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Split dataset into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_pipeline(preprocessor, model):
    """
    Combine preprocessing and model into a single pipeline.
    """
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    return pipeline