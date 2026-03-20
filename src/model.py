import os
import joblib
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

def build_xgb_model(
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42
) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        eval_metric="logloss"
    )

def train_model(pipeline: Pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

def save_model(pipeline: Pipeline, model_name="xgb_pipeline.pkl", models_dir="../models"):
    # Anchor relative to src/model.py
    base_dir = os.path.dirname(__file__)
    target_dir = os.path.join(base_dir, models_dir)
    os.makedirs(target_dir, exist_ok=True)

    filepath = os.path.join(target_dir, model_name)
    joblib.dump(pipeline, filepath)
    print(f"Model saved to {filepath}")


def load_model(model_name="xgb_pipeline.pkl", models_dir="../models") -> Pipeline:
    filepath = os.path.join(models_dir, model_name)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
    return joblib.load(filepath)