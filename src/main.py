from src.data_loader import load_data
from src.preprocessing import (
    split_features_target,
    build_preprocessor,
    train_test_split_data,
    build_pipeline
)
from src.model import build_xgb_model, train_model, save_model
from src.evaluation import evaluate_classification, plot_confusion_matrix, plot_roc_curve

def main():
    # 1. Load data
    df = load_data("train.csv")
    # 2. Split features and target
    X, y = split_features_target(df)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # 4. Build preprocessor + model pipeline
    preprocessor = build_preprocessor(X)
    xgb = build_xgb_model()
    pipeline = build_pipeline(preprocessor, xgb)

    # 5. Train model
    pipeline = train_model(pipeline, X_train, y_train)

    # 6. Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = evaluate_classification(y_test, y_pred, y_proba)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba, label="XGBoost")

    # 7. Save trained pipeline
    save_model(pipeline, "xgb_pipeline.pkl")

if __name__ == "__main__":
    main()