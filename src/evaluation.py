import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

FIG_DIR = "../reports/figures"

def ensure_dir(path=FIG_DIR):
    os.makedirs(path, exist_ok=True)

def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    ensure_dir()
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(FIG_DIR, filename))
    plt.close()   # important: closes the figure so no popup

def plot_roc_curve(y_true, y_proba, label="Model", filename="roc_curve.png"):
    ensure_dir()
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")
    plt.plot([0,1],[0,1],"--",color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, filename))
    plt.close()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

def evaluate_classification(y_true, y_pred, y_proba=None):
    """
    Print and return key classification metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }

    if y_proba is not None:
        auc = roc_auc_score(y_true, y_proba)
        metrics["roc_auc"] = auc

    print("\n--- Classification Report ---")
    print(classification_report(y_true, y_pred))
    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    return metrics