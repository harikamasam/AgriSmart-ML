"""
evaluate_models.py
--------------------
Evaluates every trained model on the same held-out test set and computes:
    - Accuracy
    - Precision (weighted)
    - Recall (weighted)
    - F1-score (weighted)
    - Confusion matrix (for the best model)

Weighted averaging is used because this is a multi-class (22-class)
classification problem with balanced classes; weighted precision/recall/F1
give a single comparable number per model across all classes.

Results are written to results/model_comparison.csv and the confusion
matrix of the best model is saved to results/confusion_matrix.png.
No accuracy value is ever hard-coded — everything here is computed from
the actual fitted models and the actual test split.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def evaluate_all_models(fitted_models: dict, X_test, y_test):
    """Compute metrics for every fitted model. Returns list of result dicts."""
    results = []
    predictions_cache = {}

    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        predictions_cache[name] = y_pred

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        })

    return results, predictions_cache


def select_best_model(results: list):
    """Selects the model with the highest accuracy."""
    best = max(results, key=lambda r: r["Accuracy"])
    return best["Model"], best["Accuracy"]


def save_comparison_csv(results: list, path: str = None):
    path = path or os.path.join(RESULTS_DIR, "model_comparison.csv")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
    df.to_csv(path, index=False)
    return path


def save_confusion_matrix(y_test, y_pred, model_name: str, path: str = None):
    path = path or os.path.join(RESULTS_DIR, "confusion_matrix.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name} (Best Model)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def get_classification_report_text(y_test, y_pred):
    return classification_report(y_test, y_pred, zero_division=0)
