"""Evaluation and research-result artifact helpers for XGBoost."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

from src.train_model import build_xgboost_classifier, RANDOM_STATE


def predict_bundle(bundle: dict, X):
    """Predict original crop labels from a saved model bundle."""
    encoded = bundle["model"].predict(X)
    return bundle["label_encoder"].inverse_transform(encoded.astype(int))


def predict_proba_bundle(bundle: dict, X):
    return bundle["model"].predict_proba(X)


def calculate_metrics(bundle: dict, X_test, y_test) -> dict:
    predictions = predict_bundle(bundle, X_test)
    encoded_actual = bundle["label_encoder"].transform(y_test)
    probabilities = predict_proba_bundle(bundle, X_test)
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0),
        "ROC_AUC": roc_auc_score(encoded_actual, probabilities, multi_class="ovr", average="weighted"),
        "predictions": predictions,
    }


def cross_validate_xgboost(X, y, folds: int = 5) -> dict:
    """Run stratified cross-validation on the proposed feature representation."""
    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(y)
    estimator = build_xgboost_classifier(len(encoder.classes_))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(estimator, X, encoded_y, cv=splitter, scoring="accuracy", n_jobs=1)
    return {
        "fold_scores": scores,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
    }


def save_model_comparison(result_rows: list[dict], path: Path) -> Path:
    """Save the measured results and selected feature names for all methods."""
    pd.DataFrame(result_rows).to_csv(path, index=False)
    return path


def save_confusion_matrix(bundle: dict, X_test, y_test, path: Path, title: str = "Confusion Matrix") -> Path:
    predictions = predict_bundle(bundle, X_test)
    labels = list(bundle["label_encoder"].classes_)
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted Crop")
    ax.set_ylabel("Actual Crop")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_feature_importance(bundle: dict, path: Path, title: str = "XGBoost Feature Importance") -> pd.DataFrame:
    importance = pd.DataFrame({
        "Feature": bundle["feature_columns"],
        "Importance": bundle["model"].feature_importances_,
    }).sort_values("Importance", ascending=False)
    importance.to_csv(path.with_suffix(".csv"), index=False)

    plot_data = importance.sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_data["Feature"], plot_data["Importance"], color="#2f6f4f")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Agricultural Feature")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return importance


def classification_report_text(bundle: dict, X_test, y_test) -> str:
    predictions = predict_bundle(bundle, X_test)
    return classification_report(y_test, predictions, zero_division=0)
