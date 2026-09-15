"""Publication-style figures generated from the executed experiment objects."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.evaluate_model import predict_proba_bundle, save_confusion_matrix


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _labels(records):
    return [f"{r['Feature_Selection']}\n{r['Model']}" for r in records]


def generate_research_figures(records, bundles, X_test, y_test, best_row, output_dir: Path):
    """Create figures only from measured records and held-out probabilities."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    labels = _labels(records)
    x = np.arange(len(frame))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x, frame["Test_Accuracy"] * 100, color="#2f6f4f")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xlabel("Feature-selection strategy and classifier")
    ax.set_title("Test Accuracy Comparison Across Nine Experiments")
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_dir / "accuracy_comparison.png")

    fig, ax = plt.subplots(figsize=(14, 7))
    width = 0.2
    for i, (metric, color) in enumerate((("Precision", "#1f77b4"), ("Recall", "#ff7f0e"), ("F1_Score", "#9467bd"))):
        ax.bar(x + (i - 1) * width, frame[metric] * 100, width, label=metric.replace("_", " "), color=color)
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Feature-selection strategy and classifier")
    ax.set_title("Precision, Recall and F1-score Comparison")
    ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _save(fig, output_dir / "precision_recall_f1_comparison.png")

    best_key = f"{best_row['Feature_Selection']} | {best_row['Model']}"
    best_bundle = bundles[best_key]
    best_X_test = X_test[best_bundle["feature_columns"]]
    actual = best_bundle["label_encoder"].transform(y_test)
    actual_binary = label_binarize(actual, classes=np.arange(len(best_bundle["label_encoder"].classes_)))
    probabilities = predict_proba_bundle(best_bundle, best_X_test)
    fig, ax = plt.subplots(figsize=(9, 7))
    for index, class_name in enumerate(best_bundle["label_encoder"].classes_):
        fpr, tpr, _ = roc_curve(actual_binary[:, index], probabilities[:, index])
        ax.plot(fpr, tpr, linewidth=1.1, label=f"{class_name} (AUC={auc(fpr, tpr):.3f})")
    micro_fpr, micro_tpr, _ = roc_curve(actual_binary.ravel(), probabilities.ravel())
    ax.plot(micro_fpr, micro_tpr, color="black", linewidth=2.2,
            label=f"Micro-average (AUC={auc(micro_fpr, micro_tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"Multiclass One-vs-Rest ROC — {best_key}")
    ax.legend(fontsize=6, ncol=2, loc="lower right")
    ax.grid(alpha=0.25)
    _save(fig, output_dir / "roc_curve_best_model.png")

    fig, ax = plt.subplots(figsize=(11, 7))
    for record in records:
        key = f"{record['Feature_Selection']} | {record['Model']}"
        bundle = bundles[key]
        encoded = bundle["label_encoder"].transform(y_test)
        binary = label_binarize(encoded, classes=np.arange(len(bundle["label_encoder"].classes_)))
        probs = predict_proba_bundle(bundle, X_test[bundle["feature_columns"]])
        fpr, tpr, _ = roc_curve(binary.ravel(), probs.ravel())
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{key} (micro AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Micro-average ROC Comparison of All Nine Experiments")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.25)
    _save(fig, output_dir / "roc_comparison_all_experiments.png")

    importance = pd.DataFrame({"Feature": best_bundle["feature_columns"], "Importance": best_bundle["model"].feature_importances_}).sort_values("Importance")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(importance["Feature"], importance["Importance"], color="#8a5a2b")
    ax.set_xlabel("Model feature importance")
    ax.set_title(f"Selected-Feature Importance — {best_key}")
    ax.grid(axis="x", alpha=0.25)
    _save(fig, output_dir / "feature_importance_best_model.png")

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axis("off")
    rows = [
        ["Original inputs", "N, P, K, temperature, humidity, pH, rainfall"],
        ["Nutrient features", "N/P, N/K, P/K ratios; Total NPK; Mean NPK; Nutrient Balance"],
        ["Environmental features", "Temperature-Humidity Index; Rainfall-Humidity Index"],
        ["Output representation", "15-column agricultural feature matrix"],
    ]
    table = ax.table(cellText=rows, colLabels=["Stage", "Representation"], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.0)
    ax.set_title("Domain-Based Agricultural Feature Engineering Representation", pad=18, fontweight="bold")
    _save(fig, output_dir / "feature_engineering_representation.png")

    save_confusion_matrix(best_bundle, best_X_test, y_test,
                          output_dir / "confusion_matrix_best_model.png",
                          title=f"Confusion Matrix — Best Model: {best_key}")
