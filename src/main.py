"""Run the nine fair feature-selection/classifier experiments from the terminal."""

from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd

from src.data_preprocessing import prepare_raw_data
from src.evaluate_model import (
    calculate_metrics,
    classification_report_text,
    save_confusion_matrix,
    save_feature_importance,
    save_model_comparison,
)
from src.feature_engineering import add_domain_features, ENGINEERED_FEATURE_COLUMNS
from src.feature_selection import get_feature_selection_results
from src.predict import collect_user_input, display_prediction, predict_crop
from src.research_figures import generate_research_figures
from src.train_models import classifier_parameters, train_classifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SELECTION_DIR = RESULTS_DIR / "feature_selection"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
IMPORTANCE_DIR = RESULTS_DIR / "feature_importance"
SELECTION_ORDER = ["MI", "RFE", "MI+RFE"]
MODEL_ORDER = ["XGBoost", "LightGBM", "Random Forest"]
SELECTION_LABELS = {"MI": "MI", "RFE": "RFE", "MI+RFE": "MI+RFE"}


def line(char="=", length=78):
    print(char * length)


def print_metric_block(metrics: dict):
    print(f"Accuracy  : {metrics['Accuracy'] * 100:.2f}%")
    print(f"Precision : {metrics['Precision'] * 100:.2f}%")
    print(f"Recall    : {metrics['Recall'] * 100:.2f}%")
    print(f"F1 Score  : {metrics['F1 Score'] * 100:.2f}%")
    print(f"ROC-AUC   : {metrics['ROC_AUC']:.6f}")


def slug(text: str) -> str:
    return text.lower().replace(" ", "_").replace("+", "plus")


def overlap_report(X_train: pd.DataFrame, X_test: pd.DataFrame) -> dict:
    """Check exact and rounded-to-2-decimal feature overlap across the split."""
    exact_train = set(map(tuple, X_train.to_numpy()))
    exact_test = set(map(tuple, X_test.to_numpy()))
    rounded_train = set(map(tuple, X_train.round(2).to_numpy()))
    rounded_test = set(map(tuple, X_test.round(2).to_numpy()))
    return {
        "exact_feature_overlap": len(exact_train & exact_test),
        "near_identical_feature_overlap_rounded_2dp": len(rounded_train & rounded_test),
    }


def main():
    line()
    print("ML BASED CROP RECOMMENDATION SYSTEM")
    print("9 EXPERIMENTS: 3 FEATURE-SELECTION METHODS × 3 CLASSIFIERS")
    line()

    data = prepare_raw_data()
    summary = data["summary"]
    print("\n## DATASET INFORMATION")
    print(f"Samples          : {summary['num_samples']}")
    print(f"Original features: {summary['num_features']}")
    print(f"Crop classes     : {summary['num_classes']}")
    print(f"Missing values   : {summary['missing_values']}")
    print(f"Duplicates removed: {summary['duplicates_removed']}")
    print(f"Class distribution: {summary['class_distribution']}")

    overlap = overlap_report(data["X_train"], data["X_test"])
    print("\n## PREPROCESSING AND LEAKAGE CHECKS")
    print("Stratified split : 80% training / 20% testing, random_state=42")
    print("Target excluded  : label is not part of the input matrix")
    print("Exact train-test feature overlap              :", overlap["exact_feature_overlap"])
    print("Near-identical overlap after rounding to 2 dp :", overlap["near_identical_feature_overlap_rounded_2dp"])
    print("Feature selectors fitted on training data only: YES")
    print("Same split used by all 9 experiments          : YES")

    X_train = add_domain_features(data["X_train"])
    X_test = add_domain_features(data["X_test"])
    print("\n## DOMAIN-BASED AGRICULTURAL FEATURE ENGINEERING")
    print("Derived features:", ", ".join(ENGINEERED_FEATURE_COLUMNS))
    print(f"Training matrix: {X_train.shape[0]} x {X_train.shape[1]}")
    print(f"Testing matrix : {X_test.shape[0]} x {X_test.shape[1]}")

    selection_results = get_feature_selection_results(X_train, data["y_train"])
    for selection_name, details in selection_results.items():
        if details["mi_scores"] is not None:
            details["mi_scores"].to_csv(SELECTION_DIR / f"{slug(selection_name)}_mi_scores.csv", index=False)
        if details["rfe_ranking"] is not None:
            details["rfe_ranking"].to_csv(SELECTION_DIR / f"{slug(selection_name)}_rfe_ranking.csv", index=False)

    all_results = []
    bundles = {}
    print("\n## NINE EXPERIMENTS")
    for selection_name in SELECTION_ORDER:
        selected = selection_results[selection_name]["selected_features"]
        print(f"\n{selection_name} selected features ({len(selected)}): {', '.join(selected)}")
        for model_name in MODEL_ORDER:
            start = perf_counter()
            bundle = train_classifier(model_name, X_train[selected], data["y_train"], selected)
            training_time = perf_counter() - start
            metrics = calculate_metrics(bundle, X_test[selected], data["y_test"])
            train_metrics = calculate_metrics(bundle, X_train[selected], data["y_train"])
            experiment_key = f"{selection_name} | {model_name}"
            bundle["feature_selection"] = selection_name
            bundle["selection_details"] = {
                "selected_features": selected,
                "mi_candidates": selection_results[selection_name]["mi_candidates"],
            }
            bundles[experiment_key] = bundle
            print(f"\n{experiment_key}")
            print_metric_block(metrics)
            print(f"Training Accuracy: {train_metrics['Accuracy'] * 100:.2f}%")
            print(f"Training Time    : {training_time:.3f} seconds")

            file_key = f"{slug(selection_name)}_{slug(model_name)}"
            save_confusion_matrix(
                bundle, X_test[selected], data["y_test"],
                CONFUSION_DIR / f"confusion_matrix_{file_key}.png",
                title=f"Confusion Matrix — {selection_name} + {model_name}",
            )
            save_feature_importance(
                bundle, IMPORTANCE_DIR / f"feature_importance_{file_key}.png",
                title=f"Feature Importance — {selection_name} + {model_name}",
            )
            report_path = RESULTS_DIR / f"classification_report_{file_key}.txt"
            report_path.write_text(
                classification_report_text(bundle, X_test[selected], data["y_test"]),
                encoding="utf-8",
            )
            all_results.append({
                "Feature_Selection": SELECTION_LABELS[selection_name],
                "Model": model_name,
                "Selected_Features": " | ".join(selected),
                "Num_Features": len(selected),
                "Train_Accuracy": train_metrics["Accuracy"],
                "Test_Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1_Score": metrics["F1 Score"],
                "ROC_AUC": metrics["ROC_AUC"],
                "Training_Time_Seconds": training_time,
            })

            print("Classification report saved to:", report_path)

    comparison = pd.DataFrame(all_results).sort_values(
        ["Test_Accuracy", "F1_Score", "Precision"], ascending=False
    ).reset_index(drop=True)
    save_model_comparison(comparison.to_dict("records"), RESULTS_DIR / "model_comparison.csv")
    best_row = comparison.iloc[0].to_dict()
    best_key = f"{best_row['Feature_Selection']} | {best_row['Model']}"
    best_bundle = bundles[best_key]
    generate_research_figures(all_results, bundles, X_test, data["y_test"], best_row, RESULTS_DIR / "figures")
    best_model_path = MODELS_DIR / "best_crop_model.pkl"
    joblib.dump(best_bundle, best_model_path)

    print("\n")
    line()
    print("FINAL 9-EXPERIMENT COMPARISON")
    line()
    print(f"{'Feature Selection':<18}{'Model':<17}{'Accuracy':>10}{'Precision':>11}{'Recall':>10}{'F1':>10}{'AUC':>10}")
    for row in all_results:
        print(f"{row['Feature_Selection']:<18}{row['Model']:<17}{row['Test_Accuracy']*100:>9.2f}%{row['Precision']*100:>10.2f}%{row['Recall']*100:>9.2f}%{row['F1_Score']*100:>9.2f}%{row['ROC_AUC']:>10.6f}")
    line()
    print(f"BEST COMBINATION : {best_row['Feature_Selection']} + {best_row['Model']}")
    print(f"BEST TEST ACCURACY: {best_row['Test_Accuracy'] * 100:.2f}%")
    print("Model parameters:", classifier_parameters(best_row["Model"]))
    print("Saved best model:", best_model_path)
    print("Feature-selection audit files:", SELECTION_DIR)

    line()
    print("CROP RECOMMENDATION")
    line()
    while True:
        input_df = collect_user_input()
        result = predict_crop(best_bundle, input_df)
        display_prediction(result)
        if input("\nPredict another crop? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    for directory in (MODELS_DIR, RESULTS_DIR, SELECTION_DIR, CONFUSION_DIR, IMPORTANCE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    main()
