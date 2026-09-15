"""
data_preprocessing.py
----------------------
Handles dataset loading, basic data-understanding checks, and preparation
of features/target for model training.

Dataset: data/crop_data.csv
Source columns: N, P, K, temperature, humidity, ph, rainfall, label

The dataset was inspected before writing this module:
    - 2200 rows, 8 columns, no missing values
    - 22 crop classes, perfectly balanced (100 samples each)
    - All 7 input features are numeric (no categorical encoding needed)
    - Target ('label') is a string crop name (handled natively by sklearn)

Because the dataset has no missing values and no categorical input features,
no imputation or one-hot encoding is performed here (adding them would be
fabricated preprocessing that the data does not actually require).
The active proposed model is tree-based XGBoost, so agricultural inputs
remain in their original units and no unnecessary scaling is applied.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET_COLUMN = "label"

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "crop_data.csv")


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw crop recommendation dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df


def validate_dataset_schema(df: pd.DataFrame) -> None:
    """Validate that the CSV contains the expected research inputs/target."""
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    non_numeric = [column for column in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise TypeError(f"Agricultural features must be numeric: {non_numeric}")


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Return basic data-understanding statistics used for terminal display."""
    summary = {
        "num_samples": len(df),
        "num_features": len(FEATURE_COLUMNS),
        "num_classes": df[TARGET_COLUMN].nunique(),
        "missing_values": int(df.isnull().sum().sum()),
        "class_distribution": df[TARGET_COLUMN].value_counts().to_dict(),
    }
    return summary


def exploratory_data_analysis(df: pd.DataFrame) -> dict:
    """Return descriptive EDA outputs without fitting any model."""
    return {
        "feature_statistics": df[FEATURE_COLUMNS].describe(),
        "class_distribution": df[TARGET_COLUMN].value_counts().sort_index(),
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply data cleaning only where the dataset actually requires it.
    - Drops exact duplicate rows if present.
    - Drops rows with missing values in required columns if present.
    (No such issues exist in this dataset, but the checks are kept so the
    pipeline stays correct if the CSV is ever replaced/updated.)
    """
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    after = len(df)
    if before != after:
        print(f"Data cleaning removed {before - after} invalid/duplicate rows.")
    return df.reset_index(drop=True)


def split_features_target(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def prepare_raw_data(path: str = DATA_PATH, test_size: float = 0.2, random_state: int = 42):
    """Prepare leakage-free raw numeric features for the XGBoost pipeline.

    Tree-based XGBoost does not require standardization, so the proposed
    methodology keeps the agricultural measurements in their original units.
    The split is still performed before any model fitting and is stratified by
    crop label.
    """
    df = load_dataset(path)
    validate_dataset_schema(df)
    duplicate_count = int(df.duplicated().sum())
    missing_row_count = int(df[FEATURE_COLUMNS + [TARGET_COLUMN]].isnull().any(axis=1).sum())
    df = clean_dataset(df)
    summary = summarize_dataset(df)
    eda = exploratory_data_analysis(df)
    summary["duplicates_removed"] = duplicate_count
    summary["missing_rows_removed"] = missing_row_count

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return {
        "X_train": X_train.reset_index(drop=True),
        "X_test": X_test.reset_index(drop=True),
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "summary": summary,
        "eda": eda,
        "feature_columns": FEATURE_COLUMNS,
    }
