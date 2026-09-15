"""Terminal input and inference for the engineered-feature XGBoost model."""

import numpy as np
import pandas as pd

from src.crop_info import get_crop_info
from src.data_preprocessing import FEATURE_COLUMNS
from src.feature_engineering import add_domain_features
from src.evaluate_model import predict_proba_bundle

VALID_RANGES = {
    "N": (0, 500), "P": (0, 500), "K": (0, 500),
    "temperature": (-10, 60), "humidity": (0, 100),
    "ph": (0, 14), "rainfall": (0, 1000),
}

PROMPTS = {
    "N": "Nitrogen (N)", "P": "Phosphorus (P)", "K": "Potassium (K)",
    "temperature": "Temperature (°C)", "humidity": "Humidity (%)",
    "ph": "Soil pH", "rainfall": "Rainfall (mm)",
}


def get_validated_float(feature: str) -> float:
    low, high = VALID_RANGES[feature]
    while True:
        raw = input(f"{PROMPTS[feature]} [{low}-{high}]: ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("  Invalid input. Please enter a numeric value.")
            continue
        if not low <= value <= high:
            print(f"  Value out of expected range ({low} - {high}). Please re-enter.")
            continue
        return value


def collect_user_input() -> pd.DataFrame:
    print("\nEnter agricultural parameters:\n")
    values = {feature: get_validated_float(feature) for feature in FEATURE_COLUMNS}
    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


def predict_crop(bundle: dict, input_df: pd.DataFrame) -> dict:
    """Engineer the user's seven inputs and predict using the saved bundle."""
    if list(input_df.columns) != FEATURE_COLUMNS:
        raise ValueError("Prediction input columns must match the original training feature order.")

    engineered_input = add_domain_features(input_df)
    expected_columns = bundle["feature_columns"]
    engineered_input = engineered_input[expected_columns]
    encoded_prediction = bundle["model"].predict(engineered_input).astype(int)[0]
    prediction = bundle["label_encoder"].inverse_transform(np.array([encoded_prediction]))[0]
    probability = predict_proba_bundle(bundle, engineered_input)[0]
    confidence = float(np.max(probability)) * 100
    info = get_crop_info(prediction)
    return {
        "recommended_crop": prediction,
        "confidence_score": confidence,
        "crop_season": info.get("season", "N/A"),
        "growing_period": info.get("growing_period", "N/A"),
        "soil_type": info.get("soil_type", "N/A"),
        "water_requirement": info.get("water_need", "N/A"),
        "cultivation_tips": info.get("tips", "N/A"),
    }


def display_prediction(result: dict):
    print("\n" + "=" * 50)
    print("CROP RECOMMENDATION")
    print("=" * 50)
    print(f"Recommended Crop : {result['recommended_crop'].upper()}")
    print(f"Confidence        : {result['confidence_score']:.2f}%")
    print("\nCrop Information")
    print(f"  Season          : {result['crop_season']}")
    print(f"  Growing Period  : {result['growing_period']}")
    print(f"  Soil Type       : {result['soil_type']}")
    print(f"  Water Requirement: {result['water_requirement']}")
    print(f"  Cultivation Tips: {result['cultivation_tips']}")
    print("=" * 50)
