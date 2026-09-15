"""Domain-based agricultural feature engineering.

The functions in this module are used both during training and inference so
that the model always receives the same feature definitions.
"""

import numpy as np
import pandas as pd

from src.data_preprocessing import FEATURE_COLUMNS

ENGINEERED_FEATURE_COLUMNS = [
    "N_P_Ratio",
    "N_K_Ratio",
    "P_K_Ratio",
    "Total_NPK",
    "Mean_NPK",
    "Nutrient_Balance",
    "Temperature_Humidity_Index",
    "Rainfall_Humidity_Index",
]

PROPOSED_FEATURE_COLUMNS = FEATURE_COLUMNS + ENGINEERED_FEATURE_COLUMNS


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Calculate a ratio without producing infinity when the denominator is 0."""
    return numerator.div(denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_domain_features(features: pd.DataFrame) -> pd.DataFrame:
    """Return original agricultural inputs plus meaningful derived features.

    - Nutrient ratios describe relative N/P/K availability.
    - Total and mean NPK describe overall nutrient level.
    - Nutrient_Balance is the minimum-to-maximum nutrient ratio (0 to 1 for
      non-negative inputs), representing how evenly the three nutrients are
      distributed.
    - Temperature/Humidity and Rainfall/Humidity indices represent combined
      environmental heat-moisture and water-availability conditions.
    """
    missing = [column for column in FEATURE_COLUMNS if column not in features.columns]
    if missing:
        raise ValueError(f"Cannot engineer features; missing columns: {missing}")

    engineered = features[FEATURE_COLUMNS].copy()
    engineered["N_P_Ratio"] = _safe_ratio(engineered["N"], engineered["P"])
    engineered["N_K_Ratio"] = _safe_ratio(engineered["N"], engineered["K"])
    engineered["P_K_Ratio"] = _safe_ratio(engineered["P"], engineered["K"])
    engineered["Total_NPK"] = engineered["N"] + engineered["P"] + engineered["K"]
    engineered["Mean_NPK"] = engineered[["N", "P", "K"]].mean(axis=1)
    nutrient_max = engineered[["N", "P", "K"]].max(axis=1)
    nutrient_min = engineered[["N", "P", "K"]].min(axis=1)
    engineered["Nutrient_Balance"] = _safe_ratio(nutrient_min, nutrient_max)
    engineered["Temperature_Humidity_Index"] = engineered["temperature"] * engineered["humidity"] / 100.0
    engineered["Rainfall_Humidity_Index"] = engineered["rainfall"] * engineered["humidity"] / 100.0
    return engineered[PROPOSED_FEATURE_COLUMNS]
