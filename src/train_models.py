"""Common, reproducible classifier builders for the 9 experiments."""

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from src.train_model import RANDOM_STATE, build_xgboost_classifier


def build_lightgbm_classifier(num_classes: int):
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=250,
        learning_rate=0.08,
        max_depth=6,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
    )


def build_random_forest_classifier():
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def build_classifier(model_name: str, num_classes: int):
    if model_name == "XGBoost":
        return build_xgboost_classifier(num_classes)
    if model_name == "LightGBM":
        return build_lightgbm_classifier(num_classes)
    if model_name == "Random Forest":
        return build_random_forest_classifier()
    raise ValueError(f"Unsupported classifier: {model_name}")


def train_classifier(model_name: str, X_train, y_train, feature_columns: list) -> dict:
    encoder = LabelEncoder()
    encoded_y = encoder.fit_transform(y_train)
    model = build_classifier(model_name, len(encoder.classes_))
    model.fit(X_train, encoded_y)
    return {
        "model": model,
        "label_encoder": encoder,
        "feature_columns": list(feature_columns),
        "model_name": model_name,
    }


def classifier_parameters(model_name: str) -> dict:
    if model_name == "XGBoost":
        return {
            "n_estimators": 250, "learning_rate": 0.08, "max_depth": 6,
            "subsample": 0.9, "colsample_bytree": 0.9,
            "random_state": RANDOM_STATE, "objective": "multi:softprob",
            "eval_metric": "mlogloss",
        }
    if model_name == "LightGBM":
        return {
            "n_estimators": 250, "learning_rate": 0.08, "max_depth": 6,
            "num_leaves": 31, "subsample": 0.9, "colsample_bytree": 0.9,
            "random_state": RANDOM_STATE, "objective": "multiclass",
            "verbosity": -1,
        }
    return {
        "n_estimators": 250, "max_depth": None, "max_features": "sqrt",
        "random_state": RANDOM_STATE,
    }
