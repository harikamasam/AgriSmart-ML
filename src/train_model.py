"""Shared XGBoost configuration used by the experiment and RFE modules."""

from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42


def build_xgboost_classifier(num_classes: int, n_estimators: int = 250):
    """Create the reproducible multi-class XGBoost classifier."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "XGBoost is required. Install project dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    return XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def train_xgboost(X_train, y_train, feature_columns: list) -> dict:
    """Fit XGBoost and return the model plus its label encoder and schema."""
    label_encoder = LabelEncoder()
    encoded_y = label_encoder.fit_transform(y_train)
    model = build_xgboost_classifier(len(label_encoder.classes_))
    model.fit(X_train, encoded_y)
    return {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": list(feature_columns),
    }
