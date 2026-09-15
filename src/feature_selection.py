"""Training-only feature selection for the three research branches."""

import pandas as pd
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from src.train_model import RANDOM_STATE

MI_FEATURE_COUNT = 12
FINAL_FEATURE_COUNT = 10


def _encode_labels(y_train):
    return LabelEncoder().fit_transform(y_train)


def _rfe_estimator(num_classes: int):
    """Use a reproducible tree estimator with feature_importances_ for RFE."""
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=None,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def mutual_information_result(X_train: pd.DataFrame, y_train, count: int = FINAL_FEATURE_COUNT):
    scores = mutual_info_classif(
        X_train, _encode_labels(y_train), random_state=RANDOM_STATE
    )
    score_table = pd.DataFrame({"Feature": X_train.columns, "MI_Score": scores})
    score_table = score_table.sort_values("MI_Score", ascending=False).reset_index(drop=True)
    selected = score_table.head(min(count, len(score_table)))["Feature"].tolist()
    return selected, score_table


def rfe_result(X_train: pd.DataFrame, y_train, count: int = FINAL_FEATURE_COUNT):
    count = min(count, X_train.shape[1])
    selector = RFE(
        estimator=_rfe_estimator(len(set(y_train))),
        n_features_to_select=count,
        step=1,
    )
    selector.fit(X_train, _encode_labels(y_train))
    ranking = pd.DataFrame({"Feature": X_train.columns, "RFE_Ranking": selector.ranking_})
    ranking["Selected"] = selector.support_
    selected = X_train.columns[selector.support_].tolist()
    return selected, ranking.sort_values(["RFE_Ranking", "Feature"]).reset_index(drop=True)


def get_feature_selection_results(X_train: pd.DataFrame, y_train) -> dict:
    """Fit all selectors on X_train only and return reproducible audit details."""
    mi_selected, mi_scores = mutual_information_result(X_train, y_train)
    rfe_selected, rfe_ranking = rfe_result(X_train, y_train)
    mi_candidates, mi_candidates_scores = mutual_information_result(
        X_train, y_train, MI_FEATURE_COUNT
    )
    combined_selected, combined_ranking = rfe_result(
        X_train[mi_candidates], y_train, FINAL_FEATURE_COUNT
    )
    return {
        "MI": {
            "selected_features": mi_selected,
            "num_features": len(mi_selected),
            "mi_scores": mi_scores,
            "rfe_ranking": None,
            "mi_candidates": mi_selected,
        },
        "RFE": {
            "selected_features": rfe_selected,
            "num_features": len(rfe_selected),
            "mi_scores": None,
            "rfe_ranking": rfe_ranking,
            "mi_candidates": None,
        },
        "MI+RFE": {
            "selected_features": combined_selected,
            "num_features": len(combined_selected),
            "mi_scores": mi_candidates_scores,
            "rfe_ranking": combined_ranking,
            "mi_candidates": mi_candidates,
        },
    }
