"""Yield prediction: gradient-boosted trees vs. logistic-regression baseline.

Evaluated with 5-fold stratified cross-validation (ROC-AUC). The preprocessing
step lives *inside* the pipeline so every fold re-learns imputations and the
kept-signal set from its own training split — no leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .preprocessing import LeakFreePreprocessor

RANDOM_STATE = 42
N_SPLITS = 5


def make_gbm_classifier(random_state: int = RANDOM_STATE):
    """Gradient-boosted trees: XGBoost when available, else sklearn's HGBT."""
    try:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=random_state,
            early_stopping=False,
        )


def gbm_model_name() -> str:
    try:
        import xgboost  # noqa: F401

        return f"xgboost {__import__('xgboost').__version__}"
    except ImportError:
        return "sklearn HistGradientBoostingClassifier"


def make_pipelines(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    return {
        "gbm": Pipeline(
            [
                ("prep", LeakFreePreprocessor()),
                ("clf", make_gbm_classifier(random_state)),
            ]
        ),
        "logreg": Pipeline(
            [
                ("prep", LeakFreePreprocessor()),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, random_state=random_state)),
            ]
        ),
    }


def evaluate_cv(
    X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE
) -> dict:
    """5-fold stratified CV ROC-AUC for GBM and logistic-regression baseline."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    results: dict = {"n_splits": N_SPLITS, "scoring": "roc_auc", "models": {}}
    for name, pipe in make_pipelines(random_state).items():
        cv_res = cross_validate(
            pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=1, return_train_score=False
        )
        scores = cv_res["test_score"]
        results["models"][name] = {
            "fold_scores": [float(s) for s in scores],
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }
    results["gbm_implementation"] = gbm_model_name()
    return results


def fit_final_gbm(
    X: pd.DataFrame, y: pd.Series, random_state: int = RANDOM_STATE
) -> Pipeline:
    """Fit the GBM pipeline on the full dataset (for serving / attribution)."""
    pipe = make_pipelines(random_state)["gbm"]
    pipe.fit(X, y)
    return pipe
