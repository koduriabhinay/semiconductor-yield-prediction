"""Tests for yield-prediction modeling."""

import numpy as np
import pandas as pd

from secom_yield.modeling import evaluate_cv, make_gbm_classifier, make_pipelines


def _synthetic(n=120, seed=0):
    rng = np.random.RandomState(seed)
    X = pd.DataFrame(rng.normal(size=(n, 12)),
                     columns=["time"] + [f"s{i}" for i in range(1, 12)])
    y = pd.Series(((X["s1"] + X["s2"]) > 0.5).astype(int), name="fail")
    return X, y


def test_gbm_classifier_has_predict_proba():
    clf = make_gbm_classifier(random_state=0)
    assert hasattr(clf, "predict_proba")


def test_cv_scores_are_valid_and_beat_chance():
    X, y = _synthetic()
    res = evaluate_cv(X, y, random_state=0)
    for name in ("gbm", "logreg"):
        m = res["models"][name]
        assert len(m["fold_scores"]) == 5
        assert all(0.0 <= s <= 1.0 for s in m["fold_scores"])
        assert m["mean"] > 0.75  # clearly separable synthetic signal
    assert res["gbm_implementation"]


def test_pipelines_contain_leak_free_preprocessor():
    from secom_yield.preprocessing import LeakFreePreprocessor

    for pipe in make_pipelines().values():
        assert isinstance(pipe.named_steps["prep"], LeakFreePreprocessor)
