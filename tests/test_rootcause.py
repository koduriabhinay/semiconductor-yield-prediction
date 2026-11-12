"""Tests for root-cause analysis."""

import numpy as np
import pandas as pd

from secom_yield.rootcause import (
    benjamini_hochberg,
    mann_whitney_screen,
    root_cause_candidates,
    shap_attribution,
)


def test_benjamini_hochberg_known_values():
    p = np.array([0.001, 0.01, 0.04, 0.5])
    adj = benjamini_hochberg(p, alpha=0.05)
    # hand-computed BH: [0.004, 0.02, 0.0533, 0.5]
    np.testing.assert_allclose(adj, [0.004, 0.02, 0.0533, 0.5], rtol=1e-2)
    assert (adj >= p).all()  # adjusted p-values never shrink


def _planted(n=200, seed=0):
    rng = np.random.RandomState(seed)
    y = pd.Series(rng.binomial(1, 0.3, size=n), name="fail")
    X = pd.DataFrame({
        "s_signal": np.where(y == 1, rng.normal(3, 1, n), rng.normal(0, 1, n)),
        "s_noise": rng.normal(size=n),
    })
    return X, y


def test_mann_whitney_screen_finds_planted_signal():
    X, y = _planted()
    screen = mann_whitney_screen(X, y)
    row = screen.set_index("signal").loc["s_signal"]
    assert bool(row["significant"])
    noise = screen.set_index("signal").loc["s_noise"]
    assert not bool(noise["significant"])


def test_root_cause_candidates_carry_honesty_disclaimer():
    X, y = _planted()
    attr = pd.Series([10.0, 0.1], index=["s_signal", "s_noise"])
    cands = root_cause_candidates(X, y, attr, top_n=5)
    assert "s_signal" in set(cands["signal"])
    assert cands["interpretation"].str.contains("Correlation").all()


def test_shap_attribution_ranks_planted_signal_first():
    from sklearn.pipeline import Pipeline
    from secom_yield.modeling import make_gbm_classifier
    from secom_yield.preprocessing import LeakFreePreprocessor

    X, y = _planted(n=300)
    X = pd.concat([X, pd.DataFrame({"time": np.arange(len(X), dtype=float)})], axis=1)
    clf = make_gbm_classifier(random_state=0)
    if hasattr(clf, "max_iter"):
        clf.set_params(max_iter=10)
    if hasattr(clf, "n_estimators"):
        clf.set_params(n_estimators=10)
    pipe = Pipeline([("prep", LeakFreePreprocessor()),
                     ("clf", clf)]).fit(X, y)
    attr = shap_attribution(pipe, X, y, max_samples=100)
    assert attr.index[0] == "s_signal"
