"""Tests for the leak-free preprocessor.

The core guarantee: every data-dependent decision (kept signals, medians) is
learned from the FIT data only. A test split must never influence them.
"""

import numpy as np
import pandas as pd

from secom_yield.preprocessing import LeakFreePreprocessor


def _frame(n=20, seed=0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "time": np.arange(n, dtype=float),
            "s1": rng.normal(size=n),
            "s2": rng.normal(size=n),
        }
    )


def test_imputation_uses_train_only_statistics():
    """The leakage test: test-fold values must not shift imputation medians."""
    X_train = pd.DataFrame({"time": [1, 2, 3, 4, 5.0],
                            "s1": [1.0, 2.0, np.nan, 1.0, 2.0]})  # train median = 1.5
    X_test = pd.DataFrame({"time": [6, 7, 8.0],
                           "s1": [np.nan, 200.0, 200.0]})  # would drag a pooled median up
    prep = LeakFreePreprocessor().fit(X_train)
    Xt = prep.transform(X_test)
    # train-only median (1.5), NOT the pooled median
    assert Xt[0, 0] == 1.5


def test_drops_high_missing_rate_signals():
    X = _frame(20)
    X["s_gappy"] = [np.nan] * 15 + list(np.random.RandomState(1).normal(size=5))
    prep = LeakFreePreprocessor(max_missing_rate=0.5).fit(X)
    assert "s_gappy" in prep.dropped_columns_
    assert "s_gappy" not in prep.get_feature_names_out()


def test_drops_constant_signals():
    X = _frame(20)
    X["s_const"] = 7.0
    prep = LeakFreePreprocessor().fit(X)
    assert "s_const" in prep.dropped_columns_


def test_kept_columns_come_from_fit_data_only():
    """A signal constant in train but varying in test is still dropped."""
    X_train = _frame(20); X_train["s_x"] = 3.0
    X_test = _frame(10, seed=9); X_test["s_x"] = np.arange(10, dtype=float)
    prep = LeakFreePreprocessor().fit(X_train)
    assert "s_x" in prep.dropped_columns_
    Xt = prep.transform(X_test)
    assert Xt.shape[1] == len(prep.get_feature_names_out())


def test_transform_output_has_no_nans():
    X = _frame(30)
    X.loc[::3, "s1"] = np.nan
    prep = LeakFreePreprocessor().fit(X)
    assert not np.isnan(prep.transform(X)).any()
