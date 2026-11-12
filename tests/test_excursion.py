"""Tests for excursion monitoring."""

import numpy as np
import pandas as pd

from secom_yield.excursion import ExcursionMonitor, order_by_time, split_reference_monitor


def _series(n=100, seed=0, shift=0.0):
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "s1": rng.normal(shift, 1, n),
        "s2": rng.normal(0, 1, n),
        "s3": rng.normal(0, 1, n),
    })


def test_order_by_time_sorts_runs():
    X = pd.DataFrame({"s1": [3.0, 1.0, 2.0]})
    y = pd.Series([0, 1, 0])
    t = pd.Series([30.0, 10.0, 20.0])
    Xo, yo = order_by_time(X, y, t)
    assert list(Xo["s1"]) == [1.0, 2.0, 3.0]
    assert list(yo) == [1, 0, 0]


def test_split_reference_monitor_halves():
    X = _series(100)
    y = pd.Series(np.zeros(100, dtype=int))
    Xr, yr, Xm, ym = split_reference_monitor(X, y, ref_frac=0.5)
    assert len(Xr) == 50 and len(Xm) == 50


def test_control_limits_are_mean_plus_minus_3sigma():
    X_ref = _series(200, seed=1)
    mon = ExcursionMonitor(top_signals=["s1", "s2", "s3"]).fit(X_ref)
    lim = mon.control_limits_["s1"]
    mu, sigma = X_ref["s1"].mean(), X_ref["s1"].std(ddof=0)  # population std, as in fit
    assert lim["lcl"] == mu - 3 * sigma
    assert lim["ucl"] == mu + 3 * sigma


def test_monitor_flags_planted_anomalies():
    X_ref = _series(300, seed=2)
    X_mon = _series(50, seed=3)
    X_mon.loc[0, "s1"] = 25.0  # blatant excursion
    X_mon.loc[1, "s2"] = -25.0
    mon = ExcursionMonitor(top_signals=["s1", "s2", "s3"]).fit(X_ref)
    scored = mon.score(X_mon)
    assert scored.loc[0, "n_chart_breaches"] >= 1
    assert "s1" in scored.loc[0, "breached_signals"]
    assert scored.loc[1, "n_chart_breaches"] >= 1
    # normal runs should mostly be clean
    assert (scored.loc[2:, "n_chart_breaches"] == 0).mean() > 0.9
