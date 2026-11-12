"""Excursion monitoring on time-ordered runs.

Design: the first half of the time-ordered runs is the in-control *reference*
period. An Isolation Forest fitted on the reference period scores every later
run for anomalousness, and Shewhart 3-sigma control charts on the top-ranked
signals flag individual signal excursions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_TOP_SIGNALS = 3


def order_by_time(
    X: pd.DataFrame, y: pd.Series, run_time: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Sort runs by the SECOM timestamp column (NaN times go last)."""
    order = np.argsort(run_time.fillna(np.inf).to_numpy(), kind="stable")
    return X.iloc[order].reset_index(drop=True), y.iloc[order].reset_index(drop=True)


def split_reference_monitor(
    X: pd.DataFrame, y: pd.Series, ref_frac: float = 0.5
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """First ``ref_frac`` of time-ordered runs = reference; rest = monitored."""
    n_ref = int(len(X) * ref_frac)
    X_ref, y_ref = X.iloc[:n_ref], y.iloc[:n_ref]
    X_mon, y_mon = X.iloc[n_ref:], y.iloc[n_ref:]
    return X_ref, y_ref, X_mon, y_mon


class ExcursionMonitor:
    """Isolation Forest + Shewhart 3-sigma charts, fitted on a reference period."""

    def __init__(
        self,
        top_signals: list[str] | None = None,
        n_top_signals: int = N_TOP_SIGNALS,
        contamination: float = 0.05,
        random_state: int = RANDOM_STATE,
    ):
        self.top_signals = top_signals
        self.n_top_signals = n_top_signals
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X_ref: pd.DataFrame, attribution: pd.Series | None = None):
        """Fit on reference-period (already imputed) data."""
        if self.top_signals is None:
            if attribution is None:
                raise ValueError("Provide top_signals or attribution.")
            self.top_signals_ = list(attribution.head(self.n_top_signals).index)
        else:
            self.top_signals_ = list(self.top_signals)

        self.scaler_ = StandardScaler()
        Z = self.scaler_.fit_transform(X_ref.to_numpy(dtype=float))
        self.isoforest_ = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.isoforest_.fit(Z)

        # Shewhart control limits from the reference period.
        self.control_limits_ = {}
        for sig in self.top_signals_:
            vals = X_ref[sig].to_numpy(dtype=float)
            mu, sigma = float(np.mean(vals)), float(np.std(vals))
            self.control_limits_[sig] = {
                "mean": mu,
                "std": sigma,
                "lcl": mu - 3 * sigma,
                "ucl": mu + 3 * sigma,
            }
        self.feature_names_ = list(X_ref.columns)
        return self

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        """Score runs: anomaly score (higher = more anomalous) + chart breaches."""
        Z = self.scaler_.transform(X.to_numpy(dtype=float))
        anomaly = -self.isoforest_.decision_function(Z)  # higher => more anomalous
        pred = self.isoforest_.predict(Z)  # -1 = anomaly, 1 = normal

        breaches: list[list[str]] = []
        for _, row in X.iterrows():
            hit = []
            for sig in self.top_signals_:
                lim = self.control_limits_[sig]
                v = float(row[sig])
                if v < lim["lcl"] or v > lim["ucl"]:
                    hit.append(sig)
            breaches.append(hit)

        return pd.DataFrame(
            {
                "anomaly_score": anomaly,
                "is_anomaly": pred == -1,
                "n_chart_breaches": [len(h) for h in breaches],
                "breached_signals": breaches,
            }
        )
