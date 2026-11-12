"""Leak-free preprocessing for wafer-fab sensor data.

All data-dependent decisions (which signals to keep, imputation values) are
learned from the *training fold only* via ``fit`` and then applied to new data
via ``transform``. Wrapped in an sklearn ``Pipeline``, this guarantees no
train-test leakage inside cross-validation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LeakFreePreprocessor(BaseEstimator, TransformerMixin):
    """Drop gappy/constant signals and median-impute, fit on train data only.

    Parameters
    ----------
    max_missing_rate : float
        Drop signals whose fraction of missing values (in the fit data)
        exceeds this threshold.
    drop_time_column : bool
        Drop the ``time`` column (it is an ordering index, not a signal).
        Kept by default so run order is preserved as a feature-free index;
        set True to model on pure sensor signals.
    """

    def __init__(self, max_missing_rate: float = 0.5, drop_time_column: bool = True):
        self.max_missing_rate = max_missing_rate
        self.drop_time_column = drop_time_column

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        cols = list(X.columns)
        if self.drop_time_column and "time" in cols:
            cols.remove("time")

        missing_rate = X[cols].isna().mean()
        keep = missing_rate[missing_rate <= self.max_missing_rate].index.tolist()

        # Drop constant signals (zero variance on non-missing values).
        std = X[keep].std(skipna=True)
        keep = [c for c in keep if std[c] > 0 and not pd.isna(std[c])]

        self.kept_columns_ = keep
        self.dropped_columns_ = [c for c in cols if c not in keep]
        self.medians_ = X[keep].median(skipna=True)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        missing = [c for c in self.kept_columns_ if c not in X.columns]
        if missing:
            raise ValueError(f"Missing expected columns in transform input: {missing}")
        out = X[self.kept_columns_].copy()
        return out.fillna(self.medians_).to_numpy(dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.kept_columns_, dtype=object)
