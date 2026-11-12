"""Root-cause analysis: per-signal Mann-Whitney U tests with BH-FDR control,
cross-checked against model-based feature attribution (SHAP when available,
permutation importance otherwise).

Results are reported as *correlations for process engineers to confirm* —
a significant test or a high attribution score does not prove causation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

FDR_ALPHA = 0.05


def benjamini_hochberg(p_values: np.ndarray, alpha: float = FDR_ALPHA) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (vectorized, no statsmodels needed)."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    # enforce monotonicity from largest to smallest
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


def mann_whitney_screen(
    X: pd.DataFrame, y: pd.Series, alpha: float = FDR_ALPHA
) -> pd.DataFrame:
    """Two-sided Mann-Whitney U test per signal (fail vs pass), BH-FDR at alpha."""
    y = np.asarray(y)
    rows = []
    for col in X.columns:
        vals = X[col].to_numpy(dtype=float)
        fail = vals[y == 1]
        passed = vals[y == 0]
        fail = fail[~np.isnan(fail)]
        passed = passed[~np.isnan(passed)]
        if len(fail) < 3 or len(passed) < 3:
            p = np.nan
            u = np.nan
        else:
            res = mannwhitneyu(fail, passed, alternative="two-sided")
            u, p = float(res.statistic), float(res.pvalue)
        rows.append({"signal": col, "u_statistic": u, "p_value": p})
    out = pd.DataFrame(rows)
    out["p_adj"] = benjamini_hochberg(out["p_value"].fillna(1.0).to_numpy(), alpha)
    out["significant"] = out["p_adj"] < alpha
    return out.sort_values("p_adj").reset_index(drop=True)


def shap_attribution(
    fitted_pipeline, X: pd.DataFrame, y: pd.Series, max_samples: int = 500
) -> pd.Series:
    """Mean |SHAP value| per signal for the fitted GBM pipeline.

    Falls back to permutation importance if the ``shap`` package is missing.
    """
    prep = fitted_pipeline.named_steps["prep"]
    clf = fitted_pipeline.named_steps["clf"]
    feature_names = list(prep.get_feature_names_out())
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=min(max_samples, len(X)), replace=False)
    Xp = prep.transform(X.iloc[idx])
    y_idx = np.asarray(y)[idx]

    try:
        import shap

        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xp)
        if isinstance(sv, list):  # binary classification may return per-class list
            sv = sv[1] if len(sv) > 1 else sv[0]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
        importance = np.abs(sv).mean(axis=0)
        method = "shap"
    except ImportError:
        from sklearn.inspection import permutation_importance

        r = permutation_importance(
            clf, Xp, y_idx,
            n_repeats=5, random_state=42, n_jobs=1,
        )
        importance = r.importances_mean
        method = "permutation_importance (shap not installed)"

    s = pd.Series(importance, index=feature_names, name="attribution").sort_values(
        ascending=False
    )
    s.attrs["method"] = method
    return s


def root_cause_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    attribution: pd.Series,
    top_n: int = 25,
    alpha: float = FDR_ALPHA,
) -> pd.DataFrame:
    """Top candidate signals: BH-significant, ranked by model attribution.

    The ``interpretation`` column carries the honesty disclaimer: these are
    statistical associations for engineers to confirm, not proven root causes.
    """
    screen = mann_whitney_screen(X, y, alpha=alpha)
    screen = screen.set_index("signal")
    attr_rank = attribution.rank(ascending=False, method="min").astype(int)

    sig = screen[screen["significant"]].copy()
    sig["attribution_rank"] = attr_rank.reindex(sig.index)
    sig = sig.sort_values(["attribution_rank", "p_adj"]).head(top_n).reset_index()
    sig["interpretation"] = (
        "Correlation with yield loss at FDR 5% — confirm physical/process "
        "causation with process engineers before acting."
    )
    return sig[
        ["signal", "p_value", "p_adj", "attribution_rank", "interpretation"]
    ]
