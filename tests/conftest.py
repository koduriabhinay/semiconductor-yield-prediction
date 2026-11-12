import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from secom_yield.preprocessing import LeakFreePreprocessor  # noqa: E402
from secom_yield.modeling import make_gbm_classifier  # noqa: E402
from secom_yield.excursion import ExcursionMonitor  # noqa: E402


@pytest.fixture(scope="session")
def secom():
    """Real SECOM data if downloaded, else skip."""
    from secom_yield.data import load_secom

    try:
        return load_secom(ROOT / "data")
    except FileNotFoundError:
        pytest.skip("SECOM data not downloaded; run scripts/download_data.py")


@pytest.fixture()
def tiny_artifacts(tmp_path):
    """Minimal valid artifact bundle so the API can be tested in seconds."""
    from sklearn.pipeline import Pipeline

    rng = np.random.RandomState(0)
    n, p = 80, 590
    cols = ["time"] + [f"s{i}" for i in range(1, 590)]
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=cols)
    # a couple of truly informative signals
    X["s1"] = X["s1"] + 1.5
    y = pd.Series((X["s1"] + X["s2"] + rng.normal(scale=0.5, size=n) > 1.0).astype(int))

    prep = LeakFreePreprocessor().fit(X, y)
    clf = make_gbm_classifier(random_state=0)
    # shrink for speed in tests
    if hasattr(clf, "max_iter"):
        clf.set_params(max_iter=10)
    if hasattr(clf, "n_estimators"):
        clf.set_params(n_estimators=10)
    pipe = Pipeline([("prep", LeakFreePreprocessor()), ("clf", clf)]).fit(X, y)

    Xi = pd.DataFrame(pipe.named_steps["prep"].transform(X),
                      columns=list(pipe.named_steps["prep"].get_feature_names_out()))
    mon = ExcursionMonitor(top_signals=list(Xi.columns[:3])).fit(Xi)

    (tmp_path / "feature_names.json").write_text(json.dumps(cols))
    joblib.dump(pipe, tmp_path / "gbm.joblib")
    joblib.dump(pipe.named_steps["prep"], tmp_path / "prep.joblib")
    joblib.dump(mon, tmp_path / "monitor.joblib")
    (tmp_path / "metrics.json").write_text(json.dumps(
        {"models": {"gbm": {"mean": 0.9, "std": 0.01, "fold_scores": [0.9]}},
         "gbm_implementation": "test"}))
    return tmp_path
