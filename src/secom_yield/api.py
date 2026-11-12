"""FastAPI service: yield prediction and excursion scoring.

Endpoints
---------
GET  /health     service + artifact status
POST /predict    {"signals": [590 floats, null allowed]} -> fail probability
POST /excursion  {"signals": [590 floats, null allowed]} -> excursion scores

Artifacts are produced by ``scripts/train.py`` and loaded once at startup.
Set ARTIFACTS_DIR to override the default ``artifacts/`` directory.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

N_INPUTS = 590  # time + 589 sensor signals, matching secom.data columns

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))


class Signals(BaseModel):
    signals: list[Optional[float]] = Field(
        ..., description=f"Exactly {N_INPUTS} values (time + 589 signals); null = missing"
    )


def _load_artifacts(base: Path) -> dict:
    needed = ["prep.joblib", "gbm.joblib", "monitor.joblib",
              "feature_names.json", "metrics.json"]
    missing = [f for f in needed if not (base / f).exists()]
    if missing:
        raise RuntimeError(
            f"Missing artifacts in {base}: {missing}. Run `python scripts/train.py` first."
        )
    return {
        "prep": joblib.load(base / "prep.joblib"),
        "gbm": joblib.load(base / "gbm.joblib"),
        "monitor": joblib.load(base / "monitor.joblib"),
        "feature_names": json.loads((base / "feature_names.json").read_text()),
        "metrics": json.loads((base / "metrics.json").read_text()),
    }


def create_app(artifacts_dir: Path | None = None) -> FastAPI:
    base = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
    bundle = _load_artifacts(base)
    feature_names: list[str] = bundle["feature_names"]

    app = FastAPI(
        title="Semiconductor Yield Prediction API",
        description="Fail-probability prediction and excursion scoring for wafer-fab runs.",
        version="0.1.0",
    )

    def _frame(payload: Signals) -> pd.DataFrame:
        if len(payload.signals) != N_INPUTS:
            raise HTTPException(
                status_code=422,
                detail=f"Expected {N_INPUTS} signals, got {len(payload.signals)}.",
            )
        vals = [np.nan if v is None else float(v) for v in payload.signals]
        return pd.DataFrame([vals], columns=feature_names)

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": bundle["metrics"].get("gbm_implementation", "gbm"),
            "cv_roc_auc_mean": bundle["metrics"]["models"]["gbm"]["mean"],
            "n_features": len(feature_names),
        }

    @app.post("/predict")
    def predict(payload: Signals):
        X = _frame(payload)
        proba = float(bundle["gbm"].predict_proba(X)[0, 1])
        return {"fail_probability": proba}

    @app.post("/excursion")
    def excursion(payload: Signals):
        X = _frame(payload)
        prep = bundle["prep"]
        mon = bundle["monitor"]
        Xi = pd.DataFrame(
            prep.transform(X), columns=list(prep.get_feature_names_out())
        )
        scored = mon.score(Xi).iloc[0]
        return {
            "anomaly_score": float(scored["anomaly_score"]),
            "is_anomaly": bool(scored["is_anomaly"]),
            "n_chart_breaches": int(scored["n_chart_breaches"]),
            "breached_signals": list(scored["breached_signals"]),
            "is_excursion": bool(
                scored["is_anomaly"] or scored["n_chart_breaches"] > 0
            ),
        }

    return app


app = None
try:
    app = create_app()
except RuntimeError:
    # Artifacts not built yet (e.g. during `pip install` or docs builds).
    # Importing the module still exposes create_app() for tests.
    app = None
