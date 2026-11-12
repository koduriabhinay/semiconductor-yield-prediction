"""API smoke tests (FastAPI TestClient against tiny synthetic artifacts)."""

import numpy as np
from fastapi.testclient import TestClient

from secom_yield.api import N_INPUTS, create_app


def _client(tiny_artifacts):
    return TestClient(create_app(tiny_artifacts))


def _payload():
    rng = np.random.RandomState(7)
    return {"signals": [float(v) for v in rng.normal(size=N_INPUTS)]}


def test_health(tiny_artifacts):
    r = _client(tiny_artifacts).get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_features"] == N_INPUTS


def test_predict_returns_probability(tiny_artifacts):
    r = _client(tiny_artifacts).post("/predict", json=_payload())
    assert r.status_code == 200
    p = r.json()["fail_probability"]
    assert 0.0 <= p <= 1.0


def test_predict_with_missing_values(tiny_artifacts):
    payload = _payload()
    payload["signals"][::7] = [None] * len(payload["signals"][::7])
    r = _client(tiny_artifacts).post("/predict", json=payload)
    assert r.status_code == 200
    assert 0.0 <= r.json()["fail_probability"] <= 1.0


def test_predict_rejects_wrong_length(tiny_artifacts):
    r = _client(tiny_artifacts).post("/predict", json={"signals": [1.0, 2.0]})
    assert r.status_code == 422


def test_excursion_endpoint_shape(tiny_artifacts):
    r = _client(tiny_artifacts).post("/excursion", json=_payload())
    assert r.status_code == 200
    body = r.json()
    for key in ("anomaly_score", "is_anomaly", "n_chart_breaches",
                "breached_signals", "is_excursion"):
        assert key in body
    assert isinstance(body["is_excursion"], bool)
