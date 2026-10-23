# Tests of the API: they train the model once (small and fast) and call the endpoints

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from creditapi.app import app
from creditapi.monitoring import psi, reference_bins
from creditapi.train import MODELO, main

cliente_bueno = {"LIMIT_BAL": 200000, "AGE": 35, "PAY_1": -1, "PAY_2": -1, "PAY_3": -1,
                 "BILL_AMT1": 5000, "PAY_AMT1": 5000, "PAY_AMT2": 5000}
cliente_malo = {"LIMIT_BAL": 20000, "AGE": 25, "PAY_1": 3, "PAY_2": 2, "PAY_3": 2, "PAY_4": 2,
                "BILL_AMT1": 21000, "PAY_AMT1": 0, "PAY_AMT2": 0}


@pytest.fixture(scope="session")
def api():
    if not (MODELO / "model.joblib").exists():
        main()
    return TestClient(app)


def test_health(api):
    r = api.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"]


def test_prediction_is_a_probability(api):
    r = api.post("/predict", json=cliente_bueno)
    assert r.status_code == 200
    assert 0 <= r.json()["pd"] <= 1


def test_late_client_has_higher_pd(api):
    bueno = api.post("/predict", json=cliente_bueno).json()["pd"]
    malo = api.post("/predict", json=cliente_malo).json()["pd"]
    assert malo > bueno


def test_invalid_input_is_rejected(api):
    r = api.post("/predict", json={**cliente_bueno, "AGE": 250})
    assert r.status_code == 422
    r = api.post("/predict", json={**cliente_bueno, "LIMIT_BAL": -5})
    assert r.status_code == 422


def test_batch_returns_drift_report(api):
    clientes = [cliente_bueno, cliente_malo] * 60
    r = api.post("/predict-batch", json={"clients": clientes})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 120 and body["drift"] is not None


def test_psi_is_zero_for_same_distribution():
    x = np.random.default_rng(0).normal(size=20000)
    ref = reference_bins(x)
    assert psi(ref, x) < 0.01
    assert psi(ref, x + 2) > 0.25


def test_psi_of_discrete_variable_is_zero_for_same_data():
    # most clients have 0 months late: many deciles are equal, the PSI must still be 0
    x = np.random.default_rng(1).choice([0, 0, 0, 0, 0, 0, 1, 2, 3], size=20000)
    assert psi(reference_bins(x), x) < 0.01
