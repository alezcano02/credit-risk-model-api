# REST API to score credit card clients
#
# uvicorn creditapi.app:app --reload

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from creditapi.features import build_features
from creditapi.monitoring import drift_report

RAIZ = Path(__file__).resolve().parents[1]
app = FastAPI(title="Credit PD model", version="0.1.0")
_modelo = None


def modelo():
    global _modelo
    if _modelo is None:
        ruta = RAIZ / "model" / "model.joblib"
        if not ruta.exists():
            raise HTTPException(status_code=503, detail="Model not trained yet: run python -m creditapi.train")
        _modelo = joblib.load(ruta)
    return _modelo


class Client(BaseModel):
    # the validation of the inputs is part of the model: a negative limit or an age of 200 must not get a score
    LIMIT_BAL: float = Field(gt=0, description="Credit limit")
    AGE: int = Field(ge=18, le=100)
    PAY_1: int = Field(ge=-2, le=9, description="Payment status last month (-2/-1/0 = on time, n = n months late)")
    PAY_2: int = Field(ge=-2, le=9)
    PAY_3: int = Field(ge=-2, le=9)
    PAY_4: int = Field(default=0, ge=-2, le=9)
    PAY_5: int = Field(default=0, ge=-2, le=9)
    PAY_6: int = Field(default=0, ge=-2, le=9)
    BILL_AMT1: float
    PAY_AMT1: float = Field(ge=0)
    PAY_AMT2: float = Field(ge=0)


class Batch(BaseModel):
    clients: list[Client] = Field(min_length=1, max_length=10000)


def grade(pd_: float) -> str:
    for limite, g in [(0.05, "A"), (0.10, "B"), (0.20, "C"), (0.40, "D")]:
        if pd_ < limite:
            return g
    return "E"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": (RAIZ / "model" / "model.joblib").exists()}


@app.get("/model-info")
def model_info():
    return json.loads((RAIZ / "model" / "model_card.json").read_text())


@app.post("/predict")
def predict(client: Client):
    x = build_features(pd.DataFrame([client.model_dump()]))
    p = float(modelo().predict_proba(x)[:, 1][0])
    return {"pd": round(p, 4), "grade": grade(p)}


@app.post("/predict-batch")
def predict_batch(batch: Batch):
    df = pd.DataFrame([c.model_dump() for c in batch.clients])
    x = build_features(df)
    p = modelo().predict_proba(x)[:, 1]
    drift = drift_report(x, p) if len(df) >= 100 else None
    return {"n": len(df), "pd": [round(float(v), 4) for v in p], "mean_pd": round(float(p.mean()), 4),
            "drift": None if drift is None else drift.astype({"status": str}).to_dict(orient="records")}
