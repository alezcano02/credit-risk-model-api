# Train the probability of default model, log the run in MLflow and save the artifact that the API loads.
#
# python -m creditapi.train

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from creditapi.features import FEATURES, build_features
from creditapi.monitoring import reference_bins

RAIZ = Path(__file__).resolve().parents[1]
DATOS = RAIZ / "data" / "default_credit_card_clients.xls"
MODELO = RAIZ / "model"


def load_data() -> pd.DataFrame:
    if not DATOS.exists():
        DATOS.parent.mkdir(exist_ok=True)
        url = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
        z = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=120).content))
        DATOS.write_bytes(z.read([n for n in z.namelist() if n.endswith(".xls")][0]))
    df = pd.read_excel(DATOS, header=1)
    return df.rename(columns={"default payment next month": "default", "PAY_0": "PAY_1"})


def main():
    df = load_data()
    x = build_features(df)
    y = df["default"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)

    params = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0}
    # MLflow tracking in a local SQLite database (the file store is deprecated)
    mlflow.set_tracking_uri("sqlite:///" + (RAIZ / "mlflow.db").as_posix())
    mlflow.set_experiment("credit-pd-model")
    with mlflow.start_run() as run:
        model = HistGradientBoostingClassifier(**params, random_state=42).fit(x_train, y_train)
        p = model.predict_proba(x_test)[:, 1]
        metrics = {"auc": roc_auc_score(y_test, p), "gini": 2 * roc_auc_score(y_test, p) - 1,
                   "brier": brier_score_loss(y_test, p), "default_rate_test": float(y_test.mean())}
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        MODELO.mkdir(exist_ok=True)
        joblib.dump(model, MODELO / "model.joblib")
        # the training distribution of each feature (bins and share of clients in each bin), used later to measure drift
        reference = {c: reference_bins(x_train[c]) for c in FEATURES}
        reference["_score"] = reference_bins(model.predict_proba(x_train)[:, 1])
        card = {"model": "HistGradientBoostingClassifier", "version": run.info.run_id[:8],
                "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "features": FEATURES, "metrics_test": {k: round(v, 4) for k, v in metrics.items()},
                "training_rows": len(x_train), "intended_use": "Probability of default next month for credit card clients",
                "limitations": "Trained on one portfolio (Taiwan, 2005); must be recalibrated before use on another population"}
        (MODELO / "model_card.json").write_text(json.dumps(card, indent=2))
        (MODELO / "reference_quantiles.json").write_text(json.dumps(reference))
        mlflow.log_artifacts(str(MODELO))
        print(json.dumps(card["metrics_test"], indent=2))


if __name__ == "__main__":
    main()
