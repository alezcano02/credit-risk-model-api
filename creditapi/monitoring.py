# Drift monitoring: compare the data that arrives to the API with the training data (PSI per feature and for the score)

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]


def reference_bins(values) -> dict:
    """Deciles of the training data and the share of observations in each bin.

    We store the real shares because for discrete variables (months late, payment status) many deciles
    are the same number, and the bins do not have 10% of the data each.
    """
    cortes = np.unique(np.quantile(np.asarray(values, dtype=float), np.linspace(0, 1, 11)))
    edges = np.concatenate([[-np.inf], cortes[1:-1], [np.inf]])
    props = np.histogram(values, edges)[0] / len(values)
    return {"edges": [float(e) for e in edges], "props": props.tolist()}


def psi(reference: dict, values) -> float:
    """Population stability index of `values` against the training distribution."""
    edges = np.array(reference["edges"], dtype=float)
    esperado = np.clip(np.array(reference["props"]), 1e-6, None)
    real = np.clip(np.histogram(values, edges)[0] / max(len(values), 1), 1e-6, None)
    return float(np.sum((real - esperado) * np.log(real / esperado)))


def drift_report(x: pd.DataFrame, scores: np.ndarray, reference: dict | None = None) -> pd.DataFrame:
    """PSI of each feature and of the score. Usual rule: < 0.10 stable, 0.10-0.25 watch, > 0.25 shift."""
    if reference is None:
        reference = json.loads((RAIZ / "model" / "reference_quantiles.json").read_text())
    filas = [{"variable": c, "psi": psi(reference[c], x[c])} for c in x.columns if c in reference]
    filas.append({"variable": "_score", "psi": psi(reference["_score"], scores)})
    r = pd.DataFrame(filas)
    r["status"] = pd.cut(r["psi"], [-np.inf, 0.10, 0.25, np.inf], labels=["stable", "watch", "shift"])
    return r.sort_values("psi", ascending=False).reset_index(drop=True)
