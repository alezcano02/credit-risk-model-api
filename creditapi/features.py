# The same feature engineering is used in training and in the API, so there is no training/serving skew

import pandas as pd

FEATURES = ["LIMIT_BAL", "AGE", "PAY_1", "PAY_2", "PAY_3", "BILL_AMT1", "PAY_AMT1", "PAY_AMT2",
            "utilisation", "months_late", "max_delay"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    pagos = [c for c in ["PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"] if c in d.columns]
    d["utilisation"] = (d["BILL_AMT1"] / d["LIMIT_BAL"]).clip(-0.5, 2)
    d["months_late"] = (d[pagos] > 0).sum(axis=1)
    d["max_delay"] = d[pagos].max(axis=1).clip(lower=0)
    return d[FEATURES]
