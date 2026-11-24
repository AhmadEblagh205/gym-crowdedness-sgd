"""Cyclic calendar features for a linear model."""
import numpy as np
import pandas as pd

FIRST_HARMONIC = ["hour_sin", "hour_cos"]
SECOND_HARMONIC = ["hour_sin2", "hour_cos2"]
FEATURES = (FIRST_HARMONIC + SECOND_HARMONIC
            + ["dow_sin", "dow_cos", "month_sin", "month_cos",
               "is_weekend", "is_during_semester", "temperature",
               "wknd_sin", "wknd_cos", "sem_sin", "sem_cos"])
FEATURES_FIRST_ONLY = [f for f in FEATURES if f not in SECOND_HARMONIC]
RAW_NEEDED = ["hour", "day_of_week", "month", "is_weekend",
              "is_during_semester", "temperature"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in RAW_NEEDED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    d = df.copy()
    if "date" in d.columns:          # use minutes when the timestamp has them
        dt = pd.to_datetime(d["date"])
        h = dt.dt.hour + dt.dt.minute / 60.0
    else:
        h = d["hour"].astype(float)
    a = 2 * np.pi * h / 24
    d["hour_sin"], d["hour_cos"] = np.sin(a), np.cos(a)
    d["hour_sin2"], d["hour_cos2"] = np.sin(2 * a), np.cos(2 * a)
    b = 2 * np.pi * d["day_of_week"] / 7
    d["dow_sin"], d["dow_cos"] = np.sin(b), np.cos(b)
    c = 2 * np.pi * (d["month"] - 1) / 12
    d["month_sin"], d["month_cos"] = np.sin(c), np.cos(c)
    d["wknd_sin"] = d["hour_sin"] * d["is_weekend"]
    d["wknd_cos"] = d["hour_cos"] * d["is_weekend"]
    d["sem_sin"] = d["hour_sin"] * d["is_during_semester"]
    d["sem_cos"] = d["hour_cos"] * d["is_during_semester"]
    return d
