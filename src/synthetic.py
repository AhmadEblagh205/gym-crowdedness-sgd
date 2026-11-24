"""Synthetic gym-crowdedness data matching the Kaggle dataset's schema.

make_synthetic_legacy(): the original v0.2 generator, kept unchanged so the
legacy 0.36 -> 0.60 result stays reproducible.
make_world(): chronological known-truth generator used by the audit.
"""
import numpy as np
import pandas as pd


def make_synthetic_legacy(n: int = 15000, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hour = rng.integers(6, 23, n)
    day = rng.integers(0, 7, n)
    is_weekend = (day >= 5).astype(int)
    during_sem = rng.choice([0, 1], n, p=[0.3, 0.7])
    temp = rng.normal(60, 12, n)

    peak = 12 * np.exp(-0.5 * ((hour - 18) / 2.6) ** 2) \
        + 6 * np.exp(-0.5 * ((hour - 8) / 1.8) ** 2)
    people = (peak * (1 - 0.35 * is_weekend) * (0.5 + 0.5 * during_sem)
              + 0.05 * (temp - 60) + rng.normal(0, 2.2, n))
    return pd.DataFrame({
        "number_people": np.clip(people, 0, None).round(),
        "day_of_week": day,
        "is_weekend": is_weekend,
        "is_holiday": np.zeros(n, dtype=int),
        "temperature": temp.round(1),
        "is_during_semester": during_sem,
        "hour": hour,
    })


def make_world(seed: int = 0, days: int = 180,
               start: str = "2024-01-15") -> pd.DataFrame:
    """One known-truth world: half-hourly observations 06:00-22:30.

    Latent attendance = morning peak (about 08:00) + larger evening peak
    (about 18:00), scaled down at weekends, scaled by semester and holidays,
    plus a seasonal temperature effect, observation noise, and a mild linear
    drift that the model has no feature for.
    """
    rng = np.random.default_rng(seed)
    slots = np.arange(6 * 60, 23 * 60, 30)            # minutes after midnight
    d, s = np.meshgrid(np.arange(days), slots, indexing="ij")
    d, s = d.ravel(), s.ravel()
    date = (pd.Timestamp(start) + pd.to_timedelta(d, unit="D")
            + pd.to_timedelta(s, unit="min"))
    hour_f = s / 60.0
    dow = np.asarray(date.day_of_week)
    wknd = (dow >= 5).astype(int)
    month = np.asarray(date.month)
    # semester calendar: winter break, term, spring break, term
    sem = (((d >= 20) & (d < 120)) | (d >= 140)).astype(int)
    holiday = np.isin(d, rng.choice(np.arange(days), min(4, days), replace=False)).astype(int)
    doy = np.asarray(date.day_of_year)
    temp_day = 55 + 18 * np.sin(2 * np.pi * (doy - 110) / 365)
    temp = (temp_day + 4 * np.sin(np.pi * (hour_f - 6) / 16)
            + rng.normal(0, 4, days)[d])
    morning = 6.0 * np.exp(-0.5 * ((hour_f - 8) / 1.6) ** 2)
    evening = 12.0 * np.exp(-0.5 * ((hour_f - 18) / 2.4) ** 2)
    scale = (1 - 0.35 * wknd) * (0.5 + 0.5 * sem) * (1 - 0.5 * holiday)
    drift = rng.normal(2.5, 0.5) * d / max(days - 1, 1)   # unmodelled
    people = (scale * (morning + evening) + 0.05 * (temp - 60) + drift
              + rng.normal(0, 2.2, len(d)))
    return pd.DataFrame({
        "number_people": np.clip(people, 0, None).round(),
        "date": date,
        "timestamp": s * 60,
        "day_of_week": dow,
        "is_weekend": wknd,
        "is_holiday": holiday,
        "temperature": temp.round(1),
        "is_during_semester": sem,
        "hour": (s // 60).astype(int),
        "month": month,
    })


def make_synthetic(seed: int = 0, days: int = 180) -> pd.DataFrame:
    """Default checked-in synthetic data: one chronological known-truth world."""
    return make_world(seed=seed, days=days)
