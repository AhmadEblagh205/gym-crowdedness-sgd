"""Schema validation, date parsing/sorting and the chronological split."""
import pandas as pd

# column -> (min, max) for the model-facing columns
RANGES = {
    "number_people": (0, 500), "day_of_week": (0, 6), "is_weekend": (0, 1),
    "is_holiday": (0, 1), "temperature": (-40, 130),
    "is_during_semester": (0, 1), "hour": (0, 23), "month": (1, 12),
}
REQUIRED = ["date"] + list(RANGES)


def validate_schema(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    for col, (lo, hi) in RANGES.items():
        v = pd.to_numeric(df[col], errors="coerce")
        if v.isna().any():
            raise ValueError(f"column {col!r} has missing or non-numeric values")
        if (v < lo).any() or (v > hi).any():
            raise ValueError(f"column {col!r} outside [{lo}, {hi}]")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, parse dates, reject duplicate timestamps, sort by time."""
    validate_schema(df)
    out = df.copy()
    try:
        out["date"] = pd.to_datetime(out["date"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"unparseable date column: {exc}") from exc
    if out["date"].duplicated().any():
        raise ValueError("duplicate timestamps in date column")
    return out.sort_values("date").reset_index(drop=True)


def chrono_split(df: pd.DataFrame, train_frac: float = 0.8):
    """Oldest train_frac of rows train, newest rest test."""
    df = df.sort_values("date")
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
