"""Training-only historical-profile baseline with hierarchical fallback."""
import numpy as np
import pandas as pd


class ProfileBaseline:
    """Mean by hour x weekend x semester, then hour x weekend, hour, global."""

    def fit(self, df: pd.DataFrame, y):
        d = df[["hour", "is_weekend", "is_during_semester"]].copy()
        d["y"] = np.asarray(y, dtype=float)
        self.global_ = float(d["y"].mean())
        self.t3_ = d.groupby(["hour", "is_weekend", "is_during_semester"])["y"].mean().to_dict()
        self.t2_ = d.groupby(["hour", "is_weekend"])["y"].mean().to_dict()
        self.t1_ = d.groupby("hour")["y"].mean().to_dict()
        return self

    def predict(self, df: pd.DataFrame):
        return np.array([
            self.t3_.get((h, w, s), self.t2_.get((h, w), self.t1_.get(h, self.global_)))
            for h, w, s in zip(df["hour"], df["is_weekend"], df["is_during_semester"])
        ], dtype=float)
