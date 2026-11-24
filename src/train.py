"""Gym crowdedness forecasting with chronological validation.

Usage:
    python src/train.py --synthetic
    python src/train.py --data data/raw/gym_data.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import ProfileBaseline
from data import chrono_split, prepare
from features import FEATURES, RAW_NEEDED, add_features
from synthetic import make_synthetic

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "figures"


def build_model(kind: str = "sgd", max_iter: int = 300):
    """Scaler lives inside the pipeline so it is fitted on training rows only."""
    reg = (SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-4,
                        max_iter=max_iter, tol=1e-4, random_state=0)
           if kind == "sgd" else Ridge(alpha=1.0))
    return make_pipeline(StandardScaler(), reg)


def rmse(y, p) -> float:
    return float(np.sqrt(mean_squared_error(y, p)))


def fit_eval(df: pd.DataFrame, max_iter: int = 300):
    """Chronological holdout; returns (metrics, fitted models, test frame)."""
    tr, te = chrono_split(prepare(df))
    ftr, fte = add_features(tr), add_features(te)
    ytr, yte = tr["number_people"], te["number_people"]
    sgd = build_model("sgd", max_iter).fit(ftr[FEATURES], ytr)
    ridge = build_model("ridge").fit(ftr[FEATURES], ytr)
    prof = ProfileBaseline().fit(tr, ytr)
    preds = {"full_sgd": sgd.predict(fte[FEATURES]),
             "ridge": ridge.predict(fte[FEATURES]), "profile": prof.predict(te)}
    m = {k: {"rmse": round(rmse(yte, p), 4), "r2": round(float(r2_score(yte, p)), 4)}
         for k, p in preds.items()}
    m["rmse_reduction_vs_profile_pct"] = round(
        100 * (1 - m["full_sgd"]["rmse"] / m["profile"]["rmse"]), 2)
    m["n_train"], m["n_test"] = len(tr), len(te)
    m["train_end"], m["test_start"] = str(tr["date"].max()), str(te["date"].min())
    te = te.assign(pred=preds["full_sgd"])
    return m, {"sgd": sgd, "ridge": ridge, "profile": prof}, te


def recommend_times(model, candidate_rows: pd.DataFrame, n: int = 3):
    """Rank complete candidate rows supplied by the caller, quietest first.

    No rows are invented: every candidate must carry all raw columns.
    Returns [(row_label, predicted_people), ...] for the n quietest.
    """
    missing = [c for c in RAW_NEEDED if c not in candidate_rows.columns]
    if missing:
        raise ValueError(f"candidate rows missing columns: {missing}")
    pred = model.predict(add_features(candidate_rows)[FEATURES])
    order = np.argsort(pred, kind="stable")[:n]
    return [(candidate_rows.index[i], float(pred[i])) for i in order]


def make_figures(te: pd.DataFrame, m: dict) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(te["number_people"], te["pred"], s=8, alpha=0.3)
    lim = [0, max(te["number_people"].max(), te["pred"].max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="perfect prediction")
    ax.set(xlabel="Actual number of people", ylabel="Predicted",
           title=f"Future holdout (RMSE {m['full_sgd']['rmse']:.2f}, "
                 f"R2 {m['full_sgd']['r2']:.2f})")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / "actual_vs_predicted.png", dpi=150)
    prof = te.groupby("hour")[["number_people", "pred"]].mean()
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.plot(prof.index, prof["number_people"], "o-", label="actual (mean)")
    ax2.plot(prof.index, prof["pred"], "s--", label="predicted (mean)")
    ax2.set(xlabel="Hour of day", ylabel="Number of people",
            title="Average crowdedness through the day (future holdout)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(FIGURES / "hourly_profile.png", dpi=150)
    plt.close("all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data" / "raw" / "gym_data.csv")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    df = make_synthetic() if args.synthetic else pd.read_csv(args.data)
    m, _, te = fit_eval(df)
    m["source"] = ("synthetic known-truth world (seed 0)" if args.synthetic
                   else str(args.data))
    make_figures(te, m)
    out = ROOT / "reports" / "metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
