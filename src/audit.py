"""100 independent known-truth worlds: validation and feature audit.

Usage: python src/audit.py [--worlds 100]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import ProfileBaseline
from data import chrono_split, prepare
from features import FEATURES, FEATURES_FIRST_ONLY, add_features
from synthetic import make_synthetic_legacy, make_world
from train import build_model, rmse

ROOT = Path(__file__).resolve().parents[1]
ITER = 300


def run_world(seed: int) -> dict:
    df = prepare(make_world(seed))
    tr, te = chrono_split(df)
    ftr, fte = add_features(tr), add_features(te)
    ytr, yte = tr["number_people"], te["number_people"]
    full = build_model("sgd", ITER).fit(ftr[FEATURES], ytr)
    first = build_model("sgd", ITER).fit(ftr[FEATURES_FIRST_ONLY], ytr)
    ridge = build_model("ridge").fit(ftr[FEATURES], ytr)
    prof = ProfileBaseline().fit(tr, ytr)
    p_full = full.predict(fte[FEATURES])
    r2_full = r2_score(yte, p_full)
    r2_first = r2_score(yte, first.predict(fte[FEATURES_FIRST_ONLY]))
    rm_full, rm_prof = rmse(yte, p_full), rmse(yte, prof.predict(te))
    rm_ridge = rmse(yte, ridge.predict(fte[FEATURES]))
    # random row split on the same world (same model, same features)
    a, b = train_test_split(add_features(df), test_size=0.2, random_state=seed)
    rnd = build_model("sgd", ITER).fit(a[FEATURES], a["number_people"])
    r2_rand = r2_score(b["number_people"], rnd.predict(b[FEATURES]))
    return {"seed": seed, "r2_first_harmonic": r2_first, "r2_full": r2_full,
            "r2_gain_second_harmonic": r2_full - r2_first,
            "rmse_full": rm_full, "rmse_profile": rm_prof,
            "rmse_reduction_pct": 100 * (1 - rm_full / rm_prof),
            "rmse_ridge": rm_ridge, "sgd_minus_ridge_rmse": rm_full - rm_ridge,
            "r2_random_split": r2_rand,
            "random_minus_future_r2": r2_rand - r2_full}


def dist(s) -> dict:
    q = np.percentile(s, [5, 50, 95])
    return {"median": round(float(q[1]), 4), "p05": round(float(q[0]), 4),
            "p95": round(float(q[2]), 4)}


def summarise(rows: pd.DataFrame) -> dict:
    imp = rows.r2_gain_second_harmonic > 0
    win = rows.rmse_full < rows.rmse_profile
    rnd = rows.random_minus_future_r2 > 0
    return {
        "n_worlds": len(rows),
        "second_harmonic": {"fraction_improved": float(imp.mean()),
                            "n_improved": int(imp.sum()),
                            "r2_gain": dist(rows.r2_gain_second_harmonic)},
        "full_model_future": {"r2": dist(rows.r2_full),
                              "rmse": dist(rows.rmse_full)},
        "profile_baseline": {"rmse": dist(rows.rmse_profile),
                             "fraction_full_model_wins": float(win.mean()),
                             "n_wins": int(win.sum()),
                             "rmse_reduction_pct": dist(rows.rmse_reduction_pct)},
        "random_vs_chronological": {
            "fraction_random_looks_better": float(rnd.mean()),
            "n_random_better": int(rnd.sum()),
            "random_minus_future_r2": dist(rows.random_minus_future_r2)},
        "sgd_vs_ridge": {"sgd_minus_ridge_rmse": dist(rows.sgd_minus_ridge_rmse)},
    }


def legacy_reproduction() -> dict:
    """Original v0.2 generator + original random split + original SGD settings."""
    df = make_synthetic_legacy()
    a = 2 * np.pi * df["hour"] / 24
    df["hour_sin"], df["hour_cos"] = np.sin(a), np.cos(a)
    df["hour_sin2"], df["hour_cos2"] = np.sin(2 * a), np.cos(2 * a)
    df["wknd_sin"] = df["hour_sin"] * df["is_weekend"]
    df["wknd_cos"] = df["hour_cos"] * df["is_weekend"]
    base = ["hour_sin", "hour_cos", "day_of_week", "is_weekend",
            "temperature", "is_during_semester"]
    sets = {"first_harmonic": base,
            "plus_second_harmonic": base + ["hour_sin2", "hour_cos2"],
            "plus_weekend_interaction": base + ["hour_sin2", "hour_cos2",
                                                "wknd_sin", "wknd_cos"]}
    out = {}
    for name, cols in sets.items():
        X_tr, X_te, y_tr, y_te = train_test_split(
            df[cols], df["number_people"], test_size=0.25, random_state=42)
        m = make_pipeline(StandardScaler(), SGDRegressor(
            loss="squared_error", penalty="l2", alpha=1e-4, max_iter=2000,
            random_state=0)).fit(X_tr, y_tr)
        out[name] = round(float(r2_score(y_te, m.predict(X_te))), 4)
    return out


def run_audit(n_worlds: int = 100, write: bool = True) -> dict:
    t0 = time.time()
    rows = pd.DataFrame([run_world(s) for s in range(1000, 1000 + n_worlds)])
    summary = summarise(rows)
    summary["legacy_reproduction"] = legacy_reproduction()
    summary["design"] = {"days_per_world": 180, "train_frac": 0.8,
                         "seeds": f"1000..{999 + n_worlds}", "sgd_max_iter": ITER,
                         "runtime_s": round(time.time() - t0, 1)}
    if write:
        out = ROOT / "reports"
        out.mkdir(exist_ok=True)
        (out / "audit_summary.json").write_text(json.dumps(summary, indent=2))
        rows.round(5).to_csv(out / "audit_per_world.csv", index=False)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=100)
    print(json.dumps(run_audit(ap.parse_args().worlds), indent=2))
