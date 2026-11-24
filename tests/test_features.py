import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from audit import run_audit
from baselines import ProfileBaseline
from data import chrono_split, prepare, validate_schema
from features import FEATURES, FEATURES_FIRST_ONLY, add_features
from synthetic import make_synthetic_legacy, make_world
from train import build_model, fit_eval, recommend_times, rmse


@pytest.fixture(scope="module")
def world():
    return prepare(make_world(seed=3))


def test_cyclic_encodings_unit_circle(world):
    f = add_features(world)
    for a, b in [("hour_sin", "hour_cos"), ("dow_sin", "dow_cos"),
                 ("month_sin", "month_cos")]:
        assert np.allclose(f[a] ** 2 + f[b] ** 2, 1.0)


def test_hour_23_adjacent_to_hour_0():
    rows = pd.DataFrame({"hour": [23, 0, 12], "day_of_week": 2, "month": 3,
                         "is_weekend": 0, "is_during_semester": 1,
                         "temperature": 60.0})
    f = add_features(rows)
    d = lambda i, j: np.hypot(f.hour_sin[i] - f.hour_sin[j],
                              f.hour_cos[i] - f.hour_cos[j])
    assert d(0, 1) < d(0, 2) / 5


def test_day_of_week_wraps():
    rows = pd.DataFrame({"hour": 8, "day_of_week": [6, 0, 3], "month": [12, 1, 6],
                         "is_weekend": 0, "is_during_semester": 1,
                         "temperature": 60.0})
    f = add_features(rows)
    assert abs(f.dow_sin[0] - f.dow_sin[1]) < abs(f.dow_sin[0] - f.dow_sin[2])
    assert abs(f.month_cos[0] - f.month_cos[1]) < 0.9


def test_all_model_features_present(world):
    f = add_features(world)
    assert set(FEATURES) <= set(f.columns)
    assert "day_of_week" not in FEATURES
    for c in ["hour_sin2", "hour_cos2", "wknd_sin", "sem_cos", "temperature"]:
        assert c in FEATURES


def test_duplicate_timestamp_rejected():
    df = make_world(seed=1, days=3)
    df = pd.concat([df, df.iloc[[5]]])
    with pytest.raises(ValueError, match="duplicate"):
        prepare(df)


def test_schema_validation_failures():
    df = make_world(seed=1, days=3)
    with pytest.raises(ValueError, match="missing columns"):
        validate_schema(df.drop(columns=["temperature"]))
    bad = df.copy()
    bad.loc[0, "hour"] = 99
    with pytest.raises(ValueError, match="outside"):
        validate_schema(bad)
    bad = df.copy()
    bad["date"] = bad["date"].astype(str)
    bad.loc[0, "date"] = "not a date"
    with pytest.raises(ValueError):
        prepare(bad)


def test_prepare_sorts_by_date():
    df = make_world(seed=1, days=3).sample(frac=1, random_state=0)
    assert prepare(df)["date"].is_monotonic_increasing


def test_chronological_split_ordering(world):
    tr, te = chrono_split(world)
    assert tr["date"].max() < te["date"].min()
    assert len(tr) == int(len(world) * 0.8)


def test_scaler_inside_pipeline():
    m = build_model("sgd")
    assert isinstance(m, Pipeline)
    assert isinstance(m.steps[0][1], StandardScaler)


def test_profile_baseline_fallbacks():
    tr = pd.DataFrame({"hour": [8, 8, 9], "is_weekend": [0, 1, 0],
                       "is_during_semester": [1, 1, 1], "y": [10., 2., 4.]})
    b = ProfileBaseline().fit(tr, tr["y"])
    q = pd.DataFrame({"hour": [8, 8, 9, 15], "is_weekend": [0, 0, 1, 0],
                      "is_during_semester": [1, 0, 1, 1]})
    p = b.predict(q)
    assert p[0] == 10.0                          # full combination seen
    assert p[1] == 10.0                          # semester unseen -> hour x weekend
    assert p[2] == 4.0                           # weekend/9 unseen -> hour mean
    assert p[3] == pytest.approx(16 / 3)         # hour unseen -> global mean


def test_second_harmonic_improves_future_r2(world):
    tr, te = chrono_split(world)
    ftr, fte = add_features(tr), add_features(te)
    full = build_model("sgd").fit(ftr[FEATURES], tr["number_people"])
    first = build_model("sgd").fit(ftr[FEATURES_FIRST_ONLY], tr["number_people"])
    y = te["number_people"]
    assert (r2_score(y, full.predict(fte[FEATURES]))
            > r2_score(y, first.predict(fte[FEATURES_FIRST_ONLY])) + 0.1)


def test_full_model_beats_profile_baseline(world):
    m, _, _ = fit_eval(world)
    assert m["full_sgd"]["rmse"] < m["profile"]["rmse"]
    assert m["train_end"] < m["test_start"]


def _candidates():
    return pd.DataFrame({"hour": [18, 14, 11], "day_of_week": 1, "month": 3,
                         "is_weekend": 0, "is_during_semester": 1,
                         "temperature": 60.0}, index=["a", "b", "c"])


def test_recommender_ranks_supplied_rows_only(world):
    _, models, _ = fit_eval(world)
    out = recommend_times(models["sgd"], _candidates(), n=2)
    assert len(out) == 2
    assert {k for k, _ in out} <= {"a", "b", "c"}
    assert out[0][1] <= out[1][1]
    assert out[0][0] != "a"                      # evening rush is not the quietest


def test_recommender_raises_on_missing_columns(world):
    _, models, _ = fit_eval(world)
    with pytest.raises(ValueError, match="missing"):
        recommend_times(models["sgd"], _candidates().drop(columns=["temperature"]))


def test_generator_reproducible_by_seed():
    a, b, c = make_world(5, 10), make_world(5, 10), make_world(6, 10)
    pd.testing.assert_frame_equal(a, b)
    assert not a["number_people"].equals(c["number_people"])


def test_world_schema_and_observation_times():
    w = make_world(2, 5)
    assert {"number_people", "date", "timestamp", "day_of_week", "is_weekend",
            "is_holiday", "temperature", "is_during_semester", "hour",
            "month"} <= set(w.columns)
    assert w["date"].dt.hour.min() == 6
    assert (w["date"].dt.hour * 60 + w["date"].dt.minute).max() == 22 * 60 + 30


def test_drift_present():
    # average over worlds: late-window people exceed early-window people
    # beyond what the (balanced) calendar alone explains; check drift directly
    diffs = []
    for s in range(6):
        w = make_world(s, 180)
        w = w[(w.is_weekend == 0) & (w.is_during_semester == 1)
              & (w.hour.between(17, 19))]
        diffs.append(w.groupby(w.date.dt.dayofyear < w.date.dt.dayofyear.median())
                     ["number_people"].mean().diff().iloc[-1])
    assert np.mean(diffs) < 0                    # first-half (True) minus second-half


def test_morning_and_evening_peaks():
    w = make_world(4, 60)
    p = w[(w.is_weekend == 0) & (w.is_during_semester == 1)].groupby("hour")["number_people"].mean()
    assert p[8] > p[13] and p[18] > p[13] and p[18] > p[8]


def test_legacy_generator_unchanged_shape():
    df = make_synthetic_legacy()
    assert df.shape == (15000, 7)
    assert "date" not in df.columns
    assert df["hour"].between(6, 22).all()


def test_sgd_and_ridge_use_identical_columns(world):
    _, models, _ = fit_eval(world)
    assert (models["sgd"].n_features_in_ == models["ridge"].n_features_in_
            == len(FEATURES))
    tr, te = chrono_split(world)
    f = add_features(te)[FEATURES]
    assert rmse(te["number_people"], models["ridge"].predict(f)) < 5


def test_audit_smoke_three_worlds():
    s = run_audit(3, write=False)
    assert s["n_worlds"] == 3
    assert s["second_harmonic"]["fraction_improved"] == 1.0
    assert s["legacy_reproduction"]["plus_second_harmonic"] > \
        s["legacy_reproduction"]["first_harmonic"]
