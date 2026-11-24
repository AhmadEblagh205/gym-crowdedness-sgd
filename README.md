# Gym Crowdedness Forecasting & Temporal Validation

[![CI](https://github.com/AhmadEblagh205/gym-crowdedness-sgd/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/gym-crowdedness-sgd/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Linear forecasting of campus-gym crowdedness with cyclic calendar features, a chronological holdout, a serious baseline and known-truth stress tests. The repository name still says SGD, but the audit below shows that the optimiser is not what drives the result.

## Provenance

I first built this as a Dataquest guided project. This repository is my later independent rebuild and audit of that exercise; the guided origin is unchanged. The real data is the Kaggle dataset "Crowdedness at the Campus Gym" (`nsrose7224/crowdedness-at-the-campus-gym`). I do not vendor it: save it as `data/raw/gym_data.csv` (git-ignored) to run on it. Everything reported below comes from my own synthetic generator, not from that file.

## What changed in the audit (v0.3)

- Random row splitting is replaced by a chronological holdout: oldest 80% of rows train, newest 20% test.
- Dates are parsed and sorted, duplicate timestamps are rejected, and model-facing columns are range-checked.
- Standardisation lives inside the sklearn pipeline, so scaler parameters come from training rows only.
- Raw numeric `day_of_week` is replaced by cyclic sine/cosine; I added cyclic month encoding.
- I kept the first and second hourly Fourier harmonics and added first-harmonic interactions with the weekend and in-semester flags.
- The weak global-mean benchmark is replaced by a training-only historical-profile baseline (hour x weekend x semester, falling back to hour x weekend, then hour, then the global training mean).
- Ridge is fitted on exactly the same feature basis as SGD, to separate feature value from optimiser value.
- The synthetic generator is now a chronological known-truth process (two daily peaks, weekend and semester effects, weather, noise, mild unmodelled drift). The original generator is kept as `make_synthetic_legacy()`.
- `recommend_times` now ranks complete candidate rows that the caller supplies, instead of inventing 24 hours of context from one base row.
- `src/audit.py` runs 100 independent worlds; `docs/METHODOLOGY.md` documents the design; the test suite grew from 3 to 21 tests.

## The legacy 0.36 -> 0.60 result, and why I demoted it

With the original generator and the original random split I reproduced my old claim exactly (`reports/audit_summary.json`, key `legacy_reproduction`):

| Feature set | R2 |
|---|---|
| first hourly harmonic | 0.3595 |
| + second hourly harmonic | 0.5965 |
| + weekend interaction | 0.6036 |

The number was not wrong, but it was one seed, one random split and one synthetic world. That is interpolation evidence, not forecasting evidence, so I no longer treat it as a headline.

## 100-world chronological known-truth audit

Each of 100 independently seeded worlds has 180 days, observations every 30 minutes from 06:00 to 22:30, separate morning and evening peaks, calendar and weather effects, noise and a mild linear drift the model cannot see. Oldest 80% trains; newest 20% is the future holdout. Figures are the output of my run (`reports/audit_summary.json`, per-world rows in `reports/audit_per_world.csv`).

| Question | Result |
|---|---|
| Does the second hourly harmonic help on the future holdout? | Improved R2 in 100/100 worlds; median gain +0.3634 (5th-95th percentile +0.3354 to +0.3867) |
| Full-model future R2 / RMSE | Median R2 0.6312 (0.5968-0.6568); median RMSE 2.4200 (2.3204-2.5420) |
| Full model vs training-only profile baseline | Full model has lower RMSE in 100/100 worlds; median RMSE reduction 25.23% (18.70-32.21%); profile median RMSE 3.2517 |
| Random row split vs chronological holdout (same model) | Random split gave higher R2 in 94/100 worlds; median random-minus-future R2 +0.0425 (-0.0025 to +0.0811) |
| SGD vs Ridge, identical features | Median SGD RMSE minus Ridge RMSE +0.0034 (-0.0128 to +0.0243) |

Interpretation: under this controlled drift mechanism a random split was usually optimistic for the real question, which is predicting the future. The size of the gap depends entirely on the drift I put in, so it is not a correction factor for real data. SGD and Ridge are practically indistinguishable, so I do not claim SGD drives the result; the feature representation and the validation design are the substance. The profile baseline is weaker here partly because it cannot use temperature, and because my generator was written with a smooth parametric shape that a Fourier basis fits well, so the margin over the profile is a property of the generator as much as of the method.

The default checked-in run (`python src/train.py --synthetic`, world seed 0) gives, on its future holdout, full-model RMSE 2.4306 and R2 0.6284, Ridge RMSE 2.4289, profile RMSE 3.2095 (a 24.27% RMSE reduction). These are illustrative synthetic outputs, not a real-gym score.

## Usage

```bash
pip install -r requirements.txt
python src/train.py --synthetic          # writes reports/metrics.json and two figures
python src/audit.py                      # 100-world audit, a few seconds
# real data: save the Kaggle file as data/raw/gym_data.csv, then
python src/train.py --data data/raw/gym_data.csv
```

`recommend_times(model, candidate_rows, n)` in `src/train.py` takes a DataFrame of complete candidate rows (hour, day_of_week, month, is_weekend, is_during_semester, temperature) and returns the n quietest with their predictions. It raises `ValueError` if columns are missing.

## Tests

```bash
pytest -q
```

21 offline tests, a few seconds in total: cyclic encodings, feature list, duplicate-timestamp and schema rejection, chronological ordering, scaler placement, baseline fallbacks, second-harmonic and baseline checks on one world, recommender contract, generator reproducibility and drift, legacy generator shape, SGD/Ridge column parity and a 3-world audit smoke test. CI runs them on push.

## Licence

MIT, see [LICENSE](LICENSE).
