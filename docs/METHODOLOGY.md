# Methodology

## Estimands

I ask two different questions and keep them apart.

1. Future-period forecasting: given calendar and weather for a period after the training window, how well can I predict the number of people? This is what the chronological holdout estimates, and it is the question that matters for a gym.
2. Within-period interpolation: how well can I predict a randomly withheld row from the same period? A random split estimates this, and I report it only as a comparison.

Metrics are RMSE and R2 on the held-out rows. R2 depends on the variance of the test window, so I compare R2 only across the same window and never across datasets.

## Feature design

The model is linear, so the features carry the structure.

- Hour is cyclic: 23:00 and 00:00 are neighbours. I use the first and second Fourier harmonics (sin/cos of 2*pi*h/24 and 4*pi*h/24). One harmonic gives a single daily bump; the second lets a linear model fit separate morning and evening peaks. Where `date` carries minutes, the fractional hour is used.
- Day of week is encoded as sin/cos of 2*pi*d/7 rather than a raw 0-6 number, and month as sin/cos of 2*pi*(m-1)/12.
- First-harmonic interactions: hour sin/cos multiplied by `is_weekend`, and by `is_during_semester`, so weekends and term time can have their own daily amplitude and phase.
- `is_weekend`, `is_during_semester` and `temperature` enter directly.
- A `StandardScaler` sits inside the sklearn Pipeline, so scaling parameters are learned from training rows only.

SGDRegressor (squared loss, L2, alpha 1e-4, modest max_iter) and Ridge (alpha 1) are fitted on the identical feature matrix.

## Chronological validation geometry

The data are parsed on `date`, checked for duplicates (a duplicate timestamp raises `ValueError`), and sorted. The oldest 80% of rows form the training set and the newest 20% the test set, so every training timestamp precedes every test timestamp. The profile baseline is also fitted on the training rows only. The audit's random-split comparison uses a random 80/20 split of the same world with the same model and features.

## Baseline definition

The historical-profile baseline predicts the training mean for the row's (hour, is_weekend, is_during_semester) cell. If that cell is unseen in training it falls back to (hour, is_weekend), then to hour, then to the global training mean. It uses no test information. It cannot use temperature, month or drift, which is part of why a regression can beat it.

## Known-truth generator

`make_world(seed, days=180)` produces half-hourly observations from 06:00 to 22:30. Expected attendance is a morning peak (about 08:00, height 6) plus a larger evening peak (about 18:00, height 12), multiplied by a weekend factor (0.65), a semester factor (0.5 out of term, 1 in term) and a holiday factor, plus 0.05 per degree of temperature relative to 60, where temperature has a seasonal component and daily weather noise. A linear drift (about +2.5 people over the window, varying by world) is added, and the model has no feature for it. Gaussian observation noise (sd 2.2) is added, and counts are rounded and clipped at zero. The semester calendar is winter break, term, spring break, term, so both states occur in training and the test window falls in term. Columns follow the Kaggle schema so the real file passes the same code path. The original v0.2 generator is preserved as `make_synthetic_legacy()` so my old result is reproducible.

Because I wrote the generator and the model features from the same mental picture, the audit tests whether the pipeline recovers known structure and whether validation choices matter. It cannot tell me how the model behaves on real attendance.

## The audit

`src/audit.py` generates 100 worlds (seeds 1000-1099) and records, per world: the R2 gain from the second harmonic on the future holdout; the full model's RMSE reduction against the profile baseline; the random-split minus chronological R2; and SGD minus Ridge RMSE. It also reproduces the legacy result with the old generator, the old 25% random split (random_state 42) and the old SGD settings. Results are in `reports/audit_summary.json` and `reports/audit_per_world.csv`. Seeds are fixed, so the audit is deterministic.

## Claim boundaries

- Synthetic evidence is not real-gym accuracy. I have not claimed any score on the Kaggle data.
- The audit does not show that SGD drives the result; SGD and Ridge are close on identical features.
- The random-versus-chronological gap is specific to my drift mechanism and is not a universal correction.
- The recommendation helper only ranks rows the caller supplies. I have not validated recommendation quality or any effect on user behaviour.
- There is no production forecasting service.

## Notes moved from the README

Changes made in version 0.3 (my audit of the earlier rebuild):

- Random row splitting is replaced by a chronological holdout: the oldest 80% of rows train and the newest 20% test.
- Dates are parsed and sorted, duplicate timestamps are rejected and model-facing columns are range-checked.
- Standardisation sits inside the sklearn pipeline, so scaler parameters come from training rows only.
- Raw numeric `day_of_week` is replaced by cyclic sine and cosine terms, and I added cyclic month encoding.
- I kept the first and second hourly Fourier harmonics and added first-harmonic interactions with the weekend and in-semester flags.
- The global-mean benchmark is replaced by a training-only historical-profile baseline (hour x weekend x semester, falling back to hour x weekend, then hour, then the global training mean).
- Ridge is fitted on the same feature basis as SGD, to separate feature value from optimiser value.
- The synthetic generator is now a chronological known-truth process (two daily peaks, weekend and semester effects, weather, noise and mild unmodelled drift). The original generator is kept as `make_synthetic_legacy()`.
- `recommend_times` ranks complete candidate rows supplied by the caller instead of inventing 24 hours of context from one base row.
- The test suite grew from 3 to 21 tests.

Legacy result: with the original generator and the original random split I reproduced my earlier claim (`reports/audit_summary.json`, key `legacy_reproduction`). R2 was 0.3595 with the first hourly harmonic, 0.5965 with the second added and 0.6036 with a weekend interaction. It was one seed, one random split and one synthetic world, so it is interpolation evidence and not a forecasting result, and I no longer treat it as a headline.

Interpretation of the 100-world audit: the random split was usually optimistic under the drift I built in, and the size of the gap depends entirely on that drift, so it is not a correction factor for real data. The profile baseline is weaker partly because it cannot use temperature, and partly because the generator has a smooth parametric shape that a Fourier basis fits well, so the margin over the profile is a property of the generator as much as of the method.
