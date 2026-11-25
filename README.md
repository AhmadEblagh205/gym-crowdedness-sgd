# Gym Crowdedness Forecasting & Temporal Validation

[![CI](https://github.com/AhmadEblagh205/gym-crowdedness-sgd/actions/workflows/ci.yml/badge.svg)](https://github.com/AhmadEblagh205/gym-crowdedness-sgd/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

I first built this as a Dataquest guided project. This repository is my later independent rebuild and audit of that exercise. It forecasts campus-gym crowdedness with a linear model, cyclic calendar features and a chronological holdout. The real data is the Kaggle dataset "Crowdedness at the Campus Gym" (`nsrose7224/crowdedness-at-the-campus-gym`), which I do not include. Everything reported below comes from my own synthetic generator, not from that file.

## What it does

- Encodes hour, day of week and month as cyclic features, with two hourly Fourier harmonics and weekend and semester interactions.
- Trains an SGD regressor and a Ridge model on the same features, and compares both with a historical-profile baseline built from training rows only.
- Validates on the newest 20% of rows, with the oldest 80% used for training.
- Generates synthetic worlds with a known pattern, and runs a 100-world audit.
- Provides `recommend_times(model, candidate_rows, n)` in `src/train.py`, which returns the n quietest of the candidate rows you supply.

The repository name still says SGD, but the audit shows the optimiser makes almost no difference.

## Results

All figures come from synthetic data. Each of the 100 worlds has 180 days with observations every 30 minutes and a mild drift the model cannot see. The full output is in `reports/audit_summary.json` and `reports/audit_per_world.csv`.

| Question | Result |
|---|---|
| Does the second hourly harmonic help on the future holdout? | R2 improved in 100/100 worlds, median gain +0.3634 |
| Full model on the future holdout | Median R2 0.6312, median RMSE 2.4200 |
| Full model against the profile baseline | Lower RMSE in 100/100 worlds, median reduction 25.23% |
| Random row split against chronological holdout | Random split gave higher R2 in 94/100 worlds, median difference +0.0425 |
| SGD against Ridge, same features | Median RMSE difference +0.0034 |

The default run (`python src/train.py --synthetic`, seed 0) gave RMSE 2.4306 and R2 0.6284 for the full model, 2.4289 for Ridge and 3.2095 for the profile baseline. These are synthetic outputs, not a real-gym score.

With my original generator and a random split I had earlier reached R2 0.36 with one harmonic and 0.60 with two. I no longer treat that as a headline, because it came from one seed and one split.

## How to run

```bash
pip install -r requirements.txt
python src/train.py --synthetic          # writes reports/metrics.json and two figures
python src/audit.py                      # 100-world audit, a few seconds
# real data: save the Kaggle file as data/raw/gym_data.csv (git-ignored), then
python src/train.py --data data/raw/gym_data.csv
pytest -q
```

There are 21 offline tests. They cover the cyclic encodings, schema and duplicate-timestamp rejection, chronological ordering, scaler placement, baseline fallbacks, the recommender, generator reproducibility, SGD and Ridge column parity and a 3-world audit smoke test. CI runs them on push.

## Notes

The gap between the random split and the chronological holdout depends on the drift I put into the generator, so it does not carry over to real data. Version 0.3 is my audit of the earlier rebuild. The list of changes, the earlier result and the design of the audit are in [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Licence

MIT, see [LICENSE](LICENSE).
