# Changelog

## 0.3.0 — defensibility and temporal-validation audit
I rebuilt the project around chronological validation. Changes: oldest-80%/newest-20% holdout instead of a random split; schema/range validation and duplicate-timestamp rejection; cyclic day-of-week and month features, first and second hourly harmonics and weekend/semester interactions; StandardScaler inside the pipeline; a training-only hour x weekend x semester profile baseline with fallbacks; Ridge on the identical feature basis; a chronological known-truth synthetic generator (the old one is kept as make_synthetic_legacy); a recommend_times that ranks caller-supplied complete rows; src/audit.py with 100 worlds plus a reproduction of the legacy 0.36 -> 0.60 result; docs/METHODOLOGY.md and machine-readable reports; tests grown from 3 to 21. Synthetic results are not real-gym accuracy.

## 0.2.0, update round 4
Added hour by weekend interaction features (a weekend specific amplitude for the first harmonic) and a recommend_times helper that ranks the quietest hours from the fitted model, the when should I go answer.

## 0.1.0
Initial release.
