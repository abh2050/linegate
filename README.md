# linegate

Predict quality failures on the Bosch production line, price each decision in
dollars, and route uncertain parts to a human. See `CLAUDE.md` for the
operating contract.

## Getting the data (Gate 0)

1. Put Kaggle credentials in `~/.kaggle/kaggle.json` and accept the rules of
   the `bosch-production-line-performance` competition.
2. `make setup`
3. `make data` downloads the six competition zips into `data/raw/`.
4. `make manifest` records their sha256 in `data/manifest.json`. Commit it.
5. `make gate-0`
