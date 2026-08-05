# Trading_fno — NIFTY next-day option model

Predict **next trading day's NIFTY direction** and map it to an **ATM CE vs PE** suggestion.

This is a **daily / EOD strategy model**, not tick-by-tick prediction.

## Prediction target

| Output | Meaning |
|--------|---------|
| `predicted_side=CE` | Expect NIFTY up next day → lean ATM/near-ATM Call |
| `predicted_side=PE` | Expect NIFTY down next day → lean ATM/near-ATM Put |
| `prob_up` / `confidence` | Calibrated probabilities for filters / sizing |

Label used in training: `next_up = 1` if tomorrow's close > today's close.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/train.py          # time-series CV + save model
python scripts/predict.py        # latest spot + live NSE OC → CE/PE
python scripts/snapshot_oc.py    # store today's OC features
```

## Honest performance expectations

Next-day index direction is extremely noisy. On ~5y NIFTY history, walk-forward CV is typically near **50–53%** accuracy. That is expected.

Practical use:
- treat output as a **weak signal**, not a trade order
- prefer trades only when `confidence` is elevated (e.g. ≥ 0.55–0.60)
- manage premium risk (IV crush / theta can hurt even if direction is right)
- accumulate daily OC snapshots (`snapshot_oc.py`) so PCR / OI / IV features can improve the model over time

## Data used

1. **NIFTY spot** daily OHLCV via Yahoo (`^NSEI`)
2. **Live NSE option chain** via `option-chain-v3` (nearest expiry by default)
3. **Stored OC features** in `data/processed/oc_features.csv` after each snapshot

Until you have many OC history rows, training is mostly technical/spot features.

## Layout

```
src/            # fetchers, features, model
scripts/        # train / predict / snapshot_oc
models/         # trained artifact + metrics.json
data/processed/ # OC feature history
data/raw/       # ignored downloads / full OC CSVs
```

## Suggested daily workflow

1. After close: `python scripts/snapshot_oc.py`
2. Weekly/monthly: `python scripts/train.py`
3. Before next session: `python scripts/predict.py`
