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

## Reward / penalty learning

The model can **learn from outcomes**:

| Result | Effect |
|--------|--------|
| Correct CE/PE call | **Reward** (confidence × move size) |
| Wrong CE/PE call | **Penalty** (same formula, negative) |

Learning updates:
1. **Online learner** (`models/online_learner.joblib`) — `partial_fit` on each settlement; wrong calls get a harder update
2. **Batch model** — retrained with higher sample weight on penalised days, mild boost on rewarded days
3. **Blended prediction** — live `predict.py` mixes batch + online probabilities

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/train.py                 # initial batch model
python scripts/learn.py --bootstrap 60  # warm-start rewards from recent history
python scripts/predict.py               # predict + log for tomorrow
python scripts/snapshot_oc.py           # store today's OC features

# Next day (after market close):
python scripts/learn.py                 # settle yesterday → reward/penalty → update models
python scripts/predict.py               # next signal using updated models
```

## Honest performance expectations

Next-day index direction is extremely noisy. On ~5y NIFTY history, walk-forward CV is typically near **50–53%** accuracy. That is expected.

Practical use:
- treat output as a **weak signal**, not a trade order
- prefer trades only when `confidence` is elevated (e.g. ≥ 0.55–0.60)
- manage premium risk (IV crush / theta can hurt even if direction is right)
- run `learn.py` daily so rewards/penalties accumulate
- accumulate OC snapshots so PCR / OI / IV features improve over time

## Data used

1. **NIFTY spot** daily OHLCV via Yahoo (`^NSEI`)
2. **Live NSE option chain** via `option-chain-v3` (nearest expiry by default)
3. **Stored OC features** in `data/processed/oc_features.csv`
4. **Prediction log** in `data/processed/prediction_log.csv` (rewards/penalties)

## Layout

```
src/            # fetchers, features, model, learning loop
scripts/        # train / predict / learn / snapshot_oc
models/         # batch model + online learner + metrics
data/processed/ # OC features, prediction log, learning stats
data/raw/       # ignored downloads / full OC CSVs
```

## Suggested daily workflow

1. After close: `python scripts/snapshot_oc.py`
2. After close: `python scripts/learn.py`  (settle prior prediction, reward/penalise, retrain)
3. After learn: `python scripts/predict.py` (log tonight's call for tomorrow)
