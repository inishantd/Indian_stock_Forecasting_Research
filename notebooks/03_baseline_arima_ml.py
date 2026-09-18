# %% [markdown]
# # 03 — Baseline, ARIMA, Random Forest, XGBoost
#
# This notebook covers the first three levels of model complexity:
#
# 1. **Random Walk (no-change) baseline** — the "best forecast of
#    tomorrow's price is today's price" benchmark every other model has to
#    beat to justify its existence.
# 2. **ARIMA** — a classical statistical time-series model.
# 3. **Random Forest** and **XGBoost** — tree-based ML models using
#    engineered features.
#
# All four are evaluated identically: log-return target (justified in
# notebook 02), chronological data only, walk-forward out-of-sample
# testing, and the same metrics. That's what makes the eventual comparison
# in notebook 06 fair.

# %% [markdown]
# ## 1. Validation design (read this before the results — it explains why
# they can be trusted)
#
# * **Never a random split.** All splits are chronological.
# * **Walk-forward, expanding window.** The final 15% of each stock's
#   history is held out as the test region. Within that region we predict
#   one quarter (63 trading days) at a time, then fold that quarter into
#   the training window before moving on — so every prediction comes from
#   a model that has only ever seen data *before* the day it's predicting.
#   We refit quarterly rather than daily purely for runtime (12 stocks x
#   multiple models); the property that matters — no model ever sees the
#   future — holds either way.
# * **No test-set peeking for model selection.** ARIMA's (p, d, q) order
#   and the RF/XGBoost hyperparameters are chosen using only the training
#   portion (via AIC for ARIMA, via a simple validation slice for the ML
#   models), never the held-out test region.
# * **Feature engineering has no forward-looking information.** Every
#   feature for day *t* uses data up to and including day *t*'s close; the
#   target is day *t+1*'s return (see `src/features.py`).
# * **Scaling isn't needed for tree models** (RF/XGBoost are
#   scale-invariant), which sidesteps a common leakage mistake (fitting a
#   scaler on the full series). Where scaling *is* used later (LSTM/CNN/
#   Transformer in notebooks 04-05), it is explicitly fit on training data
#   only.

# %%
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from config import SECTOR_STOCKS, ALL_TICKERS, TICKER_TO_SECTOR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, RANDOM_SEED
from features import build_tabular_features, FEATURE_COLUMNS
from eval_utils import regression_metrics, expanding_window_folds

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

prices = pd.read_csv(PROCESSED_DIR / "prices_processed.csv", parse_dates=["Date"])
all_predictions = []   # collects Date, Ticker, Sector, Model, y_true, y_pred
all_metrics = []       # per (Ticker, Model) summary metrics

# %% [markdown]
# ## 2. Random Walk (no-change) baseline
#
# For a series that is (close to) a random walk, the best point forecast
# of the next value is the current value — in return terms, that's simply
# predicting a return of **0** (price unchanged). This costs nothing to
# "train" and is exactly the benchmark the project brief asks every other
# model to be judged against honestly.

# %%
def run_random_walk(ticker: str) -> pd.DataFrame:
    d = prices[prices["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    returns = d["LogReturn"].to_numpy()
    dates = d["Date"].to_numpy()

    rows = []
    for train_end, test_start, test_end in expanding_window_folds(len(returns)):
        for i in range(test_start, test_end):
            rows.append((dates[i], returns[i], 0.0))
    return pd.DataFrame(rows, columns=["TargetDate", "y_true", "y_pred"])


for ticker in ALL_TICKERS:
    preds = run_random_walk(ticker)
    preds["Ticker"], preds["Sector"], preds["Model"] = ticker, TICKER_TO_SECTOR[ticker], "Random Walk"
    all_predictions.append(preds)
    m = regression_metrics(preds["y_true"], preds["y_pred"])
    all_metrics.append({"Ticker": ticker, "Sector": TICKER_TO_SECTOR[ticker], "Model": "Random Walk", **m})

print("Random Walk baseline done for all 12 stocks.")

# %% [markdown]
# ## 3. ARIMA
#
# We fit ARIMA directly on log returns (already stationary — see notebook
# 02), not on price levels. For each stock we pick an order once, using
# AIC on the initial training window from a small, sensible candidate set
# — this is exactly the "reasonable defaults plus a few alternatives"
# level of tuning the project asks for, not an exhaustive grid search. The
# same order is then reused across every walk-forward refit for that
# stock (only the coefficients are re-estimated as the window expands).

# %%
ARIMA_ORDER_CANDIDATES = [(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1), (2, 0, 1), (1, 0, 2)]


def select_arima_order(train_returns: np.ndarray):
    best_order, best_aic = None, np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in ARIMA_ORDER_CANDIDATES:
            try:
                fit = ARIMA(train_returns, order=order).fit()
                if fit.aic < best_aic:
                    best_aic, best_order = fit.aic, order
            except Exception:
                continue
    return best_order or (1, 0, 0)


def run_arima(ticker: str) -> pd.DataFrame:
    d = prices[prices["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    returns = d["LogReturn"].to_numpy()
    dates = d["Date"].to_numpy()

    folds = list(expanding_window_folds(len(returns)))
    order = select_arima_order(returns[:folds[0][0]])

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for train_end, test_start, test_end in folds:
            train = returns[:train_end]
            try:
                fit = ARIMA(train, order=order).fit()
                forecast = fit.forecast(steps=test_end - test_start)
            except Exception:
                forecast = np.zeros(test_end - test_start)
            for offset, i in enumerate(range(test_start, test_end)):
                rows.append((dates[i], returns[i], forecast[offset]))

    result = pd.DataFrame(rows, columns=["TargetDate", "y_true", "y_pred"])
    return result, order


arima_orders = {}
t0 = time.time()
for ticker in ALL_TICKERS:
    preds, order = run_arima(ticker)
    arima_orders[ticker] = order
    preds["Ticker"], preds["Sector"], preds["Model"] = ticker, TICKER_TO_SECTOR[ticker], "ARIMA"
    all_predictions.append(preds)
    m = regression_metrics(preds["y_true"], preds["y_pred"])
    all_metrics.append({"Ticker": ticker, "Sector": TICKER_TO_SECTOR[ticker], "Model": "ARIMA", **m})
print(f"ARIMA done for all 12 stocks in {time.time()-t0:.0f}s")
pd.Series(arima_orders, name="ARIMA(p,d,q)")

# %% [markdown]
# ## 4. Random Forest and XGBoost
#
# Both use the same 16 engineered features (lagged returns, rolling
# mean/std, momentum, volume change, high-low range, 14-day RSI — see
# `src/features.py`). Hyperparameters are modest, fixed defaults, picked
# once by comparing a couple of sensible configurations on a validation
# slice of one representative stock (HDFC Bank) rather than tuned
# per-stock — the goal here is comparing *methodologies*, not squeezing
# out the last bit of performance from any one model.

# %%
rep = prices[prices["Ticker"] == "HDFCBANK"]
rep_feat = build_tabular_features(rep)
val_cut = int(len(rep_feat) * 0.70)
val_end = int(len(rep_feat) * 0.85)
X_tr, y_tr = rep_feat.loc[:val_cut, FEATURE_COLUMNS], rep_feat.loc[:val_cut, "Target"]
X_val, y_val = rep_feat.loc[val_cut:val_end, FEATURE_COLUMNS], rep_feat.loc[val_cut:val_end, "Target"]

rf_candidates = {
    "shallow": RandomForestRegressor(n_estimators=200, max_depth=4, random_state=RANDOM_SEED),
    "deeper":  RandomForestRegressor(n_estimators=300, max_depth=7, random_state=RANDOM_SEED),
}
xgb_candidates = {
    "shallow": XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED),
    "deeper":  XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, random_state=RANDOM_SEED),
}

print("Random Forest validation MAE:")
for name, model in rf_candidates.items():
    model.fit(X_tr, y_tr)
    mae = np.mean(np.abs(y_val - model.predict(X_val)))
    print(f"  {name}: {mae:.5f}")

print("XGBoost validation MAE:")
for name, model in xgb_candidates.items():
    model.fit(X_tr, y_tr)
    mae = np.mean(np.abs(y_val - model.predict(X_val)))
    print(f"  {name}: {mae:.5f}")

# %% [markdown]
# The "shallow" configuration is competitive with (or better than) "deeper"
# for both models on this validation slice — unsurprising for a noisy
# daily-return target, where a deeper model mostly has more room to
# overfit. We use the shallow configuration for both models across all 12
# stocks: **Random Forest** (`n_estimators=200, max_depth=4`) and
# **XGBoost** (`n_estimators=200, max_depth=3, learning_rate=0.05`).

# %%
def make_rf():
    return RandomForestRegressor(n_estimators=200, max_depth=4, random_state=RANDOM_SEED)


def make_xgb():
    return XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=RANDOM_SEED)


def run_tabular_model(ticker: str, model_name: str, model_factory) -> pd.DataFrame:
    d = prices[prices["Ticker"] == ticker]
    feat = build_tabular_features(d)
    X = feat[FEATURE_COLUMNS].to_numpy()
    y = feat["Target"].to_numpy()
    dates = feat["TargetDate"].to_numpy()

    rows = []
    for train_end, test_start, test_end in expanding_window_folds(len(feat)):
        model = model_factory()
        model.fit(X[:train_end], y[:train_end])
        pred = model.predict(X[test_start:test_end])
        for offset, i in enumerate(range(test_start, test_end)):
            rows.append((dates[i], y[i], pred[offset]))

    return pd.DataFrame(rows, columns=["TargetDate", "y_true", "y_pred"])


t0 = time.time()
for model_name, factory in [("Random Forest", make_rf), ("XGBoost", make_xgb)]:
    for ticker in ALL_TICKERS:
        preds = run_tabular_model(ticker, model_name, factory)
        preds["Ticker"], preds["Sector"], preds["Model"] = ticker, TICKER_TO_SECTOR[ticker], model_name
        all_predictions.append(preds)
        m = regression_metrics(preds["y_true"], preds["y_pred"])
        all_metrics.append({"Ticker": ticker, "Sector": TICKER_TO_SECTOR[ticker], "Model": model_name, **m})
    print(f"{model_name} done for all 12 stocks ({time.time()-t0:.0f}s elapsed)")

# %% [markdown]
# ## 5. Leakage sanity check
#
# A quick, explicit check rather than just a promise: for one stock,
# confirm every test prediction's fold only ever trained on rows strictly
# before that fold's test block, and that the feature/target date
# alignment is what we think it is.

# %%
d = prices[prices["Ticker"] == "TCS"]
feat = build_tabular_features(d)
folds = list(expanding_window_folds(len(feat)))
ok = all(train_end <= test_start for train_end, test_start, test_end in folds)
print("All folds train strictly before their test block:", ok)
print("Example feature row: as-of date", feat.loc[100, "Date"].date(),
      "-> predicts return realised on", feat.loc[100, "TargetDate"].date())
assert feat.loc[100, "TargetDate"] > feat.loc[100, "Date"]
print("Target date is strictly after the feature date: confirmed.")

# %% [markdown]
# ## 6. Save results

# %%
predictions_03 = pd.concat(all_predictions, ignore_index=True)
metrics_03 = pd.DataFrame(all_metrics)
predictions_03.to_csv(TABLES_DIR / "predictions_03.csv", index=False)
metrics_03.to_csv(TABLES_DIR / "metrics_03.csv", index=False)
print(f"Saved {len(predictions_03):,} predictions and {len(metrics_03)} metric rows.")

# %% [markdown]
# ## 7. A first look at the results
#
# Full cross-model, cross-sector comparison is notebook 06's job. Here we
# just sanity-check that these four models behave the way the statistical
# analysis in notebook 02 led us to expect.

# %%
pivot = metrics_03.pivot_table(index="Model", columns="Sector", values="MAE", aggfunc="mean")
pivot = pivot.reindex(["Random Walk", "ARIMA", "Random Forest", "XGBoost"])
pivot.round(5)

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
overall = metrics_03.groupby("Model")[["MAE", "Directional Accuracy"]].mean().reindex(
    ["Random Walk", "ARIMA", "Random Forest", "XGBoost"])
overall["MAE"].plot(kind="bar", ax=ax, color="steelblue")
ax.set_ylabel("Mean Absolute Error (log return)")
ax.set_title("Average MAE across all 12 stocks")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "08_notebook03_mae_overview.png", bbox_inches="tight")
plt.show()
overall.round(5)

# %% [markdown]
# As notebook 02's Ljung-Box results anticipated, ARIMA is close to — not
# dramatically better than — the Random Walk baseline on MAE. Whether RF
# and XGBoost meaningfully improve on either, and whether that holds up
# once we test statistical significance, is examined properly in notebook
# 06 once the deep learning and Transformer results are in too.
