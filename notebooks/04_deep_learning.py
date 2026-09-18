# %% [markdown]
# # 04 — Deep Learning: LSTM and 1D-CNN
#
# The two deep-learning models in this comparison. Both take the same
# input: the last **30 trading days** (about 6 weeks) of 3 channels — log
# return, volume change, and intraday high-low range — and predict the
# next day's log return. 30 days was chosen as a lookback long enough to
# cover the rolling windows used in notebook 03's features (up to 21 days)
# without making the input needlessly long; we are not trying to capture
# multi-year dependencies with a daily model.
#
# Both networks are intentionally small. This is a ~2,600-observation
# problem with a genuinely noisy target (notebook 02's Jarque-Bera /
# ARCH results); a large network here would mean more capacity to overfit
# noise, not more genuine signal to capture.
#
# Same walk-forward protocol as notebook 03: expanding window, quarterly
# refit, a validation slice carved from the *end of the training fold only*
# for early stopping (never the test fold), and scaling fit on training
# data only, per fold (see `src/dl_utils.py`).

# %%
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn

from config import ALL_TICKERS, TICKER_TO_SECTOR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR, RANDOM_SEED
from features import build_sequences
from eval_utils import regression_metrics
from dl_utils import walk_forward_sequence_model, set_seed

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
set_seed(RANDOM_SEED)

prices = pd.read_csv(PROCESSED_DIR / "prices_processed.csv", parse_dates=["Date"])
LOOKBACK = 30

# %% [markdown]
# ## 1. Architectures
#
# **LSTM** — a single LSTM layer (32 hidden units) reading the 30-day
# window, followed by one linear layer mapping the final hidden state to a
# single predicted return. This is close to the smallest LSTM regressor
# one would reasonably use; we're not stacking layers or adding attention
# on top.
#
# **1D-CNN** — two small convolutional layers (16 then 32 filters, kernel
# size 3) that slide over the time dimension, followed by global average
# pooling and a linear output layer. This lets the model pick up local
# patterns (e.g. a 3-5 day return/volume pattern) without any recurrence.

# %%
class LSTMNet(nn.Module):
    def __init__(self, n_features=3, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class CNN1DNet(nn.Module):
    def __init__(self, n_features=3, channels=(16, 32), kernel_size=3):
        super().__init__()
        c1, c2 = channels
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(n_features, c1, kernel_size, padding=pad), nn.ReLU(),
            nn.Conv1d(c1, c2, kernel_size, padding=pad), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(c2, 1)

    def forward(self, x):
        # x: (batch, time, channels) -> conv1d wants (batch, channels, time)
        x = x.permute(0, 2, 1)
        x = self.net(x).squeeze(-1)
        return self.fc(x)


n_params_lstm = sum(p.numel() for p in LSTMNet().parameters())
n_params_cnn = sum(p.numel() for p in CNN1DNet().parameters())
print(f"LSTM parameters: {n_params_lstm:,}")
print(f"1D-CNN parameters: {n_params_cnn:,}")

# %% [markdown]
# ## 2. Run both models on all 12 stocks

# %%
all_predictions, all_metrics = [], []

model_factories = {
    "LSTM": lambda: LSTMNet(),
    "1D-CNN": lambda: CNN1DNet(),
}

for model_name, factory in model_factories.items():
    t0 = time.time()
    for ticker in ALL_TICKERS:
        d = prices[prices["Ticker"] == ticker]
        X, y, dates = build_sequences(d, lookback=LOOKBACK)
        preds = walk_forward_sequence_model(factory, X, y, dates, epochs=25, patience=4,
                                             batch_size=64, lr=1e-3, seed=RANDOM_SEED)
        preds["Ticker"], preds["Sector"], preds["Model"] = ticker, TICKER_TO_SECTOR[ticker], model_name
        all_predictions.append(preds)
        m = regression_metrics(preds["y_true"], preds["y_pred"])
        all_metrics.append({"Ticker": ticker, "Sector": TICKER_TO_SECTOR[ticker], "Model": model_name, **m})
    print(f"{model_name} done for all 12 stocks in {time.time()-t0:.0f}s")

# %% [markdown]
# ## 3. Save results

# %%
predictions_04 = pd.concat(all_predictions, ignore_index=True)
metrics_04 = pd.DataFrame(all_metrics)
predictions_04.to_csv(TABLES_DIR / "predictions_04.csv", index=False)
metrics_04.to_csv(TABLES_DIR / "metrics_04.csv", index=False)
print(f"Saved {len(predictions_04):,} predictions and {len(metrics_04)} metric rows.")

# %% [markdown]
# ## 4. A first look

# %%
prior_metrics = pd.read_csv(TABLES_DIR / "metrics_03.csv")
combined = pd.concat([prior_metrics, metrics_04], ignore_index=True)
order = ["Random Walk", "ARIMA", "Random Forest", "XGBoost", "LSTM", "1D-CNN"]

fig, ax = plt.subplots(figsize=(9, 4.5))
overall = combined.groupby("Model")[["MAE", "Directional Accuracy"]].mean().reindex(order)
overall["MAE"].plot(kind="bar", ax=ax, color="darkorange")
ax.set_ylabel("Mean Absolute Error (log return)")
ax.set_title("Average MAE across all 12 stocks — models so far")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "09_notebook04_mae_overview.png", bbox_inches="tight")
plt.show()
overall.round(5)

# %% [markdown]
# So far, added model complexity (RF/XGBoost, then LSTM/CNN) is not
# producing a clear, large drop in MAE relative to the Random Walk /
# ARIMA baselines — consistent with the statistical picture from notebook
# 02 (returns are close to, but not perfectly, unpredictable). We hold
# off on any real interpretation until notebook 06, once the Transformer
# results and formal significance testing (Diebold-Mariano) are in —
# small differences in a table like this one are exactly what that test
# is for.
