# %% [markdown]
# # 05 — Transformer
#
# The most complex model in the comparison: a small Transformer encoder
# over the same 30-day, 3-channel input used by the LSTM and 1D-CNN in
# notebook 04. Same walk-forward protocol, same features, same metrics —
# the only thing that changes from notebook 04 is the architecture, which
# is what makes the eventual "does more complexity help?" comparison in
# notebook 06 meaningful.
#
# We deliberately use a **single encoder layer with a small model
# dimension**, not a multi-layer, multi-head "real" Transformer of the
# kind used for language modelling. Two reasons: (1) with ~2,600
# observations and a noisy daily-return target, a large Transformer has
# far more capacity than the data can support without overfitting, and
# (2) the research question is about whether attention-based sequence
# modelling helps *at all* on this problem, which a small model can
# already answer.

# %%
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import time
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
# ## 1. Architecture
#
# * Linear projection of the 3 input channels up to a 32-dimensional
#   model space.
# * A learned positional embedding (30 positions) so the model can tell
#   *which* day in the window it's looking at — plain self-attention has
#   no sense of order on its own.
# * **One** `TransformerEncoderLayer` (4 attention heads, feed-forward
#   width 64, dropout 0.1).
# * Mean-pool over the 30 time steps, then a linear layer to a single
#   predicted return.

# %%
class TinyTransformer(nn.Module):
    def __init__(self, n_features=3, d_model=32, nhead=4, ff=64, lookback=LOOKBACK, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, lookback, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ff,
                                            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        z = self.input_proj(x) + self.pos_embed
        z = self.encoder(z)
        z = z.mean(dim=1)
        return self.fc(z)


n_params = sum(p.numel() for p in TinyTransformer().parameters())
print(f"Transformer parameters: {n_params:,}  (LSTM: 4,769 · 1D-CNN: 1,761, for comparison)")

# %% [markdown]
# ## 2. Run on all 12 stocks
#
# This is the slowest model to train in the whole project (attention is
# more expensive per step than an LSTM or CNN of similar size), so it's
# worth a moment: it still trains in well under a minute per stock on CPU,
# because the model and dataset are both small by design.

# %%
all_predictions, all_metrics = [], []

t0 = time.time()
for ticker in ALL_TICKERS:
    d = prices[prices["Ticker"] == ticker]
    X, y, dates = build_sequences(d, lookback=LOOKBACK)
    preds = walk_forward_sequence_model(lambda: TinyTransformer(), X, y, dates,
                                         epochs=25, patience=4, batch_size=64, lr=1e-3,
                                         seed=RANDOM_SEED)
    preds["Ticker"], preds["Sector"], preds["Model"] = ticker, TICKER_TO_SECTOR[ticker], "Transformer"
    all_predictions.append(preds)
    m = regression_metrics(preds["y_true"], preds["y_pred"])
    all_metrics.append({"Ticker": ticker, "Sector": TICKER_TO_SECTOR[ticker], "Model": "Transformer", **m})
    print(f"  {ticker} done ({time.time()-t0:.0f}s elapsed)")

print(f"Transformer done for all 12 stocks in {time.time()-t0:.0f}s")

# %% [markdown]
# ## 3. Save results

# %%
predictions_05 = pd.concat(all_predictions, ignore_index=True)
metrics_05 = pd.DataFrame(all_metrics)
predictions_05.to_csv(TABLES_DIR / "predictions_05.csv", index=False)
metrics_05.to_csv(TABLES_DIR / "metrics_05.csv", index=False)
print(f"Saved {len(predictions_05):,} predictions and {len(metrics_05)} metric rows.")

# %% [markdown]
# ## 4. All seven models so far

# %%
metrics_03 = pd.read_csv(TABLES_DIR / "metrics_03.csv")
metrics_04 = pd.read_csv(TABLES_DIR / "metrics_04.csv")
combined = pd.concat([metrics_03, metrics_04, metrics_05], ignore_index=True)
order = ["Random Walk", "ARIMA", "Random Forest", "XGBoost", "LSTM", "1D-CNN", "Transformer"]

fig, ax = plt.subplots(figsize=(9.5, 4.5))
overall = combined.groupby("Model")[["MAE", "Directional Accuracy"]].mean().reindex(order)
overall["MAE"].plot(kind="bar", ax=ax, color="seagreen")
ax.set_ylabel("Mean Absolute Error (log return)")
ax.set_title("Average MAE across all 12 stocks — all 7 models")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "10_notebook05_mae_overview.png", bbox_inches="tight")
plt.show()
overall.round(5)

# %% [markdown]
# All seven models are now trained and evaluated identically. Notebook 06
# does the actual research analysis: per-sector comparison tables,
# Diebold-Mariano significance tests, regime analysis, the GARCH
# volatility side-experiment, and the answer to the research question.
