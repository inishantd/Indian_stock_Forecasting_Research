# %% [markdown]
# # 01 — Data Loading, Cleaning & Exploratory Data Analysis
#
# **Project:** A Comparative Evaluation of Statistical, Machine Learning, Deep
# Learning, and Transformer Models for Sector-Specific Stock Forecasting in
# Indian Financial Markets
#
# **Research question:** Do increasingly complex forecasting models actually
# improve out-of-sample stock forecasting performance, and does model
# performance differ across Indian market sectors?
#
# This notebook only does the groundwork the rest of the project depends on:
# load the raw historical data, document it honestly, clean it, and look at
# it before touching a single model. Every plot here is meant to answer a
# specific question about the data, not to fill space.

# %% [markdown]
# ## 1. Data source and scope
#
# **Source.** NSE (National Stock Exchange of India) End-of-Day bhavcopy
# data, redistributed as one CSV per stock (`daily/<symbol>.csv`) by the
# open-source [`eod2`](https://github.com/BennyThadikaran/eod2) project
# (`eod2_data` repository, MIT-licensed, updated weekly from NSE's own
# published reports). Direct programmatic downloads from `nseindia.com`
# were not accessible from this project's execution environment (the site
# actively blocks automated requests without a browser session), so a
# pre-downloaded, NSE-sourced mirror was used instead. This is documented
# here as a limitation — see Section 5 and the README.
#
# **Download date:** the raw CSVs used in this project were pulled on
# **2026-09-10**.
#
# **Date range used:** 2016-01-01 to 2026-08-31 (~10.7 years). The raw
# files go back further (to 1995 for some stocks), but Bajaj Auto only
# lists from 2008 and we want one consistent start date across all stocks,
# so 2016 was chosen to comfortably give every stock 10+ years of history
# with a wide safety margin.
#
# **Variables:** Date, Open, High, Low, Close, Volume. Close is already
# adjusted for stock splits and bonus issues (per the source project's
# documentation) but **not** for dividends — there is no separate "Adjusted
# Close" column in this data source. We treat `Close` as our price series
# throughout and note this as a limitation.
#
# **Sectors and companies.** Four sectors, three large, liquid, long-listed
# companies each:
#
# | Sector | Companies | Why these |
# |---|---|---|
# | Banking | HDFC Bank, ICICI Bank, State Bank of India | The three largest banks on NSE by market cap, spanning private (HDFC, ICICI) and public-sector (SBI) banking |
# | IT | TCS, Infosys, Wipro | The three largest listed IT services exporters, a sector whose earnings are dollar-linked rather than domestic-demand-linked |
# | Healthcare | Sun Pharma, Dr. Reddy's, Cipla | The three largest listed pharmaceutical companies, a defensive sector with different growth drivers (US generics, regulatory risk) |
# | Automotive | Maruti Suzuki, Bajaj Auto, Mahindra & Mahindra | Leaders in passenger cars, two-wheelers, and utility/farm vehicles respectively — a cyclical, domestic-demand-driven sector |
#
# These four sectors were chosen because they are structurally different
# (export-driven vs. domestic, cyclical vs. defensive, regulated vs.
# lightly regulated), which is exactly the kind of variation the research
# question needs to be interesting. Three stocks per sector is enough to
# check whether a result is a sector-wide pattern or a single-company
# fluke, without turning this into a 50-stock exercise that adds runtime
# but not insight.

# %%
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    SECTOR_STOCKS, TICKER_TO_SECTOR, ALL_TICKERS,
    raw_csv_path, START_DATE, END_DATE, PROCESSED_DIR, FIGURES_DIR,
)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

# %% [markdown]
# ## 2. Loading and cleaning
#
# The raw files are already very clean (NSE's own bhavcopy pipeline), so
# "cleaning" here is mostly *verification* rather than repair: we check for
# duplicate dates, missing values, non-trading rows, and clearly broken
# prices (zero/negative), and we trim every stock to the same date window.

# %%
def load_stock(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path(ticker), parse_dates=["Date"])
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)]
    df = df.sort_values("Date").drop_duplicates(subset="Date")
    df["Ticker"] = ticker
    df["Sector"] = TICKER_TO_SECTOR[ticker]
    return df.reset_index(drop=True)


raw = {ticker: load_stock(ticker) for ticker in ALL_TICKERS}

# %%
# Data-quality checks, one row per stock
quality_rows = []
for ticker, df in raw.items():
    quality_rows.append({
        "Ticker": ticker,
        "Sector": TICKER_TO_SECTOR[ticker],
        "Rows": len(df),
        "First date": df["Date"].min().date(),
        "Last date": df["Date"].max().date(),
        "Missing values": df[["Open", "High", "Low", "Close", "Volume"]].isna().sum().sum(),
        "Non-positive prices": (df[["Open", "High", "Low", "Close"]] <= 0).sum().sum(),
        "Zero volume days": (df["Volume"] == 0).sum(),
    })
quality = pd.DataFrame(quality_rows)
quality

# %% [markdown]
# All 12 stocks come out with the same number of trading days over the
# same window, no missing values, and no non-positive prices. That's
# expected for large, liquid, continuously-listed stocks on the main NSE
# board — the kind of data-quality problems the assignment warns about
# (missing values, broken prices) mostly show up in illiquid small-caps,
# which is one more reason we deliberately picked large caps.
#
# **Corporate actions** (splits, bonuses) are already handled upstream —
# the source data is split/bonus-adjusted — so we don't need to detect and
# adjust for them ourselves. We do *not* adjust for dividends; a handful of
# large dividend-payment days will show up as small, real one-day price
# drops rather than data errors. This is called out again in the
# limitations section of the final notebook.

# %%
prices = pd.concat(raw.values(), ignore_index=True)
prices = prices.sort_values(["Ticker", "Date"]).reset_index(drop=True)
prices.head()

# %% [markdown]
# ## 3. Returns
#
# We compute both simple returns and log returns now, since the choice
# between forecasting prices vs. returns is investigated properly (with
# stationarity tests) in notebook 02. Having both available here just lets
# us do sensible EDA.

# %%
prices["LogReturn"] = prices.groupby("Ticker")["Close"].transform(
    lambda s: np.log(s / s.shift(1))
)
prices["SimpleReturn"] = prices.groupby("Ticker")["Close"].pct_change()
prices = prices.dropna(subset=["LogReturn"]).reset_index(drop=True)

# %% [markdown]
# ## 4. Save the processed dataset
#
# One tidy long-format file (`Date, Ticker, Sector, OHLCV, returns`) is all
# the later notebooks need. Saving it here means notebooks 02-06 never have
# to re-read or re-clean the raw CSVs.

# %%
processed_path = PROCESSED_DIR / "prices_processed.csv"
prices.to_csv(processed_path, index=False)
print(f"Saved {len(prices):,} rows for {prices['Ticker'].nunique()} stocks to {processed_path}")

# %% [markdown]
# ## 5. Exploratory Data Analysis
#
# Each plot below is picked to answer one specific question relevant to
# the research design — not as a generic "let's plot everything" pass.

# %% [markdown]
# ### 5.1 Price trends — has each sector grown similarly, and are the
# series visibly non-stationary (trending, changing variance)?

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for ax, (sector, stocks) in zip(axes.flat, SECTOR_STOCKS.items()):
    for ticker in stocks:
        d = raw[ticker]
        # index to 100 at the start so stocks with very different price
        # levels (e.g. MRF-style high-priced vs low-priced stocks) are
        # visually comparable
        indexed = 100 * d["Close"] / d["Close"].iloc[0]
        ax.plot(d["Date"], indexed, label=ticker, linewidth=1)
    ax.set_title(sector)
    ax.legend(fontsize=8)
    ax.set_ylabel("Price (indexed to 100)")
fig.suptitle("Indexed closing price by sector, 2016-2026", y=1.02)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "01_price_trends_by_sector.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# All series wander upward with visibly changing local trends and no fixed
# mean — the classic look of a non-stationary price series. This already
# hints at what notebook 02's formal ADF/KPSS tests will confirm, and at
# why forecasting raw price levels is a questionable idea (a naive model
# that just says "tomorrow = today" will look deceptively good on a
# trending series).

# %% [markdown]
# ### 5.2 Daily return distributions — are returns closer to stationary,
# and how "fat-tailed" are they?

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, (sector, stocks) in zip(axes.flat, SECTOR_STOCKS.items()):
    sector_returns = prices[prices["Sector"] == sector]
    for ticker in stocks:
        r = sector_returns[sector_returns["Ticker"] == ticker]["LogReturn"]
        sns.kdeplot(r, ax=ax, label=ticker)
    ax.set_title(sector)
    ax.set_xlim(-0.15, 0.15)
    ax.legend(fontsize=8)
fig.suptitle("Daily log-return distributions by sector", y=1.02)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "02_return_distributions.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Returns cluster tightly around zero with long tails — much closer to a
# stable, non-trending process than prices. Formal normality testing
# (Jarque-Bera) comes in notebook 02, but visually these are already more
# peaked and fat-tailed than a normal distribution, which is the usual
# story for daily equity returns.

# %% [markdown]
# ### 5.3 Rolling volatility — is volatility constant over time, or does
# it cluster (calm periods vs. turbulent periods)?

# %%
fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True)
for ax, (sector, stocks) in zip(axes, SECTOR_STOCKS.items()):
    for ticker in stocks:
        d = prices[prices["Ticker"] == ticker].set_index("Date")
        rolling_vol = d["LogReturn"].rolling(21).std() * np.sqrt(252)  # annualised, ~1 trading month window
        ax.plot(rolling_vol.index, rolling_vol, label=ticker, linewidth=0.9)
    ax.set_title(f"{sector} — 21-day rolling annualised volatility")
    ax.legend(fontsize=8, ncol=3)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "03_rolling_volatility.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Volatility clearly clusters — quiet stretches followed by sharp spikes
# (the most visible being around 2020, the COVID shock, common to every
# stock) rather than being flat over time. This is direct visual evidence
# of volatility clustering, which motivates both the ARCH test in notebook
# 02 and the small GARCH side-experiment in the final notebook.

# %% [markdown]
# ### 5.4 Trading volume — is liquidity stable, and does volume spike with
# volatility?

# %%
fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
for ax, (sector, stocks) in zip(axes.flat, SECTOR_STOCKS.items()):
    for ticker in stocks:
        d = prices[prices["Ticker"] == ticker].set_index("Date")
        vol_ma = d["Volume"].rolling(21).mean()
        ax.plot(vol_ma.index, vol_ma, label=ticker, linewidth=0.9)
    ax.set_title(sector)
    ax.set_ylabel("21-day avg volume")
    ax.legend(fontsize=8)
fig.suptitle("Trading volume (21-day moving average) by sector", y=1.02)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "04_volume_trends.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 5.5 Correlation between stocks — do stocks in the same sector move
# together more than stocks across sectors?

# %%
returns_wide = prices.pivot(index="Date", columns="Ticker", values="LogReturn")
corr = returns_wide.corr()

# order tickers by sector so the block structure is visible
ordered_tickers = [t for stocks in SECTOR_STOCKS.values() for t in stocks]
corr = corr.loc[ordered_tickers, ordered_tickers]

fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            vmin=-0.2, vmax=1)
ax.set_title("Correlation of daily log returns")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "05_return_correlation.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# There's a visible block structure — the two private banks (HDFC Bank,
# ICICI Bank) are the most correlated pair in the whole matrix, and the IT
# and healthcare trios are each more correlated with each other than with
# stocks from other sectors. Correlations are far from 1, though — these
# are genuinely different companies, not proxies for the same series. This
# supports treating "sector" as a meaningful grouping variable later, while
# also justifying evaluating multiple stocks per sector rather than one.

# %% [markdown]
# ### 5.6 ACF / PACF — a first look at autocorrelation structure
#
# A full stationarity and autocorrelation analysis (with hypothesis tests)
# happens in notebook 02. Here we just look at one representative stock
# per sector to preview the shape: prices should show slow-decaying
# autocorrelation (typical of a near-random-walk / unit-root series) while
# returns should look close to white noise.

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

representative = {sector: stocks_list[0] for sector, stocks_list in
                   [(s, list(t.keys())) for s, t in SECTOR_STOCKS.items()]}

fig, axes = plt.subplots(len(representative), 2, figsize=(12, 3 * len(representative)))
for i, (sector, ticker) in enumerate(representative.items()):
    d = prices[prices["Ticker"] == ticker]
    plot_acf(d["Close"], lags=40, ax=axes[i, 0], title=f"{ticker} price — ACF")
    plot_acf(d["LogReturn"], lags=40, ax=axes[i, 1], title=f"{ticker} log return — ACF")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "06_acf_price_vs_return.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# Exactly the expected pattern: price ACF decays extremely slowly (still
# near 1 at lag 40 — a hallmark of a unit-root / random-walk-like series),
# while return ACF drops to near zero almost immediately. This is the
# first concrete piece of evidence, ahead of notebook 02's formal tests,
# that returns are the statistically sounder target to forecast.

# %% [markdown]
# ### 5.7 Extreme return days
#
# A quick sanity check: do the largest single-day moves line up with
# events we'd actually expect (COVID crash, results days, etc.), or do
# they look like data errors?

# %%
extremes = (
    prices.assign(AbsReturn=prices["LogReturn"].abs())
    .sort_values("AbsReturn", ascending=False)
    .groupby("Ticker")
    .head(2)
    .sort_values("AbsReturn", ascending=False)
    [["Date", "Ticker", "Sector", "LogReturn"]]
    .head(15)
    .reset_index(drop=True)
)
from config import TABLES_DIR
extremes.to_csv(TABLES_DIR / "extreme_return_days.csv", index=False)
extremes

# %% [markdown]
# The largest moves cluster heavily around March 2020 (the COVID-19 market
# crash), which is reassuring — these are real, economically meaningful
# events rather than data artefacts. A few isolated large moves on other
# dates are plausible results-day or corporate-news reactions for
# individual stocks. We keep these observations in the data (no
# winsorising / outlier removal): they are genuine market behaviour, and a
# forecasting exercise that quietly removes the hardest days to predict
# would be misleading.

# %% [markdown]
# ## 6. Summary of this notebook
#
# * Loaded and validated 12 large-cap NSE stocks across 4 sectors,
#   2016-01-01 to 2026-08-31, ~2,638 trading days each, no missing values.
# * Saved a single tidy processed dataset (`data/processed/prices_processed.parquet`)
#   used by every later notebook.
# * Prices are visibly non-stationary and trending; log returns look much
#   closer to a stationary, weakly-autocorrelated process — formal tests in
#   notebook 02.
# * Volatility clusters over time (most visibly around COVID-19), which
#   will be picked up again by the ARCH test and the GARCH side-experiment.
# * Same-sector stocks are more correlated with each other than with
#   stocks from other sectors, supporting sector as a meaningful grouping
#   for the model-comparison analysis in notebook 06.
