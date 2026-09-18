# %% [markdown]
# # 02 — Statistical Analysis
#
# This notebook runs the formal statistical tests that notebook 01's EDA
# only hinted at, and uses their results to make one concrete modelling
# decision: **do we forecast price levels or returns?**
#
# Each test below is included because it directly informs a modelling
# choice later in the project — not run just because it exists. For every
# test we state: why we're using it, its null hypothesis, the result, and
# what that result means for the modelling decisions ahead.

# %%
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")  # statsmodels is noisy about upcoming API changes; not relevant here

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from scipy.stats import jarque_bera

from config import SECTOR_STOCKS, ALL_TICKERS, TICKER_TO_SECTOR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

prices = pd.read_csv(PROCESSED_DIR / "prices_processed.csv", parse_dates=["Date"])

# %% [markdown]
# ## 1. Stationarity: ADF and KPSS
#
# **Why.** Almost every classical time-series model (ARIMA, and the whole
# idea of "features computed over a stable process" for the ML/DL models)
# assumes or works much better with a stationary series. If we forecast a
# non-stationary series directly, a model can score well simply by
# tracking the trend, without learning anything genuinely predictive.
#
# We run two complementary tests, because they check the same thing from
# opposite directions and a well-known quirk is that they can disagree:
#
# * **ADF (Augmented Dickey-Fuller).** H0: the series has a unit root
#   (is non-stationary). A low p-value lets us reject H0, i.e. conclude
#   stationarity.
# * **KPSS.** H0: the series *is* (trend-)stationary. A low p-value lets us
#   reject H0, i.e. conclude non-stationarity — the opposite direction
#   from ADF.
#
# Running both avoids relying on a single test's assumptions.

# %%
def adf_kpss(series: pd.Series) -> dict:
    series = series.dropna()
    adf_stat, adf_p, *_ = adfuller(series, autolag="AIC")
    try:
        kpss_stat, kpss_p, *_ = kpss(series, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p = np.nan, np.nan
    return {"ADF stat": adf_stat, "ADF p-value": adf_p,
            "KPSS stat": kpss_stat, "KPSS p-value": kpss_p}


stationarity_rows = []
for ticker in ALL_TICKERS:
    d = prices[prices["Ticker"] == ticker]
    price_result = adf_kpss(d["Close"])
    return_result = adf_kpss(d["LogReturn"])
    stationarity_rows.append({
        "Ticker": ticker, "Sector": TICKER_TO_SECTOR[ticker], "Series": "Price",
        **price_result,
    })
    stationarity_rows.append({
        "Ticker": ticker, "Sector": TICKER_TO_SECTOR[ticker], "Series": "Log return",
        **return_result,
    })

stationarity = pd.DataFrame(stationarity_rows)
stationarity["ADF: stationary (p<0.05)"] = stationarity["ADF p-value"] < 0.05
stationarity["KPSS: stationary (p>0.05)"] = stationarity["KPSS p-value"] > 0.05
stationarity.to_csv(TABLES_DIR / "stationarity_tests.csv", index=False)
stationarity.round(4)

# %%
summary = (
    stationarity.groupby("Series")[["ADF: stationary (p<0.05)", "KPSS: stationary (p>0.05)"]]
    .mean()
    .rename(columns=lambda c: c + " (share of 12 stocks)")
)
summary

# %% [markdown]
# **Result.** For price levels, ADF fails to reject the unit-root null for
# essentially every stock, and KPSS rejects stationarity for essentially
# every stock — both tests agree prices are non-stationary. For log
# returns, ADF strongly rejects the unit-root null and KPSS fails to
# reject stationarity for essentially every stock — both tests agree
# returns are (at least approximately) stationary.
#
# **What this means for modelling.** Forecasting raw price levels would
# mean forecasting a non-stationary series, where a naive "no change"
# prediction is already hard to beat by construction and any model can
# look artificially good just by following the trend. Log returns are
# statistically much better behaved and are the series ARIMA and the
# stationarity assumptions behind the ML/DL feature engineering actually
# expect. **We forecast log returns as the primary target for every model
# in this project**, and convert return forecasts back to a price
# trajectory only where it's useful for interpretation (e.g. the
# actual-vs-predicted price charts and the simple backtest in notebook
# 06). We don't duplicate the whole project with a parallel price-forecasting
# track — the stationarity evidence here is clear enough that it isn't
# needed, and ARIMA's own differencing step in notebook 03 effectively
# confirms the same conclusion once more.

# %% [markdown]
# ## 2. Autocorrelation of returns: Ljung-Box
#
# **Why.** Notebook 01's ACF plot suggested returns are close to white
# noise. The Ljung-Box test makes this precise: it jointly tests whether
# the first *k* autocorrelations are all zero. This tells us how much
# linear structure, if any, a model like ARIMA has to work with in the raw
# returns.
#
# H0: the return series shows no autocorrelation up to lag *k* (white
# noise).

# %%
ljung_rows = []
for ticker in ALL_TICKERS:
    d = prices[prices["Ticker"] == ticker]
    lb = acorr_ljungbox(d["LogReturn"], lags=[10, 20], return_df=True)
    ljung_rows.append({
        "Ticker": ticker, "Sector": TICKER_TO_SECTOR[ticker],
        "LB stat (lag 10)": lb.loc[10, "lb_stat"], "p-value (lag 10)": lb.loc[10, "lb_pvalue"],
        "LB stat (lag 20)": lb.loc[20, "lb_stat"], "p-value (lag 20)": lb.loc[20, "lb_pvalue"],
    })
ljung = pd.DataFrame(ljung_rows)
ljung.to_csv(TABLES_DIR / "ljung_box_returns.csv", index=False)
ljung.round(4)

# %% [markdown]
# **Result.** Roughly half the stocks show statistically significant
# autocorrelation at the 5% level at one or both lags, but the effect
# sizes (autocorrelation coefficients, not shown in the summary table but
# visible in notebook 01's ACF plot) are small. This is a realistic,
# mild-efficiency-violation result for large, liquid stocks — not the
# strong, exploitable structure the p-values alone might suggest. **This
# means ARIMA has, at best, a small amount of genuine linear structure to
# exploit, and we should not expect it to dramatically beat a random walk**
# — a useful expectation to carry into notebook 03's results.

# %% [markdown]
# ## 3. Distribution: Jarque-Bera
#
# **Why.** Several of our evaluation choices (e.g. using RMSE, which is
# sensitive to outliers) and modelling choices implicitly assume something
# about the error/return distribution. Jarque-Bera checks whether returns
# are normally distributed, based on their sample skewness and kurtosis.
#
# H0: the series is normally distributed.

# %%
jb_rows = []
for ticker in ALL_TICKERS:
    d = prices[prices["Ticker"] == ticker]
    stat, p = jarque_bera(d["LogReturn"].dropna())
    jb_rows.append({
        "Ticker": ticker, "Sector": TICKER_TO_SECTOR[ticker],
        "Skewness": d["LogReturn"].skew(), "Excess kurtosis": d["LogReturn"].kurtosis(),
        "JB stat": stat, "JB p-value": p,
    })
jb = pd.DataFrame(jb_rows)
jb.to_csv(TABLES_DIR / "jarque_bera_returns.csv", index=False)
jb.round(4)

# %% [markdown]
# **Result.** Jarque-Bera rejects normality (p < 0.001) for all 12 stocks,
# and every stock shows positive excess kurtosis (fat tails — large moves
# are far more common than a normal distribution would predict), matching
# the well-known "fat tails" stylised fact of daily equity returns and the
# COVID-era extreme days flagged in notebook 01.
#
# **What this means for modelling.** We should not assume Gaussian errors
# anywhere, and should be cautious about metrics that are very sensitive to
# a handful of extreme days. This is exactly why the evaluation in this
# project uses **MAE and directional accuracy as primary metrics, not just
# RMSE** — MAE is far less distorted by the fat tails than RMSE, and
# directional accuracy asks a question (up or down?) that doesn't depend
# on the return distribution's shape at all.

# %% [markdown]
# ## 4. Volatility clustering: ARCH test
#
# **Why.** Notebook 01's rolling-volatility plot showed visible clustering
# (calm periods, turbulent periods). Engle's ARCH-LM test makes this
# formal: it regresses squared returns on their own lags and tests whether
# that regression has any explanatory power. If it does, volatility today
# depends on volatility recently — the defining feature that GARCH-family
# models are built to capture.
#
# H0: no ARCH effect (squared returns are not autocorrelated, i.e.
# volatility is not clustering).

# %%
arch_rows = []
for ticker in ALL_TICKERS:
    d = prices[prices["Ticker"] == ticker]
    stat, p, _, _ = het_arch(d["LogReturn"].dropna(), nlags=10)
    arch_rows.append({"Ticker": ticker, "Sector": TICKER_TO_SECTOR[ticker],
                       "ARCH-LM stat": stat, "p-value": p})
arch_test = pd.DataFrame(arch_rows)
arch_test.to_csv(TABLES_DIR / "arch_test_returns.csv", index=False)
arch_test.round(4)

# %% [markdown]
# **Result.** The ARCH-LM test rejects the no-clustering null (p < 0.001)
# for all 12 stocks — strong, unambiguous evidence of volatility
# clustering, consistent with the EDA.
#
# **What this means for modelling.** Volatility clustering is a real,
# well-established feature of this data. But — and this is the point made
# in the project brief — **that does not automatically mean GARCH belongs
# in the main model comparison.** GARCH models the *variance* of returns,
# not their conditional *mean*; it is not built to predict the sign or
# magnitude of tomorrow's return the way ARIMA/RF/XGBoost/LSTM/CNN/
# Transformer are. Forcing GARCH into the same price/return-forecasting
# comparison as the other seven models would be comparing it on a task it
# isn't designed for. Given that the ARCH test here confirms clustering is
# real and worth investigating, we include a **small, separate GARCH
# volatility-forecasting side-experiment in notebook 06**, evaluated on
# its own terms (forecasting volatility, not price/return direction),
# rather than bolting it onto the main leaderboard.

# %% [markdown]
# ## 5. Sector-level summary

# %%
fig, ax = plt.subplots(figsize=(9, 4))
sector_order = list(SECTOR_STOCKS.keys())
sns.boxplot(data=jb.merge(arch_test, on=["Ticker", "Sector"], suffixes=("_jb", "_arch")),
            x="Sector", y="Excess kurtosis", order=sector_order, ax=ax)
ax.set_title("Excess kurtosis of daily log returns, by sector")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "07_kurtosis_by_sector.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Summary of decisions made in this notebook
#
# | Test | Result | Decision it drives |
# |---|---|---|
# | ADF + KPSS | Prices non-stationary, returns stationary | **Forecast log returns, not price levels**, for every model |
# | Ljung-Box | Weak-to-moderate autocorrelation in returns | Don't expect ARIMA to dramatically beat the random walk |
# | Jarque-Bera | Returns non-normal, fat-tailed | Use MAE and directional accuracy as primary metrics, not RMSE alone |
# | ARCH-LM | Strong volatility clustering, all 12 stocks | Justifies a small, separate GARCH volatility side-experiment — not a place in the main model leaderboard |
#
# These decisions carry forward into every remaining notebook.
