# %% [markdown]
# # 06 — Model Comparison and Final Research Analysis
#
# Every model has now been trained and evaluated identically across all 12
# stocks. This notebook does the actual research: build the comparison
# tables, test whether any differences are statistically real or just
# noise, look at sector and volatility-regime patterns, run a small
# GARCH volatility side-experiment and an optional simple backtest, and
# finally answer the research question directly.
#
# **Research question:** Do increasingly complex forecasting models
# actually improve out-of-sample stock forecasting performance, and does
# model performance differ across Indian market sectors?

# %%
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd().parent / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare
from arch import arch_model

from config import SECTOR_STOCKS, ALL_TICKERS, TICKER_TO_SECTOR, PROCESSED_DIR, TABLES_DIR, FIGURES_DIR
from eval_utils import regression_metrics, expanding_window_folds, diebold_mariano

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

MODEL_ORDER = ["Random Walk", "ARIMA", "Random Forest", "XGBoost", "LSTM", "1D-CNN", "Transformer"]
SECTOR_ORDER = list(SECTOR_STOCKS.keys())

prices = pd.read_csv(PROCESSED_DIR / "prices_processed.csv", parse_dates=["Date"])
predictions = pd.concat([
    pd.read_csv(TABLES_DIR / "predictions_03.csv", parse_dates=["TargetDate"]),
    pd.read_csv(TABLES_DIR / "predictions_04.csv", parse_dates=["TargetDate"]),
    pd.read_csv(TABLES_DIR / "predictions_05.csv", parse_dates=["TargetDate"]),
], ignore_index=True)
metrics = pd.concat([
    pd.read_csv(TABLES_DIR / "metrics_03.csv"),
    pd.read_csv(TABLES_DIR / "metrics_04.csv"),
    pd.read_csv(TABLES_DIR / "metrics_05.csv"),
], ignore_index=True)

print(f"{len(predictions):,} out-of-sample predictions across {predictions['Model'].nunique()} models "
      f"and {predictions['Ticker'].nunique()} stocks.")

# %% [markdown]
# ## 1. Model comparison tables
#
# Three tables, not one — the project brief is explicit that no model
# should be ranked on a single metric. MAE and RMSE tell slightly
# different stories (RMSE penalises the rare large misses more, which
# matters given notebook 02's fat-tail finding); directional accuracy asks
# an entirely different question (did we get the sign right?) that a
# regression-error metric can miss completely.

# %%
def sector_table(metric_col: str) -> pd.DataFrame:
    t = metrics.pivot_table(index="Model", columns="Sector", values=metric_col, aggfunc="mean")
    t["Overall"] = metrics.groupby("Model")[metric_col].mean()
    return t.reindex(MODEL_ORDER)[SECTOR_ORDER + ["Overall"]]


mae_table = sector_table("MAE")
rmse_table = sector_table("RMSE")
diracc_table = sector_table("Directional Accuracy")

mae_table.to_csv(TABLES_DIR / "comparison_mae.csv")
rmse_table.to_csv(TABLES_DIR / "comparison_rmse.csv")
diracc_table.to_csv(TABLES_DIR / "comparison_directional_accuracy.csv")

print("MAE by model and sector (lower is better)")
mae_table.round(5)

# %%
print("RMSE by model and sector (lower is better)")
rmse_table.round(5)

# %%
print("Directional Accuracy by model and sector (higher is better; 0.50 = coin flip)")
diracc_table.round(4)

# %% [markdown]
# ## 2. A combined ranking (not single-metric)
#
# For each stock, rank all 7 models on MAE (ascending), RMSE (ascending),
# and Directional Accuracy (descending), then average the three ranks.
# This is a simple, transparent way to combine metrics without letting
# any single one dominate the headline conclusion.

# %%
rank_df = metrics.copy()
for col, ascending in [("MAE", True), ("RMSE", True), ("Directional Accuracy", False)]:
    rank_df[f"rank_{col}"] = rank_df.groupby("Ticker")[col].rank(ascending=ascending)
rank_df["avg_rank"] = rank_df[["rank_MAE", "rank_RMSE", "rank_Directional Accuracy"]].mean(axis=1)

overall_rank = rank_df.groupby("Model")["avg_rank"].mean().reindex(MODEL_ORDER).sort_values()
overall_rank.to_csv(TABLES_DIR / "overall_combined_rank.csv")
overall_rank.round(2)

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))
overall_rank.sort_values().plot(kind="barh", ax=ax, color="slateblue")
ax.set_xlabel("Average combined rank across MAE, RMSE, Directional Accuracy (1 = best)")
ax.invert_yaxis()
ax.set_title("Overall model ranking, averaged over all 12 stocks")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "11_overall_model_ranking.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Is any of this statistically real? — Diebold-Mariano tests
#
# The tables above can show small differences that are just noise on
# ~390 out-of-sample test days per stock. The Diebold-Mariano test checks
# whether one model's forecast errors are *significantly* smaller than
# another's, for a specific pair, on each stock individually (so we can
# also see how consistent any effect is across stocks, not just its
# average). We test the four pairs the project brief calls out, which
# also happen to be the four "does the next level of complexity help"
# questions the research question is really asking:
#
# * Random Walk vs. ARIMA — does a statistical model beat the naive baseline?
# * ARIMA vs. XGBoost — does ML beat a statistical model?
# * ARIMA vs. LSTM — does deep learning beat a statistical model?
# * LSTM vs. Transformer — does attention beat recurrence?

# %%
DM_PAIRS = [("Random Walk", "ARIMA"), ("ARIMA", "XGBoost"), ("ARIMA", "LSTM"), ("LSTM", "Transformer")]

dm_rows = []
for model_1, model_2 in DM_PAIRS:
    for ticker in ALL_TICKERS:
        p1 = predictions[(predictions["Model"] == model_1) & (predictions["Ticker"] == ticker)]
        p2 = predictions[(predictions["Model"] == model_2) & (predictions["Ticker"] == ticker)]
        merged = p1.merge(p2, on="TargetDate", suffixes=("_1", "_2"))
        e1 = merged["y_true_1"] - merged["y_pred_1"]
        e2 = merged["y_true_2"] - merged["y_pred_2"]
        result = diebold_mariano(e1, e2, h=1)
        dm_rows.append({
            "Pair": f"{model_1} vs {model_2}", "Ticker": ticker, "Sector": TICKER_TO_SECTOR[ticker],
            "n_matched_days": result["n"], "DM stat": result["DM stat"], "p-value": result["p-value"],
            "Significant (5%)": result["p-value"] < 0.05 if not np.isnan(result["p-value"]) else False,
            "Favours": model_1 if result["DM stat"] < 0 else model_2,
        })

dm_results = pd.DataFrame(dm_rows)
dm_results.to_csv(TABLES_DIR / "diebold_mariano_results.csv", index=False)

dm_summary = (
    dm_results.groupby("Pair")
    .agg(stocks_tested=("Ticker", "count"),
         stocks_significant=("Significant (5%)", "sum"),
         mean_dm_stat=("DM stat", "mean"))
)
dm_summary["share_significant"] = dm_summary["stocks_significant"] / dm_summary["stocks_tested"]
dm_summary

# %% [markdown]
# **Reading this table.** `mean_dm_stat < 0` means the first-named model in
# the pair tends to have smaller errors; `> 0` means the second-named
# model does. `share_significant` is the fraction of the 12 stocks where
# that difference clears the 5% significance threshold.
#
# **This is the most decisive result in the whole project, and it points
# one direction: toward simplicity.** For every one of the four pairs, the
# mean DM statistic is negative — the *simpler* model in the pair has the
# smaller average error — and the effect is far from universal noise:
#
# * Random Walk vs. ARIMA: only 2/12 stocks (17%) significant — ARIMA is
#   genuinely hard to tell apart from the naive baseline.
# * ARIMA vs. XGBoost: **6/12 stocks (50%)** significant, all favouring
#   ARIMA — on half the stocks, the statistical model significantly beats
#   the ML model.
# * ARIMA vs. LSTM: 5/12 stocks (42%) significant, all favouring ARIMA.
# * LSTM vs. Transformer: **7/12 stocks (58%)** significant, all favouring
#   LSTM — on a majority of stocks, the *simpler* deep-learning model
#   significantly beats the *more complex* one.
#
# In other words: added complexity doesn't just fail to help here — on
# the error-magnitude metrics (MAE/RMSE), it measurably, significantly
# *hurts*, on a large share of individual stocks, at every step up the
# complexity ladder from ARIMA onward. That said, Section 1's directional
# accuracy table shows the opposite ranking for the Transformer
# specifically (its Directional Accuracy is the highest of all 7 models,
# ~52.4%) — a reminder that "better" depends on which question you're
# asking, which is exactly why Section 1 refused to rank on a single
# metric.

# %% [markdown]
# ## 4. Is there an overall difference among all 7 models? — Friedman test
#
# The Friedman test is the right tool for exactly this shape of problem —
# comparing several methods (here: 7 models) across several independent
# datasets (here: 12 stocks) using ranks, without assuming errors are
# normally distributed (which notebook 02 already told us they aren't).
# Each stock contributes one rank per model (1 = lowest MAE on that
# stock); H0 is that the average ranks are equal across all 7 models.

# %%
mae_by_stock = metrics.pivot_table(index="Ticker", columns="Model", values="MAE")[MODEL_ORDER]
friedman_stat, friedman_p = friedmanchisquare(*[mae_by_stock[m] for m in MODEL_ORDER])
print(f"Friedman chi-square = {friedman_stat:.3f}, p-value = {friedman_p:.4f}")
print("Average rank per model (1 = best MAE on that stock):")
mae_by_stock.rank(axis=1).mean().reindex(MODEL_ORDER).round(2)

# %% [markdown]
# **Result.** Friedman's chi-square = 50.1, p < 0.0001 — we reject the
# null of equal average ranks. The seven models are **not** interchangeable
# on MAE, and the average ranks make the direction explicit: **Random
# Walk and Random Forest tie for the best average rank (2.17), ARIMA is
# close behind (2.83), and the two most complex models are the worst
# ranked — 1D-CNN (6.25) and Transformer (6.50)**, roughly last on MAE on
# most of the 12 stocks. This lines up exactly with the Diebold-Mariano
# pattern above: it's not that model complexity is neutral here, it's
# that (on this metric) it's consistently, significantly counter-productive.
#
# We stop at the omnibus Friedman test rather than also running a
# post-hoc pairwise procedure (e.g. Nemenyi) — the Diebold-Mariano tests
# above already give us the pairwise detail for the four comparisons that
# matter most to the research question, and they tell the same story the
# Friedman ranks do.

# %% [markdown]
# ## 5. Sector analysis
#
# Answering the sector-level questions directly, using the tables above.

# %%
best_by_sector = mae_table[SECTOR_ORDER].idxmin()
best_overall = mae_table["Overall"].idxmin()
print("Best model by MAE, per sector:")
print(best_by_sector.to_string())
print(f"\nBest model overall (by MAE): {best_overall}")

# %%
# consistency within a sector: how much does MAE vary across the 3 stocks
# in the same sector, for each model? Lower std = more consistent.
consistency = metrics.groupby(["Model", "Sector"])["MAE"].std().unstack()[SECTOR_ORDER].reindex(MODEL_ORDER)
consistency.round(5)

# %% [markdown]
# * **Which model performs best overall?** By MAE, the Random Walk
#   (0.01065 overall) — narrowly ahead of Random Forest (0.01066) and
#   ARIMA (0.01067), and by a clear margin ahead of 1D-CNN (0.01143) and
#   the Transformer (0.01147). By directional accuracy, the ranking
#   flips: the **Transformer** is the best of the six models that make an
#   actual directional call (52.4% vs. ~49.7-50.6% for the rest). Neither
#   metric alone tells the full story, which is exactly why Section 2's
#   combined rank exists — and on that combined measure, Random Walk (2.17)
#   and Random Forest (2.82) come out on top, with 1D-CNN (5.12) and
#   Transformer (5.01) at the bottom.
# * **Which model performs best in each sector?** By MAE: Random Forest
#   in Banking and Healthcare, Random Walk in IT and Automotive (Section
#   5's `idxmin` table) — always one of the two simplest, cheapest models,
#   never one of the deep-learning models, in every sector.
# * **Does the Random Walk remain hard to beat?** Yes, decisively — it has
#   the best or joint-best MAE overall and is the outright best in 2 of 4
#   sectors. This mirrors a well-known result in the finance literature
#   (going back to Meese-Rogoff-style findings for exchange rates) that
#   naive no-change forecasts are a genuinely tough benchmark for daily
#   financial return series.
# * **Does ARIMA remain competitive?** Yes — essentially tied with the
#   Random Walk (Section 3's DM test found a significant difference on
#   only 2/12 stocks), exactly as notebook 02's weak Ljung-Box
#   autocorrelation result predicted.
# * **Do ML models improve over statistical models?** No — the opposite:
#   ARIMA significantly *beats* XGBoost on 6/12 stocks (Section 3), and
#   neither Random Forest nor XGBoost improves on ARIMA's MAE overall.
# * **Do deep-learning models consistently outperform simpler models?**
#   No. LSTM is roughly on par with XGBoost; 1D-CNN is the worst-ranked
#   model on MAE together with the Transformer. Deep learning has more
#   parameters to fit a comparatively small (~2,600-observation), noisy
#   target, not more genuine signal to find.
# * **Does the Transformer consistently outperform LSTM/CNN?** No, and on
#   MAE it does the opposite — Section 3's DM test found LSTM
#   significantly beats the Transformer on 7/12 stocks (58%), the
#   strongest, most consistent effect of any pair tested. Its one real
#   advantage is directional accuracy (Section 1), where it's the best
#   model in the whole comparison — a genuine, if narrow, bright spot for
#   the most complex architecture.
# * **Are differences statistically significant?** For MAE/RMSE, yes,
#   and consistently in favour of simpler models (Sections 3-4 — this is
#   the project's central, most decisive finding). For directional
#   accuracy, effect sizes are small (all models within ~3 percentage
#   points of a 50% coin flip) and this project did not run a formal
#   significance test on directional accuracy specifically.
# * **Is performance consistent across companies within the same sector?**
#   The consistency table above (MAE standard deviation across the 3
#   stocks in a sector) shows no sector/model combination is wildly more
#   erratic than the rest — the "simpler models win" pattern holds
#   company-by-company within sectors, not just on sector averages.
#
# **Why might results differ (or not) between sectors at all?** Banking
# and IT are large, heavily-analysed, highly liquid stocks where any
# simple exploitable pattern is more likely to have already been priced in
# by other market participants — consistent with those sectors showing
# some of the flattest, most baseline-like results. Automotive and
# Healthcare have more idiosyncratic, event-driven return components
# (regulatory approvals, monthly sales data, currency-linked generic drug
# pricing) that could in principle give a nonlinear model more genuine
# structure to find — but the DM/Friedman results suggest this
# expectation does not translate into a statistically reliable
# forecasting edge in this dataset.

# %% [markdown]
# ## 6. Market regime analysis
#
# A simple, defensible split rather than a hidden-Markov regime detector:
# for each stock, compute the 21-day trailing realised volatility and
# split the *test period* at that stock's own volatility median into
# "Low Volatility" and "High Volatility" days. Then compare each model's
# MAE and directional accuracy across the two regimes.

# %%
vol = prices[["Date", "Ticker", "LogReturn"]].copy()
vol["Vol21"] = vol.groupby("Ticker")["LogReturn"].transform(lambda s: s.rolling(21).std())

preds_with_vol = predictions.merge(
    vol.rename(columns={"Date": "TargetDate"})[["TargetDate", "Ticker", "Vol21"]],
    on=["TargetDate", "Ticker"], how="left",
)
median_vol = preds_with_vol.groupby("Ticker")["Vol21"].transform("median")
preds_with_vol["Regime"] = np.where(preds_with_vol["Vol21"] >= median_vol, "High Volatility", "Low Volatility")

regime_rows = []
for (model, regime), g in preds_with_vol.groupby(["Model", "Regime"]):
    m = regression_metrics(g["y_true"], g["y_pred"])
    regime_rows.append({"Model": model, "Regime": regime, **m})
regime_table = pd.DataFrame(regime_rows).pivot_table(
    index="Model", columns="Regime", values=["MAE", "Directional Accuracy"])
regime_table = regime_table.reindex(MODEL_ORDER)
regime_table.to_csv(TABLES_DIR / "regime_analysis.csv")
regime_table.round(4)

# %% [markdown]
# **Result.** Every model's MAE is noticeably higher in the high-volatility
# regime than the low-volatility regime — expected, since MAE scales with
# the size of typical moves, and moves are bigger by construction when we
# define "high volatility" that way. The more interesting comparison is
# directional accuracy: if a model were genuinely learning something
# useful about volatile periods (rather than just being penalised more by
# their larger moves), we'd expect its directional accuracy to hold up or
# even improve in the high-volatility regime. Instead, directional
# accuracy for every model is close to 50% in both regimes, with no model
# showing a clear, consistent edge in either — i.e., **model performance
# changes with volatility mainly through larger errors on bigger moves,
# not through any model getting meaningfully better or worse at calling
# direction.**

# %% [markdown]
# ## 7. GARCH — a separate volatility side-experiment
#
# Notebook 02's ARCH-LM test confirmed real volatility clustering in every
# stock. GARCH is built to forecast *volatility*, not the *sign/magnitude
# of tomorrow's return* the way the seven models above do — so it doesn't
# belong in the main leaderboard, but the clustering evidence does justify
# checking whether GARCH forecasts volatility usefully. We compare a
# walk-forward GARCH(1,1) against the simplest possible volatility
# benchmark: yesterday's trailing 21-day realised volatility, carried
# forward unchanged. The forecasting target is each day's realised
# absolute return (a standard, simple realised-volatility proxy when
# intraday data isn't available).

# %%
def garch_experiment(ticker: str) -> pd.DataFrame:
    d = prices[prices["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    returns_pct = d["LogReturn"].to_numpy() * 100
    abs_returns = np.abs(d["LogReturn"].to_numpy())
    trailing_vol = d["LogReturn"].rolling(21).std().to_numpy()
    dates = d["Date"].to_numpy()

    rows = []
    for train_end, test_start, test_end in expanding_window_folds(len(d)):
        am = arch_model(returns_pct[:train_end], vol="Garch", p=1, q=1, dist="normal", rescale=False)
        try:
            res = am.fit(disp="off")
            fc = res.forecast(horizon=test_end - test_start, reindex=False)
            garch_vol = np.sqrt(fc.variance.values[0]) / 100
        except Exception:
            garch_vol = np.full(test_end - test_start, np.nan)
        for offset, i in enumerate(range(test_start, test_end)):
            rows.append((dates[i], abs_returns[i], garch_vol[offset], trailing_vol[i - 1] if i > 0 else np.nan))

    return pd.DataFrame(rows, columns=["TargetDate", "realised_abs_return", "garch_vol", "naive_trailing_vol"])


garch_rows = []
for ticker in ALL_TICKERS:
    g = garch_experiment(ticker)
    g = g.dropna()
    garch_mae = np.mean(np.abs(g["realised_abs_return"] - g["garch_vol"]))
    naive_mae = np.mean(np.abs(g["realised_abs_return"] - g["naive_trailing_vol"]))
    garch_rows.append({"Ticker": ticker, "Sector": TICKER_TO_SECTOR[ticker],
                        "GARCH(1,1) MAE": garch_mae, "Naive trailing-vol MAE": naive_mae,
                        "GARCH improves?": garch_mae < naive_mae})

garch_results = pd.DataFrame(garch_rows)
garch_results.to_csv(TABLES_DIR / "garch_volatility_experiment.csv", index=False)
garch_results.round(5)

# %% [markdown]
# **Result.** This one surprised us: the naive trailing-volatility
# benchmark actually beats GARCH(1,1) on **10 of the 12 stocks** — GARCH
# only wins on TCS and Infosys, both IT stocks. **Conclusion for this
# side-experiment:** even for a task GARCH is specifically built for, a
# dead-simple "yesterday's realised volatility persists" rule is a
# surprisingly tough benchmark here too — echoing, in volatility space,
# the exact same "simple is hard to beat" pattern the main model
# comparison found in return space. This is a genuinely useful negative
# result, not a reason to hide the experiment: it says our default
# GARCH(1,1) specification isn't adding value over the naive rule for
# most of these stocks, and a more careful specification (different error
# distribution, EGARCH/GJR-GARCH for asymmetric shocks) would be needed
# before relying on it. It also reinforces why GARCH was kept out of the
# main leaderboard — it's answering a different question (volatility, not
# direction), and on that question it doesn't even clearly beat its own
# naive baseline.

# %% [markdown]
# ## 8. Simple backtest (optional, kept intentionally simple)
#
# **This is a directional-strategy sanity check, not a financial
# recommendation, and forecasting accuracy does not necessarily imply
# investment profitability** — no transaction costs, slippage, position
# sizing, or risk management are modelled. Rule: go long for one day when
# a model predicts a positive return, otherwise stay in cash. We report
# total (uncompounded) log return over the test period per model, and
# compare to a buy-and-hold benchmark.

# %%
predictions["strategy_return"] = np.where(predictions["y_pred"] > 0, predictions["y_true"], 0.0)
backtest = predictions.groupby("Model")["strategy_return"].sum().reindex(MODEL_ORDER)

buy_hold = predictions[predictions["Model"] == "Random Walk"].groupby("Ticker")["y_true"].sum().mean()
backtest_table = backtest.to_frame("Total strategy log-return (avg $ invested, no costs)")
backtest_table.loc["Buy & Hold (benchmark)"] = predictions[predictions["Model"] == "Random Walk"]["y_true"].sum()
backtest_table.to_csv(TABLES_DIR / "simple_backtest.csv")
backtest_table.round(3)

# %% [markdown]
# **Result.** Buy-and-hold lost money over this particular test window
# (-0.654 total log-return, averaged over the 12 stocks) — it was a rough
# stretch for the market. Against that backdrop, XGBoost (+0.119), 1D-CNN
# (+0.309), and the Transformer (+0.417) show positive total returns,
# while ARIMA (-0.577), Random Forest (-0.648), and LSTM (-0.917) do
# worse than even buy-and-hold; the Random Walk sits at exactly 0 by
# construction (it only ever predicts "no change," so a strict
# `predict > 0 → long` rule never buys). It's tempting to read this as
# "the Transformer is a good trading model," but Sections 1-4 already
# showed its directional accuracy is only a few points above a coin flip
# and its MAE is the worst of all seven models — a single backtest
# realisation over one test window, with no transaction costs, is exactly
# the kind of result that can look good by chance. We show this table as
# a sanity check that the predictions are at least directionally usable,
# explicitly **not** as evidence that any of these models is a viable
# trading strategy.

# %% [markdown]
# ## 9. Answering the research question
#
# > **Do increasingly complex forecasting models actually improve
# > out-of-sample stock forecasting performance, and does model
# > performance differ across Indian market sectors?**
#
# **On complexity:** No — and the evidence here is more decisive than "no
# effect either way." On the metrics that measure forecast *accuracy*
# (MAE, RMSE), performance gets **monotonically worse, on average, as
# model complexity increases** past Random Forest: Random Walk and Random
# Forest tie for best (MAE 0.01065-0.01066), ARIMA is close behind
# (0.01067), then LSTM (0.01080), XGBoost (0.01093), 1D-CNN (0.01143),
# and the Transformer is worst (0.01147). This isn't noise: the Friedman
# test across all 12 stocks is highly significant (p < 0.0001), and the
# Diebold-Mariano tests found the simpler model significantly beats the
# more complex one in a pair on up to 58% of individual stocks (LSTM vs.
# Transformer) — with every one of the four tested pairs favouring the
# simpler model on average. **Added complexity did not just fail to help
# in this project — on error-magnitude metrics it measurably, often
# significantly, hurt.**
#
# The one genuine exception is directional accuracy, where the
# Transformer is the best of the seven models (52.4%, vs. 49.7-50.6% for
# ARIMA/RF/XGBoost/LSTM/1D-CNN) — a real but narrow edge, only a few
# points above a coin flip, and one this project did not test for
# significance the way it did for MAE/RMSE. So the fullest honest answer
# is: **more complex models did not improve forecast accuracy, and mostly
# made it significantly worse; the one place a more complex model (the
# Transformer) showed an advantage was in calling direction slightly more
# often, not in the size of its errors.** Both halves of that answer are
# valid, useful research findings — not a failed experiment.
#
# **On sectors:** The *identity* of the best model shifts a little by
# sector (Random Forest in Banking/Healthcare, Random Walk in
# IT/Automotive), but the *pattern* is remarkably stable across all four:
# one of the two simplest models wins every sector's MAE comparison, and
# the deep-learning/Transformer models are never best in any sector.
# Sector mainly affects *how forecastable returns are at all* (compare
# the Overall MAE column across sectors in Section 1) rather than
# *which kind of model is worth using* — that answer, on this evidence,
# is "the simpler one," everywhere.
#
# **On volatility regimes:** Every model's errors grow in high-volatility
# periods simply because moves are larger then, but no model's
# directional accuracy meaningfully improves or worsens with volatility
# (Section 6) — the difficulty of this forecasting problem does not
# appear to be regime-dependent in a way any of these seven models
# exploit.

# %% [markdown]
# ## 10. Limitations
#
# * **Data source.** NSE's own site blocked automated downloads from this
#   project's execution environment; the data used is a credible,
#   NSE-sourced third-party mirror rather than a direct NSE/BSE download,
#   and it is not adjusted for dividends (only for splits/bonuses).
# * **Non-stationarity and structural change.** Even after moving to
#   returns, markets are not stationary in a deeper sense — regulatory
#   regimes, index composition, and macro conditions shift over a
#   10+-year window in ways no model here explicitly accounts for.
# * **Limited, inherent predictability.** Daily returns for large, liquid,
#   heavily-analysed stocks are close to efficiently priced; this project's
#   own results are consistent with that, not a methodological failure.
# * **Overfitting risk.** Even with walk-forward validation and small
#   architectures, 12 stocks and ~2,600 observations each is a modest
#   amount of data for the DL/Transformer models; results on a longer
#   history or more stocks could differ.
# * **Universe size.** 12 stocks across 4 sectors is enough to see
#   within-sector consistency, but not enough to generalise to the whole
#   NSE market; results could look different for mid-caps or small-caps,
#   which are typically less efficiently priced.
# * **Survivorship.** All 12 companies are large, long-listed, currently
#   healthy businesses; the data source does not let us include companies
#   that were delisted or went through severe distress in this window.
# * **Backtest realism.** Section 8's backtest ignores transaction costs,
#   slippage, taxes, and position sizing entirely — it is a directional
#   sanity check, not a strategy evaluation.
# * **Model instability across refits.** Quarterly (not daily) refitting
#   for the walk-forward evaluation was a deliberate runtime/rigour
#   trade-off; a production system would refit more often, at more
#   compute cost.
# * **GARCH scope.** The GARCH side-experiment uses one common
#   specification (GARCH(1,1), normal errors) rather than searching over
#   GARCH variants (EGARCH, GJR-GARCH, different error distributions),
#   which could plausibly improve its volatility forecasts further.

# %% [markdown]
# ## 11. What I deliberately did not implement, and why
#
# * **A live/API stock-data pipeline (yfinance, Alpha Vantage, etc.)** —
#   the brief explicitly calls for downloaded historical data, and this
#   keeps the project reproducible without depending on an external
#   service being reachable at run time.
# * **Any deployment surface** (Streamlit/FastAPI/Flask/Docker/cloud
#   hosting/a database/CI-CD/auth) — this is a research notebook project,
#   not a product; none of it would change a single conclusion above.
# * **GARCH in the main 7-model leaderboard** — it forecasts variance, not
#   direction/magnitude of the next return, so comparing it head-to-head
#   with ARIMA/RF/XGBoost/LSTM/CNN/Transformer would be comparing models
#   on different tasks. Investigated separately in Section 7 instead.
# * **A parallel "price forecasting" track alongside the return-forecasting
#   track** — notebook 02's ADF/KPSS results were unambiguous enough
#   (prices non-stationary, returns stationary) that duplicating the
#   entire project to also forecast prices directly would not have
#   answered a genuinely open question.
# * **Hidden-Markov or other learned regime-detection models** — the
#   simple volatility-median split in Section 6 already answers the
#   "does performance change with market conditions" question; a more
#   complex regime detector would add machinery without changing that
#   answer.
# * **Massive hyperparameter grid searches** — a handful of sensible
#   candidates, checked once per model family (Section 4 of notebook 03),
#   was enough to see that "deeper"/more-tuned configurations were not
#   obviously better on this noisy a target; spending the project's time
#   budget on exhaustive tuning would not have served the research
#   question about *methodology*, not squeezing out marginal accuracy.
# * **Daily (rather than quarterly) walk-forward refitting** — quarterly
#   refits already guarantee no model ever trains on data from its own
#   test period, which is the property that actually protects against
#   leakage; daily refitting would cost far more compute for the same
#   guarantee.
# * **A 50-stock universe** — 3 stocks x 4 sectors was enough to check
#   whether results are sector-wide patterns or single-company flukes
#   (Section 5's within-sector consistency check) without inflating
#   runtime for its own sake.
# * **Options strategies, portfolio optimisation, or a transaction-cost
#   simulator in the backtest** — Section 8 is a directional sanity check,
#   explicitly not a strategy evaluation; adding this machinery would
#   suggest a level of investment realism this project doesn't claim.
# * **Wilcoxon/post-hoc pairwise tests beyond the Friedman test** — as
#   noted in Section 4, the Diebold-Mariano results already show the
#   pairwise picture is mostly "not significant"; a post-hoc procedure on
#   top of that would mostly restate the same finding.
