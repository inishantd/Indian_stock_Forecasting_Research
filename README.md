# A Comparative Evaluation of Statistical, Machine Learning, Deep Learning, and Transformer Models for Sector-Specific Stock Forecasting in Indian Financial Markets

A reproducible research study comparing **7 forecasting models** — from a random-walk baseline to a Transformer — on **daily log-return forecasting** for **12 large-cap NSE stocks across 4 sectors** (2016–2026), using strict walk-forward validation and formal statistical significance testing.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c) ![statsmodels](https://img.shields.io/badge/statsmodels-0.14-lightgrey) ![XGBoost](https://img.shields.io/badge/XGBoost-2.x-green) ![License](https://img.shields.io/badge/data-MIT%20(eod2__data)-yellow)

---

## TL;DR

> **Research question:** Do increasingly complex models improve out-of-sample daily stock-return forecasts, and does the answer differ by sector?

**No model beats a naive "no-change" forecast in any meaningful way.**

- On MAE/RMSE, Random Walk, ARIMA, Random Forest, LSTM, 1D-CNN and the Transformer all land within **~0.2%** of each other (MAE ≈ 0.0106).
- **XGBoost is the one clear loser.** It is significantly worse than ARIMA on **9 of 12 stocks** (Diebold–Mariano, 5%), and the Friedman test across all 7 models × 12 stocks rejects equal ranks (χ² = 14.9, **p = 0.021**), mostly because XGBoost ranks last.
- **Directional accuracy is a coin flip for every model** (49.4%–50.7%). None differs from 50% under a binomial test (all p > 0.37, n ≈ 4,700 predictions per model).
- A GARCH(1,1) volatility side-experiment loses to a naive trailing-volatility rule on **10 of 12 stocks**.

Extra model capacity bought no extra accuracy. On noisy, near-efficient daily returns, the honest baseline is very hard to beat. That is a known result in financial econometrics, and this project confirms it on Indian large caps with a leakage-safe protocol.

---

## Dataset

| | |
|---|---|
| **Source** | NSE end-of-day bhavcopy data via the open-source [`eod2_data`](https://github.com/BennyThadikaran/eod2_data) mirror (MIT) |
| **Period** | 2016-01-01 → 2026-08-31 (~2,638 trading days per stock) |
| **Fields** | Date, Open, High, Low, Close, Volume (split/bonus adjusted, **not** dividend adjusted) |
| **Universe** | 12 stocks, 4 sectors |

| Sector | Stocks |
|---|---|
| Banking | HDFC Bank, ICICI Bank, SBI |
| IT | TCS, Infosys, Wipro |
| Healthcare | Sun Pharma, Dr. Reddy's, Cipla |
| Automotive | Maruti Suzuki, Bajaj Auto, M&M |

All data is included in `data/raw/`, so no internet or API keys are needed.

---

## Methodology

**1. Target selection is driven by the statistics.** ADF and KPSS agree for 12/12 stocks that prices are non-stationary and log returns are stationary, so every model forecasts **next-day log return**.

**2. Diagnostics shape the modelling choices** (notebook 02):

| Test | Finding | Decision |
|---|---|---|
| ADF + KPSS | Prices non-stationary, returns stationary | Forecast returns, not prices |
| Ljung–Box | Weak autocorrelation in ~half the stocks | Expect little linear signal for ARIMA |
| Jarque–Bera | Fat tails in all 12 (excess kurtosis 3.9–13.2) | MAE and directional accuracy are the primary metrics, not RMSE alone |
| ARCH-LM | Volatility clustering in all 12 | Run a separate GARCH volatility experiment |

**3. Leakage-safe validation**
- Chronological **expanding-window walk-forward**. The last 15% of each series is the test set, and models are refit every quarter (63 trading days).
- Features at day *t* use only information up to *t*'s close. The target is *t+1* (checked explicitly in notebook 03).
- Scalers and early-stopping validation slices are fit **only on each fold's training data**.
- Hyperparameters are chosen on a pre-test validation slice, never on the test set.

**4. Models** (increasing complexity)

| # | Model | Inputs | Notes |
|---|---|---|---|
| 1 | Random Walk | — | Predicts 0 return |
| 2 | ARIMA | Returns | Order picked by AIC from a small candidate set |
| 3 | Random Forest | 16 engineered features | 200 trees, depth 4 |
| 4 | XGBoost | 16 engineered features | 200 trees, depth 3, lr 0.05 |
| 5 | LSTM | 30-day × 3-channel window | 1 layer, 32 units (4.8k params) |
| 6 | 1D-CNN | 30-day × 3-channel window | 2 conv layers (1.8k params) |
| 7 | Transformer | 30-day × 3-channel window | 1 encoder layer, 4 heads, learned positional embedding |

The engineered features are return lags 1–5, rolling mean and std (5/10/21), 5- and 10-day momentum, volume change, high-low range and RSI-14.

**5. Evaluation.** MAE, RMSE and directional accuracy are computed out-of-sample. Significance comes from **Diebold–Mariano** pairwise tests per stock and a **Friedman** omnibus test across stocks. There is also a volatility-regime split, the GARCH side-experiment, and a naive long/cash backtest.

---

## Results (reproduced run)

### MAE by model and sector (lower is better)

| Model | Banking | IT | Healthcare | Automotive | **Overall** |
|---|---|---|---|---|---|
| Random Walk | 0.00911 | 0.01203 | 0.00971 | **0.01176** | 0.01065 |
| ARIMA | 0.00912 | 0.01211 | 0.00971 | 0.01176 | 0.01067 |
| Random Forest | **0.00911** | 0.01210 | 0.00960 | 0.01182 | 0.01066 |
| XGBoost | 0.00923 | 0.01247 | 0.00988 | 0.01221 | 0.01095 |
| LSTM | 0.00914 | 0.01204 | **0.00959** | 0.01177 | **0.01064** |
| 1D-CNN | 0.00915 | **0.01202** | 0.00960 | 0.01179 | 0.01064 |
| Transformer | 0.00919 | 0.01202 | 0.00961 | 0.01182 | 0.01066 |

### Statistical significance

| Diebold–Mariano pair | Stocks significant at 5% (of 12) | Direction |
|---|---|---|
| Random Walk vs ARIMA | 3 | All favour Random Walk |
| ARIMA vs XGBoost | **9** | All favour ARIMA |
| ARIMA vs LSTM | 2 | 1 each way |
| LSTM vs Transformer | 0 | — |

- The Friedman test on MAE ranks gives χ² = 14.86, p = 0.021. Average ranks: LSTM 3.17, Random Walk 3.33, 1D-CNN 3.42, RF 3.58, ARIMA 4.25, Transformer 4.25, XGBoost 6.00.
- Directional accuracy for every model is 49.4%–50.7%, and no model differs significantly from 50%.

### Other findings
- **Sectors:** The winning model changes by sector (RF in Banking, 1D-CNN in IT, LSTM in Healthcare, Random Walk in Automotive), but the margins are in the 4th–5th decimal place. Sector affects **how predictable** returns are (Banking MAE ≈ 0.0091 vs IT ≈ 0.0120) far more than **which model** to use.
- **Volatility regimes:** Every model's errors grow in high-volatility periods. No model gains a directional edge in either regime.
- **GARCH(1,1):** A naive 21-day trailing volatility beats it on 10/12 stocks. GARCH wins only on TCS and Infosys.
- **Backtest** (long/cash, no costs): buy-and-hold lost money over the test window (−0.65 summed log-return). LSTM was the only model with a positive total (+0.20). Given the coin-flip directional accuracy, treat that as noise, not a trading signal.

---

## Limitations
- Third-party NSE mirror, and prices are not dividend-adjusted.
- 12 large, liquid, surviving companies, so there is survivorship bias and the results may not carry over to mid or small caps.
- ~2,600 observations per stock is small for deep models.
- Refits are quarterly, not daily. **ARIMA and GARCH produce multi-step (up to 63-day-ahead) forecasts from each quarterly fit.** The feature-based models, by contrast, use fresh inputs every day. This handicaps the two statistical models slightly.
- Deep-learning results are sensitive to training-loop details (target scaling, early stopping). An earlier run of this project with a different loop showed the DL models noticeably *worse* than the baselines, not level with them.
- The backtest ignores transaction costs, slippage and taxes.

---

