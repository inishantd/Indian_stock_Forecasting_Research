# A Comparative Evaluation of Statistical, Machine Learning, Deep Learning, and Transformer Models for Sector-Specific Stock Forecasting in Indian Financial Markets

A notebook-based research project — not an application. It compares seven
forecasting models of increasing complexity on daily stock-return
forecasting across four sectors of the Indian equity market, and asks
whether the added complexity is actually worth it.

## Research question

> Do increasingly complex forecasting models actually improve out-of-sample
> stock forecasting performance, and does model performance differ across
> Indian market sectors?

## Key finding

**No — and the effect runs the other way.** On error-magnitude metrics
(MAE, RMSE), forecasting accuracy gets worse, on average, as model
complexity increases past Random Forest. A Friedman test across all 12
stocks confirms this is not noise (p < 0.0001), and Diebold-Mariano tests
show the simpler model in a pair significantly beats the more complex one
on up to 58% of individual stocks (LSTM vs. Transformer). The one
exception is directional accuracy, where the Transformer is narrowly the
best of the seven models (52.4% vs. a 50% coin flip) — a real but small
edge that doesn't show up in its error magnitude. Full detail, numbers,
and discussion are in `notebooks/06_model_comparison.ipynb`.

## Dataset

- **Source:** NSE (National Stock Exchange of India) end-of-day bhavcopy
  data, redistributed per-stock by the open-source
  [`eod2_data`](https://github.com/BennyThadikaran/eod2_data) project
  (MIT-licensed, updated weekly from NSE's own published reports). NSE's
  own site blocks automated downloads from this project's execution
  environment, so this NSE-sourced mirror was used instead — see
  Limitations.
- **Download date:** 2026-09-10.
- **Date range used:** 2016-01-01 to 2026-08-31 (~10.7 years, ~2,638
  trading days per stock).
- **Variables:** Date, Open, High, Low, Close, Volume. Close is
  split/bonus-adjusted but not dividend-adjusted (no separate "Adjusted
  Close" in this source).
- **Companies and sectors** (3 per sector, chosen for size, liquidity, and
  long listing history):

  | Sector | Companies |
  |---|---|
  | Banking | HDFC Bank, ICICI Bank, State Bank of India |
  | IT | TCS, Infosys, Wipro |
  | Healthcare | Sun Pharma, Dr. Reddy's Laboratories, Cipla |
  | Automotive | Maruti Suzuki, Bajaj Auto, Mahindra & Mahindra |

Raw CSVs live in `data/raw/<sector>/<TICKER>.csv`; the cleaned, combined
dataset used by every notebook from 01 onward is written to
`data/processed/prices_processed.csv`.

## Methodology

- **Target variable:** daily **log returns**, not price levels. Notebook
  02 shows ADF/KPSS tests agree prices are non-stationary and returns are
  stationary — forecasting non-stationary price levels directly would let
  any model look good just by tracking the trend.
- **Validation:** strictly chronological, walk-forward, expanding-window.
  The final 15% of each stock's history is held out; predictions are made
  one quarter (63 trading days) at a time, after which that quarter is
  folded into the training window before the next refit. No model ever
  trains on data from its own test period. Scaling/feature stats are
  fit on training data only, per fold.
- **Feature engineering** (Random Forest, XGBoost, LSTM, 1D-CNN,
  Transformer): lagged returns, rolling mean/std of returns, momentum,
  volume change, high-low range, and 14-day RSI — all computed only from
  information available up to the prediction day (`src/features.py`).

## Models (increasing complexity)

1. Random Walk / no-change baseline
2. ARIMA
3. Random Forest
4. XGBoost
5. LSTM
6. 1D-CNN
7. Transformer (small, single-encoder-layer)

## Evaluation metrics

MAE, RMSE, and Directional Accuracy on genuinely out-of-sample
predictions, compared per model per sector. Directional Accuracy is
undefined (not 0%) for the Random Walk baseline, since predicting exactly
zero return makes no directional call at all — see `src/eval_utils.py`.
R² is reported for reference only; MAPE is deliberately not used (returns
near zero make it unstable and misleading).

## Statistical tests

- **ADF / KPSS** — stationarity of prices vs. returns (notebook 02)
- **Ljung-Box** — autocorrelation in returns (notebook 02)
- **Jarque-Bera** — normality / fat tails of returns (notebook 02)
- **ARCH-LM (Engle)** — volatility clustering (notebook 02)
- **Diebold-Mariano** — pairwise forecast-accuracy significance (notebook 06)
- **Friedman** — omnibus test across all 7 models x 12 stocks (notebook 06)

## GARCH

Included only as a **small, separate volatility-forecasting
side-experiment** (notebook 06, Section 7), not as an eighth row in the
main model comparison — GARCH forecasts variance, not the direction or
magnitude of the next return, so it isn't answering the same question as
the other seven models. Result: a naive trailing-volatility benchmark
actually beats GARCH(1,1) on 10 of 12 stocks, a useful negative result in
its own right.

## Simple backtest

An optional, deliberately simple long/cash directional strategy (notebook
06, Section 8) — no transaction costs, slippage, or position sizing.
**Forecasting accuracy does not necessarily imply investment
profitability**, and this table should be read as a sanity check, not a
strategy evaluation.

## Limitations

Third-party (not direct NSE/BSE) data source; no dividend adjustment;
inherent limits to daily-return predictability for large, liquid,
heavily-analysed stocks; modest sample size (~2,600 observations/stock)
for the deep-learning models; 12-stock, 4-sector universe rather than the
full market; survivorship (all 12 companies are currently healthy,
long-listed businesses); quarterly (not daily) walk-forward refitting;
single GARCH(1,1)/normal specification; no transaction costs in the
backtest. Full discussion in notebook 06, Section 10.

## What was deliberately left out, and why

No API-based data pipeline, no deployment (no Streamlit/FastAPI/Docker/
cloud/database/CI-CD/auth), GARCH kept out of the main leaderboard, no
parallel price-forecasting track, no hidden-Markov regime detection, no
massive hyperparameter search, no daily refitting, no 50-stock universe,
no options/portfolio-optimisation backtest, no post-hoc pairwise tests
beyond Friedman. Full reasoning for each in notebook 06, Section 11.

## Project structure

```text
Indian-Stock-Forecasting-Research/
├── data/
│   ├── raw/{banking,it,healthcare,automotive}/*.csv   # per-stock raw OHLCV
│   └── processed/prices_processed.csv                 # cleaned, combined dataset (from notebook 01)
├── notebooks/
│   ├── 01_data_and_eda.ipynb            # load, clean, document, EDA
│   ├── 02_statistical_analysis.ipynb    # ADF/KPSS/Ljung-Box/JB/ARCH, price-vs-return decision
│   ├── 03_baseline_arima_ml.ipynb       # Random Walk, ARIMA, Random Forest, XGBoost
│   ├── 04_deep_learning.ipynb           # LSTM, 1D-CNN
│   ├── 05_transformer.ipynb             # small Transformer
│   └── 06_model_comparison.ipynb        # comparison tables, DM/Friedman tests, sector &
│                                         # regime analysis, GARCH side-experiment, backtest,
│                                         # final conclusions, limitations
├── results/
│   ├── figures/          # all saved plots (.png)
│   └── tables/           # all saved result tables (.csv) — predictions, metrics, test results
├── src/
│   ├── config.py         # tickers, sectors, paths, date range
│   ├── features.py       # tabular + sequence feature engineering (leakage-safe)
│   ├── eval_utils.py     # metrics, walk-forward splitter, Diebold-Mariano test
│   └── dl_utils.py       # shared PyTorch training/evaluation loop (LSTM/CNN/Transformer)
├── requirements.txt
└── README.md
```

`*.py` files paired with each notebook (via [Jupytext](https://jupytext.readthedocs.io/))
are also included in `notebooks/` — they're the same content in a
plain-Python, git-diff-friendly format, kept in sync with the `.ipynb`
files.

## Reproducing this project

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 01_data_and_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace 02_statistical_analysis.ipynb
jupyter nbconvert --to notebook --execute --inplace 03_baseline_arima_ml.ipynb
jupyter nbconvert --to notebook --execute --inplace 04_deep_learning.ipynb
jupyter nbconvert --to notebook --execute --inplace 05_transformer.ipynb
jupyter nbconvert --to notebook --execute --inplace 06_model_comparison.ipynb
```

Notebooks must run in order (01 → 06): each later notebook reads a
processed dataset or results table written by an earlier one. Total
runtime is roughly 20-25 minutes on a standard CPU (notebook 05, the
Transformer, is the slowest at ~8-9 minutes; everything else is under 4
minutes). No GPU or internet access is required at run time — all data is
already downloaded and stored locally under `data/raw/`.
