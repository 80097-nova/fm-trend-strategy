# Trend-filtered 80/20 portfolio: interactive backtest

- **Live app:** https://fm-trend-strategy.streamlit.app/
- **Source code:** https://github.com/80097-nova/fm-trend-strategy
- **Course:** 2270 Financial Modeling (Nova SBE), group assignment.

This app lets you change the main assumptions of our portfolio strategy and see the effect on
performance and on the comparison with the benchmarks. It uses the same data, rules and
calculations as the notebook submitted with our prospectus.

## The strategy in one paragraph

A global portfolio of five ETFs with base weights SPY 50%, EFA 20%, EEM 10%, IEF 10% and GLD 10%
(80% equities, 20% bonds and gold), rebalanced monthly. Each month every ETF gets a trend score
from its past returns over several horizons; it keeps that share of its base weight and the rest
is held in 3-month US T-bills. Benchmarks: the same five ETFs at fixed weights (static 80/20) and SPY.

## What you can change

| Control | Effect |
| --- | --- |
| Trend filter | The full strategy (several horizons), a 12-month filter only, or no filter (static 80/20) |
| Horizons in the trend score | Any mix of 1, 3, 6 and 12 months |
| Transaction cost | 0 to 100 bps per 100% turnover, paid by the strategy and the static 80/20 |
| Start of the evaluation window | Any month from Jan 2006 that leaves at least 3 years of results |
| Benchmark for alpha, beta and IR | SPY or the static 80/20 |

The app shows the growth of 1 invested against both benchmarks (all series start at 1), the
performance measures (annualized return, volatility, Sharpe ratio, alpha, beta, information ratio,
maximum drawdown, turnover), drawdowns over time and how much of the portfolio sits in T-bills.

## Same numbers as the notebook

`core.py` holds the notebook's own functions (data download, data checks, backtest engine, Sharpe
ratio, maximum drawdown), copied without changes; each block says which notebook section it comes
from. `app.py` only reads the user's choices, calls those functions and draws the results. With the
default settings the app reproduces the notebook's main results. The backtest always runs on the full
sample (Jan 2006 to Dec 2025); the start-date control only changes the window shown and measured.

## Data

- Monthly prices of SPY, EFA, EEM, IEF and GLD from Yahoo Finance through `yfinance`, adjusted for
  dividends and splits. Dec 2004 to Dec 2025; 2005 only starts the 12-month signal.
- The 3-month T-bill rate (TB3MS) from FRED's public CSV, converted to a monthly return.
- The download is cached for a day. If Yahoo Finance or FRED cannot be reached (Yahoo sometimes
  blocks cloud servers), the app uses `data/prices.csv` and `data/rf.csv`, which are exports of the
  same data from the notebook, and says so on the page.

## Run it on your own computer

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tested with Python 3.13 and the package versions pinned in `requirements.txt` (pandas 3 needs Python 3.11 or later).
The live app runs on Streamlit Community Cloud with Python 3.14.

## Files

| File | What it is |
| --- | --- |
| `app.py` | The Streamlit page: controls, charts, tables |
| `core.py` | The notebook's calculations |
| `requirements.txt` | Packages and versions |
| `data/` | Backup copy of the notebook's data, used only if the live download fails |

## Use of AI

Built with AI coding assistance (Claude). We checked the output by comparing the app's default
results with the notebook, and by running the notebook's own engine tests inside the app
(see "Method and data" in the app).

For education only. Past performance does not predict future returns. This is not investment advice.
