"""Calculations used by the Streamlit app.

Everything here comes from the group's master notebook, so the app and the
notebook give the same numbers. Each block says where it comes from:

  NOTEBOOK  copied word for word from the notebook. Do not edit it here:
            change the notebook, then copy the new version into this file.
  WRAPPED   the notebook's own lines, only put inside a function so the app
            can call them.
  STAND-IN  temporary code until the group's version exists. Replace it with
            the notebook's function as soon as it is written.
  APP ONLY  small helpers the notebook does not need (no finance maths).
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# NOTEBOOK, section 0: constants (decisions fixed in the group plan)
# ---------------------------------------------------------------------------
START = "2004-12-01"     # download start (Dec 2004 = first full month of GLD)
END   = "2026-01-01"     # download end (exclusive -> last month = Dec 2025)

TICKERS = ["SPY", "EFA", "EEM", "IEF", "GLD"]
BASE_WEIGHTS = pd.Series({"SPY": 0.50, "EFA": 0.20, "EEM": 0.10, "IEF": 0.10, "GLD": 0.10})

BT_START = "2006-01-01"  # first backtest month (2005 is only used for the 12-month signal)
BT_END   = "2025-12-01"  # last backtest month
COST_BASE = 0.0010       # 10 bps per 100% turnover (base case)

FRED_URL = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
            "?id=TB3MS&cosd=2004-12-01&coed=2025-12-01")


# ---------------------------------------------------------------------------
# WRAPPED, sections 1.1, 1.2 and 1.4: download prices and T-bill rates
# ---------------------------------------------------------------------------
def download_data():
    """Prices from Yahoo Finance and the monthly T-bill return from FRED.

    Same lines as notebook sections 1.1, 1.2 and the rf formula of 1.4.
    Returns (prices, rf_all), with rf_all on FRED's dates.
    """
    import yfinance as yf

    raw = yf.download(
        TICKERS,
        start=START,
        end=END,
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )

    prices = raw["Close"][TICKERS].copy()          # Close column only (already adjusted), fixed column order
    prices.index = pd.to_datetime(prices.index).tz_localize(None)  # remove time zone
    prices.index.name = "date"

    tb3ms = pd.read_csv(FRED_URL, index_col=0, parse_dates=True)["TB3MS"]
    tb3ms.index.name = "date"
    tb3ms = pd.to_numeric(tb3ms, errors="coerce")   # FRED marks missing values with "."

    rf_all = (1 + tb3ms / 100) ** (1 / 12) - 1
    return prices, rf_all


def load_snapshot(folder):
    """APP ONLY: backup data, used only if the live download fails.

    The files are the notebook's own exports from section 1.6 (prices.csv, rf.csv).
    """
    folder = Path(folder)
    prices = pd.read_csv(folder / "prices.csv", index_col=0, parse_dates=True)[TICKERS]
    prices.index.name = "date"
    rf_all = pd.read_csv(folder / "rf.csv", index_col=0, parse_dates=True)["rf"]
    rf_all.index.name = "date"
    return prices, rf_all


def prepare(prices, rf_all):
    """WRAPPED, sections 1.3 and 1.4: the data checks, returns and rf on the same dates."""
    # Check 1 - shape, Check 2 - dates, Check 3 - missing values
    assert prices.shape == (253, 5), f"Expected (253, 5), got {prices.shape}"
    assert prices.index[0] == pd.Timestamp("2004-12-01")
    assert prices.index[-1] == pd.Timestamp("2025-12-01")
    assert prices.isna().sum().sum() == 0, "Missing values found: check where they are before continuing"

    returns = prices.pct_change().dropna()   # only drops the first row (Dec 2004), which has no return

    rf = rf_all.reindex(returns.index)       # same dates as returns
    rf.name = "rf"

    assert returns.shape == (252, 5), f"Expected (252, 5), got {returns.shape}"
    assert returns.index[0] == pd.Timestamp("2005-01-01")
    assert rf.isna().sum() == 0, "rf has dates with no value after alignment"
    assert (rf.index == returns.index).all()
    return returns, rf


# ---------------------------------------------------------------------------
# NOTEBOOK, section 2: backtest engine
# ---------------------------------------------------------------------------
def backtest(weights, returns, rf, cost=COST_BASE):
    """Backtest a monthly-rebalanced long-only portfolio with a cash sleeve.

    weights : DataFrame (months x ETFs). Row t = weights held during month t
              (already lagged; this function does not shift). 1 - row sum = cash.
    returns : DataFrame of simple monthly ETF returns (must cover weights.index).
    rf      : Series of monthly T-bill returns in decimal (same dates as returns).
    cost    : transaction cost per 100% turnover (0.001 = 10 bps).

    Returns a DataFrame indexed by month with columns:
    gross, net, cash_w, turnover.
    """
    w = weights.astype(float)

    # Input checks
    assert w.notna().all().all(), "weights contain missing values"
    assert set(w.columns) <= set(returns.columns), "weights have tickers not in returns"
    assert w.index.isin(returns.index).all(), "weights have dates not in returns"
    assert (w >= -1e-12).all().all(), "negative weights (no short selling allowed)"
    assert (w.sum(axis=1) <= 1 + 1e-9).all(), "weights sum above 100% (no leverage allowed)"

    r = returns.loc[w.index, w.columns]
    f = rf.loc[w.index]

    cash_w = 1 - w.sum(axis=1)
    gross = (w * r).sum(axis=1) + cash_w * f

    # Weights at the end of each month, after drifting with prices
    drifted = (w * (1 + r)).div(1 + gross, axis=0)

    # Turnover when rebalancing into month t: target weights vs drifted weights of month t-1
    turnover = (w - drifted.shift(1)).abs().sum(axis=1)
    turnover.iloc[0] = 0.0          # initial purchase not counted

    net = gross - cost * turnover

    return pd.DataFrame({"gross": gross, "net": net, "cash_w": cash_w, "turnover": turnover})


def engine_self_test():
    """NOTEBOOK, section 2.1: the hand-calculated examples. Raises if the engine is wrong."""
    _idx = pd.to_datetime(["2000-01-01", "2000-02-01"])
    _w   = pd.DataFrame({"A": [0.5, 0.5], "B": [0.5, 0.5]}, index=_idx)
    _r   = pd.DataFrame({"A": [0.10, 0.00], "B": [0.00, 0.00]}, index=_idx)
    _rf  = pd.Series([0.0, 0.0], index=_idx)

    _bt = backtest(_w, _r, _rf, cost=0.001)

    assert np.isclose(_bt["gross"].iloc[0], 0.05)
    assert np.isclose(_bt["turnover"].iloc[0], 0.0)
    assert np.isclose(_bt["turnover"].iloc[1], 2 * (0.55 / 1.05 - 0.5))
    assert np.isclose(_bt["net"].iloc[1], 0.0 - 0.001 * 2 * (0.55 / 1.05 - 0.5))

    # Cash sleeve: 40% in A, 60% cash earning 1%
    _bt2 = backtest(pd.DataFrame({"A": [0.4]}, index=_idx[:1]), _r[["A"]].iloc[:1], pd.Series([0.01], index=_idx[:1]))
    assert np.isclose(_bt2["gross"].iloc[0], 0.4 * 0.10 + 0.6 * 0.01)
    assert np.isclose(_bt2["cash_w"].iloc[0], 0.6)
    return True


# ---------------------------------------------------------------------------
# WRAPPED, section 3: benchmarks
# ---------------------------------------------------------------------------
def static_weights(bt_index):
    """Static 80/20: the base weights in every month (notebook section 3)."""
    return pd.DataFrame([BASE_WEIGHTS] * len(bt_index), index=bt_index)


# ---------------------------------------------------------------------------
# NOTEBOOK, section 4: Sharpe ratio
# ---------------------------------------------------------------------------
def sharpe(r, rf):
    """Annualized Sharpe ratio from monthly simple returns."""
    ex = r - rf.loc[r.index]
    return ex.mean() / ex.std() * np.sqrt(12)


# ---------------------------------------------------------------------------
# NOTEBOOK, section 5.4: maximum drawdown
# ---------------------------------------------------------------------------
def max_drawdown(r):
    """Largest peak-to-trough fall of the value path built from returns r, starting at 1."""
    wealth = pd.concat([pd.Series([1.0]), (1 + r).cumprod().reset_index(drop=True)])
    return (wealth / wealth.cummax() - 1).min()


# ---------------------------------------------------------------------------
# STAND-IN for person 4: signals and weights
# ---------------------------------------------------------------------------
def make_weights(prices, horizons=(1, 3, 6, 12)):
    """STAND-IN until person 4's make_weights() is in the notebook.

    Trend score per ETF = share of the chosen horizons with a positive trailing
    return (so 0, 25, 50, 75 or 100% with four horizons; 0 or 100% with only 12).
    Weight = base weight x score; whatever is left goes to T-bills (cash).
    The score is measured at the end of month t-1 and held during month t,
    hence .shift(1).

    Returns months Jan 2006 to Dec 2025 in rows, the 5 ETFs in columns.
    """
    horizons = tuple(horizons)
    score = sum((prices.pct_change(h) > 0).astype(float) for h in horizons) / len(horizons)
    weights = (score * BASE_WEIGHTS).shift(1)        # decided with data up to t-1, held in t
    weights = weights.loc[BT_START:BT_END, TICKERS]
    assert weights.notna().all().all(), "not enough history for the longest horizon"
    return weights


# ---------------------------------------------------------------------------
# STAND-IN for person 5: performance measures
# ---------------------------------------------------------------------------
def perf_metrics(r, rf, bench):
    """STAND-IN until person 5's metrics function is in the notebook.

    r, bench : monthly simple returns on the same dates; rf : monthly T-bill return.
    Annualized return = mean x 12; volatility = std x sqrt(12); Sharpe from the
    notebook's sharpe(); alpha and beta from the regression of the excess return
    on the benchmark's excess return (alpha x 12, beta not annualized);
    information ratio = annualized mean / annualized std of r - bench;
    max drawdown from the notebook's max_drawdown().
    """
    rf = rf.loc[r.index]
    bench = bench.loc[r.index]
    ex, bex = r - rf, bench - rf
    beta, alpha_m = np.polyfit(bex, ex, 1)          # slope, intercept
    active = r - bench
    te = active.std()
    ir = (12 * active.mean()) / (np.sqrt(12) * te) if te > 1e-12 else np.nan
    return pd.Series({
        "Annualized return": 12 * r.mean(),
        "Volatility": np.sqrt(12) * r.std(),
        "Sharpe ratio": sharpe(r, rf),
        "Alpha (annualized)": 12 * alpha_m,
        "Beta": beta,
        "Information ratio": ir,
        "Max drawdown": max_drawdown(r),
    })


# ---------------------------------------------------------------------------
# APP ONLY: chart helpers (no finance maths beyond compounding)
# ---------------------------------------------------------------------------
def growth_of_one(r):
    """Value of 1 invested, set to 1 in the month before r starts (as in the notebook's figure)."""
    value = (1 + r).cumprod()
    start = r.index[0] - pd.DateOffset(months=1)
    return pd.concat([pd.Series([1.0], index=[start]), value])


def drawdown_path(r):
    """Fall from the running peak each month, starting from a value of 1."""
    value = growth_of_one(r)
    return value / value.cummax() - 1
