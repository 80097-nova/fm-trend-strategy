"""Streamlit app for the 2270 Financial Modeling group assignment.

All calculations come from core.py, which mirrors the group's master notebook.
This file only reads the user's choices, calls those functions and draws the results.
Run locally with:  streamlit run app.py
"""

from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import core

# ---------------------------------------------------------------------------
# Settings to fill in before submission
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/OWNER/REPO"          # replace with the GitHub repository link
GROUP_NAME = "Moodle group name"                    # replace with the group's Moodle name
STAND_INS = True    # set to False once persons 4 and 5's functions replace the stand-ins in core.py

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="Trend-filtered 80/20 portfolio", layout="wide")


# ---------------------------------------------------------------------------
# Data: live download, cached for a day, with the notebook's export as backup
# ---------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner="Downloading prices from Yahoo Finance and T-bill rates from FRED...")
def get_data():
    try:
        prices, rf_all = core.download_data()
        returns, rf = core.prepare(prices, rf_all)
        source = "live"
    except Exception as err:  # Yahoo or FRED unavailable, or the download failed a data check
        prices, rf_all = core.load_snapshot(DATA_DIR)
        returns, rf = core.prepare(prices, rf_all)
        source = f"backup ({type(err).__name__})"
    return prices, returns, rf, source


@st.cache_data
def engine_ok():
    return core.engine_self_test()


prices, returns, rf, source = get_data()
bt_index = returns.loc[core.BT_START:core.BT_END].index
assert len(bt_index) == 240


# ---------------------------------------------------------------------------
# Colours: one per portfolio, the same in every chart
# ---------------------------------------------------------------------------
try:
    dark = st.context.theme.type == "dark"
except Exception:
    dark = False
SERIES_COLORS = (
    {"Trend strategy": "#338ced", "Static 80/20": "#e76624", "SPY": "#00a888"} if dark
    else {"Trend strategy": "#257de0", "Static 80/20": "#f47230", "SPY": "#00b894"}
)


# ---------------------------------------------------------------------------
# Sidebar: the assumptions a user can change
# ---------------------------------------------------------------------------
LAYERS = {
    "Trend score, several horizons (the strategy)": "multi",
    "12-month filter only": "filter12",
    "Off: static 80/20": "static",
}

st.sidebar.header("Assumptions")
layer_label = st.sidebar.radio("Trend filter", list(LAYERS), index=0)
layer = LAYERS[layer_label]

horizons = st.sidebar.multiselect(
    "Horizons in the trend score (months)",
    options=[1, 3, 6, 12],
    default=[1, 3, 6, 12],
    disabled=layer != "multi",
    help="Each ETF keeps the share of its base weight equal to the share of these horizons "
         "with a positive past return. The rest goes to T-bills.",
)
if layer == "multi" and not horizons:
    st.sidebar.warning("Pick at least one horizon. Using all four.")
    horizons = [1, 3, 6, 12]

cost_bps = st.sidebar.slider(
    "Transaction cost (bps per 100% turnover)", min_value=0, max_value=100,
    value=int(round(core.COST_BASE * 10_000)), step=5,
    help="Charged on every trade of the strategy and of the static 80/20. SPY is buy-and-hold and pays none.",
)

starts = list(bt_index[: len(bt_index) - 35])     # keep at least 3 years in the window
start = st.sidebar.select_slider(
    "Start of the evaluation window", options=starts, value=starts[0],
    format_func=lambda d: d.strftime("%b %Y"),
    help="The backtest always runs on the full sample; this only changes the window shown and measured. "
         "Growth is set to 1 in the month before the start.",
)

bench_label = st.sidebar.radio(
    "Benchmark for alpha, beta and information ratio", ["SPY", "Static 80/20"], index=0,
)


# ---------------------------------------------------------------------------
# Calculations: all through core.py, on the full 240 months, then cut to the window
# ---------------------------------------------------------------------------
cost = cost_bps / 10_000
w_static = core.static_weights(bt_index)
bt_static = core.backtest(w_static, returns, rf, cost=cost)

if layer == "static":
    w_strat = w_static
elif layer == "filter12":
    w_strat = core.make_weights(prices, (12,))
else:
    w_strat = core.make_weights(prices, tuple(sorted(horizons)))
bt_strat = core.backtest(w_strat, returns, rf, cost=cost)

window = slice(start, bt_index[-1])
series = {}
if layer != "static":
    series["Trend strategy"] = bt_strat["net"].loc[window]
series["Static 80/20"] = bt_static["net"].loc[window]
series["SPY"] = returns["SPY"].loc[window]
rf_w = rf.loc[window]
bench = series[bench_label]

metrics = pd.DataFrame({name: core.perf_metrics(r, rf_w, bench) for name, r in series.items()})
turnover = {"Trend strategy": bt_strat["turnover"].loc[window].mean() * 12,
            "Static 80/20": bt_static["turnover"].loc[window].mean() * 12,
            "SPY": 0.0}
metrics.loc["Annual turnover"] = [turnover[c] for c in metrics.columns]
metrics.loc["Average share in T-bills"] = [
    bt_strat["cash_w"].loc[window].mean() if c == "Trend strategy" else 0.0 for c in metrics.columns
]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
period = f"{start.strftime('%b %Y')} to {bt_index[-1].strftime('%b %Y')}"
st.title("Trend-filtered 80/20 portfolio")
st.markdown(
    "A global portfolio of five ETFs (SPY 50%, EFA 20%, EEM 10%, IEF 10%, GLD 10%) that moves part of an "
    "ETF's weight into US T-bills when its price trend turns negative. Change the assumptions on the left "
    f"to see how the result and the comparison with the benchmarks change. Monthly data, {period}."
)

if STAND_INS:
    st.info(
        "Preview version: the trend signal and the performance measures use stand-in code until the "
        "group's final functions are added, so the strategy's numbers can still change. "
        "The data, the backtest engine and the benchmarks are final."
    )
if source != "live":
    st.warning(
        "The live download from Yahoo Finance or FRED failed, so the app is using the notebook's saved "
        f"data (same dates, Dec 2004 to Dec 2025). Reason: {source}."
    )


def pct(x, d=1):
    if pd.isna(x):
        return "–"
    if abs(x) < 0.5 * 10 ** -(d + 2):
        x = 0.0                      # avoid showing -0.0%
    return f"{x:.{d}%}"


def pp(x):
    return f"{x * 100:+.1f} pp"


# Headline numbers: the strategy (or the 80/20 when the filter is off) against the 80/20
lead = "Trend strategy" if layer != "static" else "Static 80/20"
m, ref = metrics[lead], metrics["Static 80/20"]
c1, c2, c3, c4 = st.columns(4)
show_delta = layer != "static"
c1.metric("Annualized return", pct(m["Annualized return"]),
          pp(m["Annualized return"] - ref["Annualized return"]) + " vs 80/20" if show_delta else None)
c2.metric("Volatility", pct(m["Volatility"]),
          pp(m["Volatility"] - ref["Volatility"]) + " vs 80/20" if show_delta else None, delta_color="inverse")
c3.metric("Sharpe ratio", f"{m['Sharpe ratio']:.2f}",
          f"{m['Sharpe ratio'] - ref['Sharpe ratio']:+.2f} vs 80/20" if show_delta else None)
c4.metric("Max drawdown", pct(m["Max drawdown"]),
          pp(m["Max drawdown"] - ref["Max drawdown"]) + " vs 80/20" if show_delta else None)


def line_chart(df, y_title, y_format, log=False, height=380):
    """Lines with a hover crosshair that lists every portfolio for the month under the pointer."""
    order = list(df.columns)
    long = df.rename_axis("Date").reset_index().melt("Date", var_name="Portfolio", value_name="Value")
    color = alt.Color(
        "Portfolio:N",
        scale=alt.Scale(domain=order, range=[SERIES_COLORS[c] for c in order]),
        legend=alt.Legend(orient="top", title=None),
    )
    y_scale = alt.Scale(type="log") if log else alt.Scale(zero=False)
    base = alt.Chart(long).encode(
        x=alt.X("Date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year", grid=False)),
        y=alt.Y("Value:Q", title=y_title, scale=y_scale, axis=alt.Axis(format=y_format)),
        color=color,
    )
    lines = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round")
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["Date"], empty=False, clear="pointerout")
    tooltip = [alt.Tooltip("Date:T", title="Month", format="%b %Y")] + [
        alt.Tooltip(field=c, type="quantitative", title=c, format=y_format) for c in order
    ]
    rule = (
        alt.Chart(long)
        .transform_pivot("Portfolio", value="Value", groupby=["Date"])
        .mark_rule(color="#888888", strokeWidth=1)
        .encode(x="Date:T", opacity=alt.condition(hover, alt.value(0.6), alt.value(0)), tooltip=tooltip)
        .add_params(hover)
    )
    points = base.mark_point(filled=True, size=70).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0))
    ).transform_filter(hover)
    return alt.layer(lines, rule, points).properties(height=height)


tab_perf, tab_dd, tab_hold, tab_about = st.tabs(
    ["Performance", "Drawdowns", "What the strategy holds", "Method and data"]
)

with tab_perf:
    growth = pd.DataFrame({name: core.growth_of_one(r) for name, r in series.items()})
    log = st.toggle("Log scale", value=False)
    st.subheader(f"Growth of 1 invested, {period}")
    st.altair_chart(line_chart(growth, "Value of 1 invested", ".1f", log=log), width="stretch")

    st.subheader("Performance measures")
    rows = {
        "Annualized return": lambda x: pct(x),
        "Volatility": lambda x: pct(x),
        "Sharpe ratio": lambda x: "–" if pd.isna(x) else f"{x:.2f}",
        "Alpha (annualized)": lambda x: pct(x, 2),
        "Beta": lambda x: "–" if pd.isna(x) else f"{x:.2f}",
        "Information ratio": lambda x: "–" if pd.isna(x) else f"{x:.2f}",
        "Max drawdown": lambda x: pct(x),
        "Annual turnover": lambda x: pct(x, 0),
        "Average share in T-bills": lambda x: pct(x, 0),
    }
    table = pd.DataFrame({c: [fmt(metrics.loc[r, c]) for r, fmt in rows.items()] for c in metrics.columns},
                         index=list(rows))
    st.dataframe(table, width="stretch")
    st.caption(
        f"Strategy and static 80/20 are net of {cost_bps} bps per 100% turnover; SPY is buy-and-hold with no costs. "
        f"Alpha, beta and information ratio are measured against {bench_label}. Annualized return = monthly mean × 12; "
        "volatility = monthly standard deviation × √12; Sharpe ratio uses the 3-month T-bill rate; beta is not annualized."
    )
    with st.expander("Show the monthly values behind the chart"):
        st.dataframe(growth.style.format("{:.3f}"), width="stretch")

with tab_dd:
    dd = pd.DataFrame({name: core.drawdown_path(r) for name, r in series.items()})
    st.subheader("Fall from the previous peak")
    st.altair_chart(line_chart(dd, "Drawdown", ".0%"), width="stretch")
    st.caption("How far each portfolio was below its highest earlier value, month by month.")

with tab_hold:
    if layer == "static":
        st.write("With the trend filter off, the portfolio holds the base weights every month and never holds T-bills.")
    else:
        cash = bt_strat["cash_w"].loc[window].rename("Share in T-bills").rename_axis("Date").reset_index()
        st.subheader("Share of the portfolio moved into T-bills")
        base = alt.Chart(cash).encode(
            x=alt.X("Date:T", title=None, axis=alt.Axis(format="%Y", tickCount="year", grid=False)),
            y=alt.Y("Share in T-bills:Q", title="Share in T-bills", axis=alt.Axis(format=".0%"),
                    scale=alt.Scale(domain=[0, 1])),
            tooltip=[alt.Tooltip("Date:T", title="Month", format="%b %Y"),
                     alt.Tooltip("Share in T-bills:Q", format=".0%")],
        )
        area = base.mark_area(color=SERIES_COLORS["Trend strategy"], opacity=0.12, interpolate="step-after")
        edge = base.mark_line(color=SERIES_COLORS["Trend strategy"], strokeWidth=2, interpolate="step-after")
        st.altair_chart((area + edge).properties(height=320), width="stretch")
        st.caption("High values mean most ETFs were in a downtrend and the strategy was mostly in T-bills.")

        st.subheader("Weights in the last month of the window")
        last = w_strat.loc[window].iloc[-1]
        held = pd.concat([last, pd.Series({"T-bills": 1 - last.sum()})])
        base_w = pd.concat([core.BASE_WEIGHTS, pd.Series({"T-bills": 0.0})])
        st.dataframe(
            pd.DataFrame({"Base weight": base_w, "Held": held}).style.format("{:.0%}"),
            width="content",
        )

with tab_about:
    st.markdown(
        f"""
**Data.** Monthly prices of SPY, EFA, EEM, IEF and GLD from Yahoo Finance (via `yfinance`), adjusted for
dividends and splits, and the 3-month US T-bill rate (TB3MS) from FRED. Sample Dec 2004 to Dec 2025;
2005 is used only to start the 12-month signal, so results run from Jan 2006. Data source for this session: **{source}**.

**Rule.** Each month every ETF gets a trend score: the share of the chosen look-back horizons over which its
return was positive. It keeps that share of its base weight; the rest is held in T-bills. The score uses
prices up to the end of the previous month only (no look-ahead).

**Costs.** Trading costs are charged on turnover (the sum of absolute weight changes after prices drift), at
the rate set on the left, for both the strategy and the static 80/20. Moving into or out of T-bills is free.

**Same numbers as the notebook.** Every calculation comes from `core.py`, which copies the functions of the
group's notebook. With the default settings the app reproduces the notebook's main table.

**Source code:** [{REPO_URL}]({REPO_URL}) · Group: {GROUP_NAME}

For education only; past performance does not predict future returns. This is not investment advice.
"""
    )
    checks = {
        "Backtest engine passes the notebook's hand-calculated examples": engine_ok(),
        "Weights plus T-bills add up to 100% every month": bool(np.allclose(w_strat.sum(axis=1) + bt_strat["cash_w"], 1)),
        "No ETF ever holds more than its base weight": bool((w_strat <= core.BASE_WEIGHTS + 1e-12).all().all()),
    }
    st.markdown("**Checks run on every change**")
    for name, ok in checks.items():
        st.markdown(f"{'✅' if ok else '❌'} {name}")
