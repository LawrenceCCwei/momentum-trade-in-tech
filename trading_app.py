import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf


def load_sector_map(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


@st.cache_data(ttl=3600)
def fetch_close_data(symbols: Tuple[str, ...], period: str, interval: str = "1d") -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    data = yf.download(
        tickers=list(symbols),
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close_df = data.xs("Close", axis=1, level=1, drop_level=False)
        close_df.columns = [c[0] for c in close_df.columns]
    else:
        close_df = pd.DataFrame({"SINGLE": data["Close"]})
    return close_df.sort_index().dropna(how="all")


def momentum_series(close_df: pd.DataFrame, lookback: int) -> pd.Series:
    if close_df.empty or len(close_df) <= lookback:
        return pd.Series(dtype=float)
    return close_df.iloc[-1] / close_df.iloc[-lookback - 1] - 1.0


def ma_filter(close_df: pd.DataFrame, ma_window: int) -> pd.Series:
    if close_df.empty:
        return pd.Series(dtype=bool)
    ma = close_df.rolling(ma_window).mean().iloc[-1]
    last_close = close_df.iloc[-1]
    return (last_close >= ma).fillna(False)


def calc_max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def calc_sharpe(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    vol = daily_returns.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float((daily_returns.mean() / vol) * np.sqrt(252))


def calc_sortino(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    downside = daily_returns[daily_returns < 0]
    downside_vol = downside.std(ddof=0) if not downside.empty else 0.0
    if downside_vol == 0 or np.isnan(downside_vol):
        return 0.0
    return float((daily_returns.mean() / downside_vol) * np.sqrt(252))


def run_backtest_no_cost(
    close_df: pd.DataFrame,
    sector_map: Dict[str, List[str]],
    lookback: int,
    ma_window: int,
    rebalance_days: int,
    top_k: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if close_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ret_df = close_df.pct_change()
    sector_returns = pd.DataFrame(index=ret_df.index)
    sector_members: Dict[str, List[str]] = {}
    for sector, symbols in sector_map.items():
        members = [s for s in symbols if s in ret_df.columns]
        if not members:
            continue
        sector_members[sector] = members
        sector_returns[sector] = ret_df[members].mean(axis=1, skipna=True)

    sector_returns = sector_returns.dropna(how="all")
    if sector_returns.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    dates = list(sector_returns.index)
    min_start = max(int(lookback), int(ma_window))
    if len(dates) <= min_start + 1:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    strategy_returns = pd.Series(0.0, index=sector_returns.index)
    rebalance_log: List[dict] = []

    i = min_start
    while i < len(dates) - 1:
        d = dates[i]
        rows = []
        for sector, members in sector_members.items():
            close_now = close_df.loc[d, members].dropna()
            if close_now.empty:
                continue
            prev_idx = i - int(lookback)
            prev_d = dates[prev_idx]
            close_prev = close_df.loc[prev_d, members].dropna()
            common = sorted(set(close_now.index).intersection(set(close_prev.index)))
            if not common:
                continue
            mom_vals = close_now[common] / close_prev[common] - 1.0
            if mom_vals.empty:
                continue

            hist_slice = close_df.loc[:d, common]
            ma_vals = hist_slice.rolling(int(ma_window)).mean().iloc[-1].dropna()
            common2 = sorted(set(common).intersection(set(ma_vals.index)))
            if not common2:
                continue
            above_ratio = float((close_now[common2] >= ma_vals[common2]).mean())
            sector_mom = float(mom_vals[common2].mean())
            score = sector_mom * 0.7 + above_ratio * 0.3
            rows.append(
                {
                    "sector": sector,
                    "score": score,
                    "sector_momentum": sector_mom,
                    "above_ma_ratio": above_ratio,
                }
            )

        score_df = pd.DataFrame(rows).sort_values("score", ascending=False)
        if score_df.empty:
            i += int(rebalance_days)
            continue
        selected = score_df.head(int(top_k))
        selected_sectors = selected["sector"].tolist()
        rebalance_log.append(
            {
                "rebalance_date": pd.to_datetime(d).strftime("%Y-%m-%d"),
                "selected_sectors": ", ".join(selected_sectors),
                "avg_score": float(selected["score"].mean()),
            }
        )

        end_i = min(i + int(rebalance_days), len(dates) - 1)
        for j in range(i + 1, end_i + 1):
            day = dates[j]
            if selected_sectors:
                day_ret = sector_returns.loc[day, selected_sectors].mean(skipna=True)
                strategy_returns.loc[day] = 0.0 if np.isnan(day_ret) else float(day_ret)
            else:
                strategy_returns.loc[day] = 0.0
        i += int(rebalance_days)

    strategy_returns = strategy_returns.loc[strategy_returns.index >= dates[min_start]]
    benchmark_returns = sector_returns.mean(axis=1, skipna=True).reindex(strategy_returns.index).fillna(0.0)
    strategy_equity = (1.0 + strategy_returns.fillna(0.0)).cumprod()
    benchmark_equity = (1.0 + benchmark_returns).cumprod()
    if strategy_equity.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    span_days = max((strategy_equity.index[-1] - strategy_equity.index[0]).days, 1)
    years = span_days / 365.25
    total_return = float(strategy_equity.iloc[-1] - 1.0)
    cagr = float(strategy_equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0

    metrics_df = pd.DataFrame(
        [
            {"metric": "Total Return", "value": total_return},
            {"metric": "CAGR", "value": cagr},
            {"metric": "Sharpe", "value": calc_sharpe(strategy_returns)},
            {"metric": "Sortino", "value": calc_sortino(strategy_returns)},
            {"metric": "Max Drawdown", "value": calc_max_drawdown(strategy_equity)},
            {"metric": "Backtest Start", "value": strategy_equity.index[0].strftime("%Y-%m-%d")},
            {"metric": "Backtest End", "value": strategy_equity.index[-1].strftime("%Y-%m-%d")},
        ]
    )

    curve_df = pd.DataFrame(
        {
            "date": strategy_equity.index,
            "strategy_equity": strategy_equity.values,
            "benchmark_equity": benchmark_equity.values,
            "strategy_return": strategy_returns.values,
            "benchmark_return": benchmark_returns.values,
        }
    )
    curve_df["drawdown"] = curve_df["strategy_equity"] / curve_df["strategy_equity"].cummax() - 1.0

    log_df = pd.DataFrame(rebalance_log)
    if not log_df.empty:
        log_df = log_df.sort_values("rebalance_date", ascending=False)
    return metrics_df, curve_df, log_df


def build_sector_scores(
    sector_map: Dict[str, List[str]],
    symbol_mom: pd.Series,
    symbol_above_ma: pd.Series,
) -> pd.DataFrame:
    rows = []
    for sector, symbols in sector_map.items():
        valid = [s for s in symbols if s in symbol_mom.index]
        if not valid:
            continue
        moms = symbol_mom[valid].dropna()
        if moms.empty:
            continue
        above_ma_ratio = float(symbol_above_ma.reindex(valid).fillna(False).mean())
        score = float(moms.mean() * 0.7 + above_ma_ratio * 0.3)
        rows.append(
            {
                "sector": sector,
                "sector_momentum": float(moms.mean()),
                "above_ma_ratio": above_ma_ratio,
                "score": score,
                "symbols": ", ".join(valid),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)


def build_trade_plan(
    top_sectors: List[str],
    sector_map: Dict[str, List[str]],
    close_df: pd.DataFrame,
    symbol_mom: pd.Series,
    symbol_above_ma: pd.Series,
    capital: float,
    max_positions: int,
) -> pd.DataFrame:
    picks = []
    for sector in top_sectors:
        members = sector_map.get(sector, [])
        ranked = []
        for symbol in members:
            if symbol not in close_df.columns:
                continue
            mom = symbol_mom.get(symbol, np.nan)
            above_ma = bool(symbol_above_ma.get(symbol, False))
            price = close_df[symbol].dropna().iloc[-1] if not close_df[symbol].dropna().empty else np.nan
            ranked.append((symbol, mom, above_ma, price, sector))
        ranked = [r for r in ranked if pd.notna(r[1]) and pd.notna(r[3]) and r[2]]
        ranked.sort(key=lambda x: x[1], reverse=True)
        if ranked:
            picks.append(ranked[0])

    picks = picks[:max_positions]
    if not picks:
        return pd.DataFrame()

    weight = 1.0 / len(picks)
    rows = []
    for symbol, mom, _, price, sector in picks:
        budget = capital * weight
        shares = int(budget // price) if price > 0 else 0
        invested = shares * price
        rows.append(
            {
                "sector": sector,
                "symbol": symbol,
                "momentum": float(mom),
                "last_price": float(price),
                "target_weight": weight,
                "shares": shares,
                "capital_used": invested,
            }
        )
    return pd.DataFrame(rows).sort_values("momentum", ascending=False).reset_index(drop=True)


def main() -> None:
    st.set_page_config(page_title="Tech Momentum Trading", layout="wide")
    st.title("Tech Momentum Trading Planner")
    st.caption("Separate trading app. Market scanner remains in app.py.")

    sector_path = Path("sectors.json")
    if not sector_path.exists():
        st.error("Missing sectors.json")
        return

    sector_map = load_sector_map(sector_path)
    symbols = sorted({s for arr in sector_map.values() for s in arr})

    with st.sidebar:
        st.header("Trading Parameters")
        period = st.selectbox("History period", ["1y", "2y", "3y", "5y"], index=1)
        lookback = st.number_input("Momentum lookback (days)", min_value=20, max_value=252, value=60, step=5)
        ma_window = st.number_input("Trend MA window", min_value=20, max_value=250, value=60, step=5)
        top_sector_n = st.number_input("Top sectors", min_value=1, max_value=min(10, len(sector_map)), value=4, step=1)
        max_positions = st.number_input("Max positions", min_value=1, max_value=20, value=4, step=1)
        capital = st.number_input("Capital (USD)", min_value=1000.0, max_value=10000000.0, value=100000.0, step=1000.0)

    close_df = fetch_close_data(tuple(symbols), period, "1d")
    if close_df.empty:
        st.warning("No price data. Try again later.")
        return

    symbol_mom = momentum_series(close_df, int(lookback))
    symbol_above_ma = ma_filter(close_df, int(ma_window))
    sector_scores = build_sector_scores(sector_map, symbol_mom, symbol_above_ma)

    if sector_scores.empty:
        st.warning("No sector scores available.")
        return

    st.subheader("Sector Ranking")
    st.dataframe(
        sector_scores.style.format(
            {
                "sector_momentum": "{:+.2%}",
                "above_ma_ratio": "{:.0%}",
                "score": "{:+.3f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    top_sectors = sector_scores.head(int(top_sector_n))["sector"].tolist()
    st.caption("Selected sectors: " + ", ".join(top_sectors))

    plot_df = sector_scores.head(int(top_sector_n)).copy()
    plot_df["score_label"] = plot_df["score"].round(3).astype(str)
    fig = px.bar(
        plot_df,
        x="sector",
        y="score",
        color="sector_momentum",
        color_continuous_scale=["#b91c1c", "#f59e0b", "#10b981"],
        title="Top Sector Momentum Score",
    )
    fig.update_layout(xaxis_tickangle=-30, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Backtest (No Trading Cost)")
    bt_col1, bt_col2, bt_col3, bt_col4 = st.columns([1, 1, 1, 1])
    with bt_col1:
        bt_frequency = st.selectbox("Backtest frequency", ["Daily", "Weekly"], index=0)
    with bt_col2:
        bt_rebalance = st.number_input("Rebalance every (bars)", min_value=1, max_value=63, value=5, step=1)
    with bt_col3:
        bt_topk = st.number_input(
            "Backtest Top sectors",
            min_value=1,
            max_value=min(10, len(sector_map)),
            value=min(int(top_sector_n), min(10, len(sector_map))),
            step=1,
        )
    with bt_col4:
        st.caption("Trading cost is fixed at 0 for this version.")

    bt_interval = "1wk" if bt_frequency == "Weekly" else "1d"
    bt_close_df = fetch_close_data(tuple(symbols), period, bt_interval)

    metrics_df, curve_df, rebalance_df = run_backtest_no_cost(
        close_df=bt_close_df,
        sector_map=sector_map,
        lookback=int(lookback),
        ma_window=int(ma_window),
        rebalance_days=int(bt_rebalance),
        top_k=int(bt_topk),
    )

    if metrics_df.empty or curve_df.empty:
        st.warning("Backtest data not sufficient. Increase history period or reduce windows.")
    else:
        metric_map = {r["metric"]: r["value"] for _, r in metrics_df.iterrows()}
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Return", f"{metric_map.get('Total Return', 0.0):.2%}")
        m2.metric("CAGR", f"{metric_map.get('CAGR', 0.0):.2%}")
        m3.metric("Sharpe", f"{metric_map.get('Sharpe', 0.0):.2f}")
        m4.metric("Sortino", f"{metric_map.get('Sortino', 0.0):.2f}")
        m5.metric("Max Drawdown", f"{metric_map.get('Max Drawdown', 0.0):.2%}")

        eq_fig = px.line(
            curve_df,
            x="date",
            y=["strategy_equity", "benchmark_equity"],
            title="Equity Curve: Strategy vs Benchmark",
        )
        eq_fig.update_layout(legend_title_text="", xaxis_title="Date", yaxis_title="Equity (Start=1)")
        st.plotly_chart(eq_fig, use_container_width=True)

        dd_fig = px.area(
            curve_df,
            x="date",
            y="drawdown",
            title="Strategy Drawdown",
        )
        dd_fig.update_layout(xaxis_title="Date", yaxis_title="Drawdown")
        st.plotly_chart(dd_fig, use_container_width=True)

        if not rebalance_df.empty:
            st.caption("Recent rebalances:")
            st.dataframe(rebalance_df.head(30), use_container_width=True, hide_index=True)

    trade_plan = build_trade_plan(
        top_sectors=top_sectors,
        sector_map=sector_map,
        close_df=close_df,
        symbol_mom=symbol_mom,
        symbol_above_ma=symbol_above_ma,
        capital=float(capital),
        max_positions=int(max_positions),
    )

    st.subheader("Suggested Orders (Paper Trading)")
    if trade_plan.empty:
        st.warning("No tradable symbols passed the MA filter.")
    else:
        st.dataframe(
            trade_plan.style.format(
                {
                    "momentum": "{:+.2%}",
                    "last_price": "${:,.2f}",
                    "target_weight": "{:.1%}",
                    "capital_used": "${:,.2f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        total_used = float(trade_plan["capital_used"].sum())
        st.caption(f"Capital used: ${total_used:,.2f} / ${capital:,.2f}")


if __name__ == "__main__":
    main()
