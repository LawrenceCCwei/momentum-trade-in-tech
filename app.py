import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

DEFAULT_REFRESH_MINUTES = 60
MAX_RETRIES = 3
RETRY_BASE_SECONDS = 1.0
KLINE_TIMEFRAMES = {
    "3M (1D)": ("3mo", "1d"),
    "6M (1D)": ("6mo", "1d"),
    "1Y (1D)": ("1y", "1d"),
    "2Y (1W)": ("2y", "1wk"),
    "5Y (1W)": ("5y", "1wk"),
}


def calc_max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def calc_sharpe(daily_returns: pd.Series) -> float:
    if daily_returns.empty:
        return 0.0
    vol = daily_returns.std(ddof=0)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float((daily_returns.mean() / vol) * np.sqrt(252))


def pick_float(payload: dict, keys: List[str]) -> Optional[float]:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def pick_str(payload: dict, keys: List[str]) -> Optional[str]:
    for key in keys:
        value = payload.get(key)
        if value is None:
            continue
        return str(value)
    return None


def fetch_symbol_quote(symbol: str) -> Optional[dict]:
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            hist = yf.Ticker(symbol).history(period="5d", interval="1d")
            if hist.empty:
                return None
            closes = hist["Close"].dropna()
            volumes = hist["Volume"].dropna()
            if closes.empty:
                return None
            last_close = closes.iloc[-1]
            prev_close = closes.iloc[-2] if len(closes) > 1 else None
            last_volume = volumes.iloc[-1] if not volumes.empty else None
            change_pct = None
            try:
                if prev_close not in (None, 0):
                    change_pct = (float(last_close) - float(prev_close)) / float(prev_close) * 100
            except (TypeError, ValueError):
                change_pct = None
            return {
                "symbol": symbol,
                "volume": last_volume,
                "change_percent": change_pct,
            }
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_SECONDS * (2**attempt))
    if last_error:
        return None
    return None


def fetch_quotes(symbols: List[str]) -> Tuple[List[dict], List[str]]:
    results: List[dict] = []
    failed_symbols: List[str] = []
    for symbol in symbols:
        quote = fetch_symbol_quote(symbol)
        if quote is None:
            failed_symbols.append(symbol)
            continue
        results.append(quote)
    return results, failed_symbols


def fetch_candles(symbol: str, period: str, interval: str) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            hist = yf.Ticker(symbol).history(period=period, interval=interval)
            if hist is None or hist.empty:
                return pd.DataFrame()
            cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in hist.columns]
            data = hist[cols].copy().dropna(subset=["Open", "High", "Low", "Close"], how="any")
            if data.empty:
                return pd.DataFrame()
            data.index = pd.to_datetime(data.index)
            return data
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_SECONDS * (2**attempt))
    if last_error:
        return pd.DataFrame()
    return pd.DataFrame()


def run_momentum_backtest(
    sector_map: Dict[str, List[str]],
    lookback_days: int,
    hold_days: int,
    top_k: int,
    years: int,
    fee_bps: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    unique_symbols = sorted({s for symbols in sector_map.values() for s in symbols})
    if not unique_symbols:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    close_df = pd.DataFrame()
    for symbol in unique_symbols:
        hist = fetch_candles(symbol, period=f"{years}y", interval="1d")
        if hist.empty or "Close" not in hist.columns:
            continue
        close_df[symbol] = hist["Close"]

    close_df = close_df.sort_index().dropna(how="all")
    if close_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    stock_rets = close_df.pct_change()
    sector_rets = pd.DataFrame(index=stock_rets.index)
    for sector, symbols in sector_map.items():
        members = [s for s in symbols if s in stock_rets.columns]
        if not members:
            continue
        sector_rets[sector] = stock_rets[members].mean(axis=1, skipna=True)

    sector_rets = sector_rets.dropna(how="all")
    if sector_rets.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    momentum_scores = (1 + sector_rets).rolling(lookback_days).apply(np.prod, raw=True) - 1
    strategy_returns = pd.Series(0.0, index=sector_rets.index)
    picks_log: List[dict] = []
    prev_selection: List[str] = []

    i = max(lookback_days, 2)
    while i < len(sector_rets.index) - 1:
        rebalance_day = sector_rets.index[i]
        scores = momentum_scores.loc[rebalance_day].dropna()
        if scores.empty:
            i += 1
            continue
        picks = list(scores.sort_values(ascending=False).head(top_k).index)
        if not picks:
            i += 1
            continue

        turnover = 0.0
        if prev_selection:
            changed = len(set(picks).symmetric_difference(set(prev_selection)))
            turnover = changed / max(len(picks), 1)
        prev_selection = picks

        picks_log.append(
            {
                "rebalance_date": rebalance_day.strftime("%Y-%m-%d"),
                "selected_sectors": ", ".join(picks),
                "avg_momentum": float(scores.loc[picks].mean()),
            }
        )

        end_i = min(i + hold_days, len(sector_rets.index) - 1)
        for j in range(i + 1, end_i + 1):
            day = sector_rets.index[j]
            day_ret = sector_rets.loc[day, picks].mean(skipna=True)
            if np.isnan(day_ret):
                day_ret = 0.0
            trade_cost = (fee_bps / 10000.0) * turnover if j == i + 1 else 0.0
            strategy_returns.loc[day] = float(day_ret) - trade_cost
        i += hold_days

    strategy_returns = strategy_returns.iloc[max(lookback_days, 2) :]
    benchmark_returns = sector_rets.mean(axis=1, skipna=True).reindex(strategy_returns.index).fillna(0.0)
    strategy_equity = (1 + strategy_returns.fillna(0.0)).cumprod()
    benchmark_equity = (1 + benchmark_returns).cumprod()
    if strategy_equity.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    span_days = max((strategy_equity.index[-1] - strategy_equity.index[0]).days, 1)
    years_span = span_days / 365.25
    total_return = float(strategy_equity.iloc[-1] - 1.0)
    cagr = float(strategy_equity.iloc[-1] ** (1 / years_span) - 1.0) if years_span > 0 else 0.0

    metrics_df = pd.DataFrame(
        [
            {"metric": "Total Return", "value": total_return},
            {"metric": "CAGR", "value": cagr},
            {"metric": "Sharpe", "value": calc_sharpe(strategy_returns)},
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
            "strategy_returns": strategy_returns.values,
        }
    )

    picks_df = pd.DataFrame(picks_log)
    if not picks_df.empty:
        picks_df = picks_df.sort_values("rebalance_date", ascending=False)
    return metrics_df, curve_df, picks_df


def merge_with_last_success(
    symbols: List[str],
    fresh_quotes: List[dict],
    last_success_by_symbol: Dict[str, dict],
) -> Tuple[List[dict], List[str], List[str]]:
    fresh_by_symbol = {pick_str(q, ["symbol"]): q for q in fresh_quotes}
    merged: List[dict] = []
    fallback_symbols: List[str] = []
    missing_symbols: List[str] = []
    for symbol in symbols:
        fresh = fresh_by_symbol.get(symbol)
        if fresh:
            merged.append(fresh)
            continue
        fallback = last_success_by_symbol.get(symbol)
        if fallback:
            merged.append(fallback)
            fallback_symbols.append(symbol)
            continue
        missing_symbols.append(symbol)
    return merged, fallback_symbols, missing_symbols


def aggregate_by_sector(
    quotes: List[dict],
    sector_map: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, List[str]]:
    by_symbol = {pick_str(q, ["symbol", "ticker", "code"]): q for q in quotes}
    rows = []
    missing = []
    for sector, symbols in sector_map.items():
        volumes = []
        changes = []
        for symbol in symbols:
            quote = by_symbol.get(symbol)
            if not quote:
                missing.append(symbol)
                continue
            volume = pick_float(quote, ["volume", "day_volume", "dayVolume"])
            change_pct = pick_float(
                quote,
                [
                    "change_percent",
                    "changePercent",
                    "percent_change",
                    "change_pct",
                    "day_change_percent",
                ],
            )
            if volume is not None:
                volumes.append(volume)
            if change_pct is not None:
                changes.append((change_pct, volume))
        total_volume = sum(volumes) if volumes else None
        if changes:
            weighted = [c * v for c, v in changes if v is not None]
            if weighted and any(v is not None for _, v in changes):
                weight_sum = sum(v for _, v in changes if v is not None)
                change = sum(weighted) / weight_sum if weight_sum else sum(c for c, _ in changes) / len(changes)
            else:
                change = sum(c for c, _ in changes) / len(changes)
        else:
            change = None
        rows.append(
            {
                "sector": sector,
                "total_volume": total_volume,
                "change_pct": change,
                "symbols": ", ".join(symbols),
            }
        )
    return pd.DataFrame(rows), missing


def build_stock_level_df(quotes: List[dict], sector_map: Dict[str, List[str]]) -> pd.DataFrame:
    by_symbol = {pick_str(q, ["symbol", "ticker", "code"]): q for q in quotes}
    rows: List[dict] = []
    for sector, symbols in sector_map.items():
        for symbol in symbols:
            quote = by_symbol.get(symbol)
            if not quote:
                continue
            volume = pick_float(quote, ["volume", "day_volume", "dayVolume"])
            change_pct = pick_float(
                quote,
                [
                    "change_percent",
                    "changePercent",
                    "percent_change",
                    "change_pct",
                    "day_change_percent",
                ],
            )
            rows.append(
                {
                    "sector": sector,
                    "symbol": symbol,
                    "volume": volume if volume is not None and volume > 0 else 1.0,
                    "change_pct": change_pct,
                }
            )
    return pd.DataFrame(rows)


def load_sector_map(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def main() -> None:
    st.set_page_config(page_title="US Tech Sector Heatmap", layout="wide")

    st.title("US Tech Sector Flow Heatmap")
    st.caption("Blocks sized by total volume, colored by daily % change.")

    sector_path = Path("sectors.json")
    if not sector_path.exists():
        st.error("Missing sectors.json")
        return

    with st.sidebar:
        st.header("Settings")
        refresh_minutes = st.number_input(
            "Refresh interval (minutes)",
            min_value=1,
            max_value=60,
            value=DEFAULT_REFRESH_MINUTES,
        )
        st.write("Daily candles from Yahoo Finance via yfinance.")

    if st_autorefresh:
        st_autorefresh(interval=int(refresh_minutes * 60 * 1000), key="auto")

    sector_map = load_sector_map(sector_path)
    symbols = sorted({s for symbols in sector_map.values() for s in symbols})

    @st.cache_data(ttl=int(refresh_minutes * 60))
    def get_quotes(symbols_cache_key: Tuple[str, ...]) -> Tuple[List[dict], List[str]]:
        return fetch_quotes(list(symbols_cache_key))

    try:
        fresh_quotes, failed_fetch_symbols = get_quotes(tuple(symbols))
    except Exception as exc:
        st.error(f"Data fetch error: {exc}")
        st.stop()

    last_success_by_symbol = st.session_state.get("last_success_quotes", {})
    quotes, fallback_symbols, missing_after_fallback = merge_with_last_success(
        symbols,
        fresh_quotes,
        last_success_by_symbol,
    )

    for quote in fresh_quotes:
        symbol = pick_str(quote, ["symbol"])
        if symbol:
            last_success_by_symbol[symbol] = quote
    st.session_state["last_success_quotes"] = last_success_by_symbol

    df, missing = aggregate_by_sector(quotes, sector_map)
    if df.empty:
        st.warning("No data returned.")
        st.stop()

    size_series = df["total_volume"].fillna(0)
    if size_series.sum() == 0:
        df["size"] = 1
    else:
        df["size"] = size_series

    valid_changes = df["change_pct"].dropna()
    if valid_changes.empty:
        range_color = (-1, 1)
    else:
        max_abs = max(abs(valid_changes.min()), abs(valid_changes.max()))
        range_color = (-max_abs, max_abs)

    stock_df = build_stock_level_df(quotes, sector_map)
    min_c, max_c = range_color
    if stock_df.empty:
        fig = go.Figure(
            go.Treemap(
                labels=df["sector"],
                parents=[""] * len(df),
                values=df["size"],
                customdata=df[["symbols", "total_volume", "change_pct"]].values,
                marker=dict(
                    colors=df["change_pct"],
                    colorscale=[[0.0, "#b91c1c"], [0.5, "#f59e0b"], [1.0, "#10b981"]],
                    cmin=min_c,
                    cmax=max_c,
                    colorbar=dict(title="% Change"),
                ),
                textinfo="label+value",
                hovertemplate=(
                    "Sector: %{label}<br>"
                    "Symbols: %{customdata[0]}<br>"
                    "Total Volume: %{customdata[1]:,.0f}<br>"
                    "% Change: %{customdata[2]:.2f}<extra></extra>"
                ),
            )
        )
    else:
        sector_nodes = []
        symbol_nodes = []
        for _, row in df.iterrows():
            sector_nodes.append(
                {
                    "id": f"sector::{row['sector']}",
                    "label": row["sector"],
                    "parent": "root",
                    "value": float(row["size"]) if pd.notna(row["size"]) else 1.0,
                    "change_pct": float(row["change_pct"]) if pd.notna(row["change_pct"]) else 0.0,
                    "is_sector": True,
                }
            )
        for _, row in stock_df.iterrows():
            symbol_nodes.append(
                {
                    "id": f"symbol::{row['sector']}::{row['symbol']}",
                    "label": row["symbol"],
                    "parent": f"sector::{row['sector']}",
                    "value": float(row["volume"]) if pd.notna(row["volume"]) else 1.0,
                    "change_pct": float(row["change_pct"]) if pd.notna(row["change_pct"]) else 0.0,
                    "is_sector": False,
                    "sector": row["sector"],
                }
            )

        tree_nodes = (
            [
                {
                    "id": "root",
                    "label": "All Sectors",
                    "parent": "",
                    "value": float(df["size"].sum()) if not df["size"].empty else 1.0,
                    "change_pct": 0.0,
                    "is_sector": True,
                }
            ]
            + sector_nodes
            + symbol_nodes
        )
        tree_df = pd.DataFrame(tree_nodes)
        customdata = np.column_stack(
            [
                tree_df["label"],
                tree_df["change_pct"],
                tree_df["value"],
                tree_df.get("sector", tree_df["label"]),
                tree_df["is_sector"],
            ]
        )

        fig = go.Figure(
            go.Treemap(
                ids=tree_df["id"],
                labels=tree_df["label"],
                parents=tree_df["parent"],
                values=tree_df["value"],
                customdata=customdata,
                branchvalues="total",
                marker=dict(
                    colors=tree_df["change_pct"],
                    colorscale=[[0.0, "#b91c1c"], [0.5, "#f59e0b"], [1.0, "#10b981"]],
                    cmin=min_c,
                    cmax=max_c,
                    colorbar=dict(title="% Change"),
                ),
                textinfo="label+value",
                hovertemplate=(
                    "Node: %{customdata[0]}<br>"
                    "Sector: %{customdata[3]}<br>"
                    "Volume: %{customdata[2]:,.0f}<br>"
                    "% Change: %{customdata[1]:.2f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(title="Click a sector block to drill down into symbols")

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sector Summary")
    st.dataframe(
        df[["sector", "symbols", "total_volume", "change_pct"]]
        .rename(
            columns={
                "sector": "Sector",
                "symbols": "Symbols",
                "total_volume": "Total Volume",
                "change_pct": "% Change",
            }
        ),
        use_container_width=True,
    )

    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if fallback_symbols:
        st.info("Using last successful data for: " + ", ".join(sorted(set(fallback_symbols))))

    unresolved = sorted(set(missing + missing_after_fallback))
    if unresolved:
        st.warning("Missing symbols: " + ", ".join(unresolved))

    hard_failures = sorted(set(failed_fetch_symbols) - set(fallback_symbols))
    if hard_failures:
        st.warning("Fetch retries failed for: " + ", ".join(hard_failures))

    # Volume trend (bar chart)
    timestamp = datetime.now().strftime("%H:%M:%S")
    history = st.session_state.get("volume_history", [])
    snapshot = [
        {"time": timestamp, "sector": row["sector"], "total_volume": row["total_volume"] or 0}
        for _, row in df.iterrows()
    ]
    history.extend(snapshot)
    # Keep last 30 time points per sector (approx)
    max_points_per_sector = 30
    trimmed = []
    counts = {}
    for item in reversed(history):
        sector = item["sector"]
        counts[sector] = counts.get(sector, 0) + 1
        if counts[sector] <= max_points_per_sector:
            trimmed.append(item)
    history = list(reversed(trimmed))
    st.session_state["volume_history"] = history

    st.subheader("Volume Trend (Bar)")
    hist_df = pd.DataFrame(history)
    if not hist_df.empty:
        fig_bar = px.bar(
            hist_df,
            x="time",
            y="total_volume",
            color="sector",
            barmode="group",
            labels={"time": "Time", "total_volume": "Total Volume", "sector": "Sector"},
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Momentum Backtest")
    bt_col1, bt_col2, bt_col3, bt_col4, bt_col5 = st.columns([1, 1, 1, 1, 1])
    with bt_col1:
        bt_years = st.selectbox("History", [1, 2, 3, 5], index=2, key="bt_years")
    with bt_col2:
        bt_lookback = st.number_input("Lookback (days)", min_value=5, max_value=252, value=20, step=1)
    with bt_col3:
        bt_hold = st.number_input("Rebalance every (days)", min_value=1, max_value=63, value=5, step=1)
    with bt_col4:
        bt_topk = st.number_input(
            "Top sectors",
            min_value=1,
            max_value=max(len(sector_map), 1),
            value=min(3, max(len(sector_map), 1)),
            step=1,
        )
    with bt_col5:
        bt_fee = st.number_input("Fee (bps)", min_value=0.0, max_value=50.0, value=2.0, step=0.5)

    @st.cache_data(ttl=int(refresh_minutes * 60))
    def get_backtest(
        sector_map_json: str,
        years: int,
        lookback: int,
        hold: int,
        topk: int,
        fee: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        parsed = json.loads(sector_map_json)
        return run_momentum_backtest(parsed, lookback, hold, topk, years, fee)

    metrics_df, curve_df, picks_df = get_backtest(
        json.dumps(sector_map, sort_keys=True),
        int(bt_years),
        int(bt_lookback),
        int(bt_hold),
        int(bt_topk),
        float(bt_fee),
    )

    if metrics_df.empty or curve_df.empty:
        st.warning("Backtest could not run with current data coverage.")
    else:
        metric_map = {row["metric"]: row["value"] for _, row in metrics_df.iterrows()}
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Return", f"{metric_map.get('Total Return', 0.0):.2%}")
        m2.metric("CAGR", f"{metric_map.get('CAGR', 0.0):.2%}")
        m3.metric("Sharpe", f"{metric_map.get('Sharpe', 0.0):.2f}")
        m4.metric("Max Drawdown", f"{metric_map.get('Max Drawdown', 0.0):.2%}")

        fig_bt = go.Figure()
        fig_bt.add_trace(
            go.Scatter(
                x=curve_df["date"],
                y=curve_df["strategy_equity"],
                mode="lines",
                name="Momentum Strategy",
                line=dict(color="#2563eb", width=2),
            )
        )
        fig_bt.add_trace(
            go.Scatter(
                x=curve_df["date"],
                y=curve_df["benchmark_equity"],
                mode="lines",
                name="Equal-Weight Sector Benchmark",
                line=dict(color="#64748b", width=2, dash="dot"),
            )
        )
        fig_bt.update_layout(
            title="Strategy vs Benchmark Equity Curve",
            xaxis_title="Date",
            yaxis_title="Equity (Start = 1.0)",
            margin=dict(l=10, r=10, t=35, b=10),
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        if not picks_df.empty:
            st.caption("Recent rebalances (latest 20):")
            st.dataframe(picks_df, use_container_width=True)

    st.subheader("Candlestick")
    kline_col1, kline_col2, kline_col3 = st.columns([1.4, 1.4, 1.2])
    with kline_col1:
        selected_sector = st.selectbox("Sector", list(sector_map.keys()), key="kline_sector")
    with kline_col2:
        sector_symbols = sector_map.get(selected_sector, [])
        selected_symbol = st.selectbox("Symbol", sector_symbols, key="kline_symbol")
    with kline_col3:
        timeframe_label = st.selectbox("Time frame", list(KLINE_TIMEFRAMES.keys()), key="kline_timeframe")

    period, interval = KLINE_TIMEFRAMES[timeframe_label]

    @st.cache_data(ttl=int(refresh_minutes * 60))
    def get_candles(symbol: str, period_key: str, interval_key: str) -> pd.DataFrame:
        return fetch_candles(symbol, period_key, interval_key)

    candle_df = get_candles(selected_symbol, period, interval)
    if candle_df.empty:
        st.warning(f"No candlestick data for {selected_symbol}.")
        return

    has_volume = "Volume" in candle_df.columns
    if has_volume:
        fig_k = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )
        fig_k.add_trace(
            go.Candlestick(
                x=candle_df.index,
                open=candle_df["Open"],
                high=candle_df["High"],
                low=candle_df["Low"],
                close=candle_df["Close"],
                name=selected_symbol,
            ),
            row=1,
            col=1,
        )
        fig_k.add_trace(
            go.Bar(
                x=candle_df.index,
                y=candle_df["Volume"],
                name="Volume",
                marker_color="#64748b",
            ),
            row=2,
            col=1,
        )
        fig_k.update_layout(
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            title=f"{selected_symbol} Candlestick ({timeframe_label})",
        )
        fig_k.update_yaxes(title_text="Price", row=1, col=1)
        fig_k.update_yaxes(title_text="Volume", row=2, col=1)
    else:
        fig_k = go.Figure(
            go.Candlestick(
                x=candle_df.index,
                open=candle_df["Open"],
                high=candle_df["High"],
                low=candle_df["Low"],
                close=candle_df["Close"],
                name=selected_symbol,
            )
        )
        fig_k.update_layout(
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            title=f"{selected_symbol} Candlestick ({timeframe_label})",
        )
    st.plotly_chart(fig_k, use_container_width=True)


if __name__ == "__main__":
    main()
