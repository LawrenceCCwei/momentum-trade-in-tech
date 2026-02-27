import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yfinance as yf


def load_sector_map(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


@st.cache_data(ttl=3600)
def fetch_close_data(symbols: Tuple[str, ...], period: str) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame()
    data = yf.download(
        tickers=list(symbols),
        period=period,
        interval="1d",
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


def alpaca_headers(api_key: str, secret_key: str) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "Content-Type": "application/json",
    }


def get_alpaca_account(base_url: str, api_key: str, secret_key: str) -> dict:
    response = requests.get(
        f"{base_url.rstrip('/')}/v2/account",
        headers=alpaca_headers(api_key, secret_key),
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def submit_alpaca_market_orders(
    base_url: str,
    api_key: str,
    secret_key: str,
    trade_plan: pd.DataFrame,
) -> List[dict]:
    results: List[dict] = []
    for _, row in trade_plan.iterrows():
        symbol = str(row["symbol"])
        qty = int(row["shares"])
        if qty <= 0:
            results.append({"symbol": symbol, "status": "skipped", "reason": "shares <= 0"})
            continue
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
        }
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/v2/orders",
                headers=alpaca_headers(api_key, secret_key),
                json=payload,
                timeout=20,
            )
            if response.status_code >= 400:
                results.append(
                    {
                        "symbol": symbol,
                        "status": "error",
                        "http_status": response.status_code,
                        "response": response.text[:300],
                    }
                )
                continue
            order = response.json()
            results.append(
                {
                    "symbol": symbol,
                    "status": "submitted",
                    "order_id": order.get("id"),
                    "filled_avg_price": order.get("filled_avg_price"),
                    "order_status": order.get("status"),
                }
            )
        except requests.RequestException as exc:
            results.append({"symbol": symbol, "status": "error", "response": str(exc)})
    return results


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
        st.divider()
        st.header("Execution")
        execution_mode = st.selectbox(
            "Execution mode",
            ["local_paper", "alpaca_paper"],
            index=0,
            help="local_paper: simulate only, alpaca_paper: submit real paper orders to Alpaca account",
        )
        default_api_key = os.getenv("APCA_API_KEY_ID", "")
        default_secret = os.getenv("APCA_API_SECRET_KEY", "")
        default_base = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        alpaca_api_key = st.text_input("Alpaca API Key", value=default_api_key, type="password")
        alpaca_secret_key = st.text_input("Alpaca Secret Key", value=default_secret, type="password")
        alpaca_base_url = st.text_input("Alpaca Base URL", value=default_base)
        dry_run = st.checkbox("Dry run only (do not send orders)", value=True)

    close_df = fetch_close_data(tuple(symbols), period)
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

    st.subheader("Execution Console")
    if execution_mode == "local_paper":
        st.info("Execution mode is local_paper. No broker API calls will be sent.")
    else:
        if not alpaca_api_key or not alpaca_secret_key:
            st.warning("Enter Alpaca API credentials to use alpaca_paper mode.")
            return
        try:
            account = get_alpaca_account(alpaca_base_url, alpaca_api_key, alpaca_secret_key)
            st.caption(
                "Alpaca account status: "
                f"{account.get('status')} | buying_power={account.get('buying_power')} | cash={account.get('cash')}"
            )
        except requests.RequestException as exc:
            st.error(f"Failed to connect Alpaca account: {exc}")
            return

        submit = st.button("Submit Buy Orders to Alpaca Paper", disabled=trade_plan.empty)
        if submit:
            if dry_run:
                st.info("Dry run enabled. Orders were not sent.")
                st.dataframe(trade_plan[["symbol", "shares", "last_price", "capital_used"]], use_container_width=True)
            else:
                results = submit_alpaca_market_orders(
                    alpaca_base_url,
                    alpaca_api_key,
                    alpaca_secret_key,
                    trade_plan,
                )
                st.dataframe(pd.DataFrame(results), use_container_width=True)


if __name__ == "__main__":
    main()
