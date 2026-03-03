import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

EMA_WINDOWS = [5, 10, 20, 60, 120, 240]
TOUCH_THRESHOLDS = [0.0, 0.0025, 0.005, 0.01, 0.02]
HOLD_DAYS = 21
SECTOR_MA_WINDOWS = [20, 60, 120, 240]


def load_sector_map(path: Path) -> Dict[str, List[str]]:
    with path.open("r", encoding="ascii") as handle:
        return json.load(handle)


def fetch_close_data(symbols: List[str], period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    data = yf.download(
        tickers=symbols,
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


def fetch_close_volume_data(
    symbols: List[str],
    period: str = "5y",
    interval: str = "1d",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    data = yf.download(
        tickers=symbols,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close_df = data.xs("Close", axis=1, level=1, drop_level=False)
        close_df.columns = [c[0] for c in close_df.columns]
        volume_df = data.xs("Volume", axis=1, level=1, drop_level=False)
        volume_df.columns = [c[0] for c in volume_df.columns]
    else:
        close_df = pd.DataFrame({"SINGLE": data["Close"]})
        volume_df = pd.DataFrame({"SINGLE": data["Volume"]})
    close_df = close_df.sort_index().dropna(how="all")
    volume_df = volume_df.sort_index().reindex(close_df.index).dropna(how="all")
    return close_df, volume_df


def build_sector_returns(close_df: pd.DataFrame, sector_map: Dict[str, List[str]]) -> pd.DataFrame:
    symbol_returns = close_df.pct_change(fill_method=None)
    sector_returns = pd.DataFrame(index=symbol_returns.index)
    for sector, symbols in sector_map.items():
        members = [s for s in symbols if s in symbol_returns.columns]
        if not members:
            continue
        sector_returns[sector] = symbol_returns[members].mean(axis=1, skipna=True)
    return sector_returns.dropna(how="all")


def calc_metrics(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
    strategy_returns = strategy_returns.dropna()
    benchmark_returns = benchmark_returns.reindex(strategy_returns.index).fillna(0.0)
    if strategy_returns.empty:
        return {}

    strat_equity = (1.0 + strategy_returns).cumprod()
    bench_equity = (1.0 + benchmark_returns).cumprod()
    span_days = max((strat_equity.index[-1] - strat_equity.index[0]).days, 1)
    years = span_days / 365.25
    total_return = float(strat_equity.iloc[-1] - 1.0)
    cagr = float(strat_equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(strategy_returns.std(ddof=0))
    sharpe = float((strategy_returns.mean() / vol) * np.sqrt(252)) if vol > 0 else 0.0
    downside = strategy_returns[strategy_returns < 0]
    downside_vol = float(downside.std(ddof=0)) if not downside.empty else 0.0
    sortino = float((strategy_returns.mean() / downside_vol) * np.sqrt(252)) if downside_vol > 0 else 0.0
    drawdown = strat_equity / strat_equity.cummax() - 1.0
    max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
    win_rate = float((strategy_returns > 0).mean())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "bench_total_return": float(bench_equity.iloc[-1] - 1.0),
    }


def compute_sector_risk_metrics(sector_returns: pd.DataFrame) -> pd.DataFrame:
    if sector_returns.empty:
        return pd.DataFrame()
    rows: List[dict] = []
    for sector in sector_returns.columns:
        r = sector_returns[sector].dropna()
        if r.empty:
            continue
        vol = float(r.std(ddof=0))
        sharpe = float((r.mean() / vol) * np.sqrt(252)) if vol > 0 else 0.0
        downside = r[r < 0]
        downside_vol = float(downside.std(ddof=0)) if not downside.empty else 0.0
        sortino = float((r.mean() / downside_vol) * np.sqrt(252)) if downside_vol > 0 else 0.0
        equity = (1.0 + r).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        max_dd = float(drawdown.min()) if not drawdown.empty else 0.0
        rows.append(
            {
                "sector": sector,
                "sharpe": sharpe,
                "sortino": sortino,
                "max_drawdown": max_dd,
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)


def backtest_sector_momentum(
    sector_returns: pd.DataFrame,
    lookback: int,
    ma_window: int,
    top_k: int,
    rebalance_bars: int,
    positive_score_only: bool = True,
) -> Tuple[pd.Series, pd.Series]:
    sector_prices = (1.0 + sector_returns.fillna(0.0)).cumprod()
    ma_df = sector_prices.rolling(ma_window).mean()
    momentum_df = sector_prices / sector_prices.shift(lookback) - 1.0

    strategy_returns = pd.Series(0.0, index=sector_returns.index)
    i = max(lookback, ma_window)
    while i < len(sector_returns.index) - 1:
        date_i = sector_returns.index[i]
        momentum = momentum_df.loc[date_i]
        above_ma = (sector_prices.loc[date_i] >= ma_df.loc[date_i]).astype(float)
        score = 0.7 * momentum + 0.3 * above_ma
        score = score.dropna().sort_values(ascending=False)
        if positive_score_only:
            score = score[score > 0]
        picks = list(score.head(top_k).index)

        end_i = min(i + rebalance_bars, len(sector_returns.index) - 1)
        for j in range(i + 1, end_i + 1):
            date_j = sector_returns.index[j]
            if picks:
                day_ret = sector_returns.loc[date_j, picks].mean(skipna=True)
                strategy_returns.loc[date_j] = 0.0 if np.isnan(day_ret) else float(day_ret)
            else:
                strategy_returns.loc[date_j] = 0.0
        i += rebalance_bars

    benchmark = sector_returns.mean(axis=1, skipna=True).reindex(strategy_returns.index).fillna(0.0)
    return strategy_returns, benchmark


def run_grid_search(train_sector_returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lookbacks = [20, 40, 60, 90]
    ma_windows = [20, 60, 120]
    top_ks = [2, 3, 4, 5]
    rebalances = [5, 10]

    for lookback in lookbacks:
        for ma_window in ma_windows:
            for top_k in top_ks:
                for rebalance in rebalances:
                    strat, bench = backtest_sector_momentum(
                        train_sector_returns,
                        lookback=lookback,
                        ma_window=ma_window,
                        top_k=top_k,
                        rebalance_bars=rebalance,
                        positive_score_only=True,
                    )
                    metrics = calc_metrics(strat, bench)
                    if not metrics:
                        continue
                    rows.append(
                        {
                            "lookback": lookback,
                            "ma_window": ma_window,
                            "top_k": top_k,
                            "rebalance_bars": rebalance,
                            **metrics,
                        }
                    )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["win_rate", "sharpe"], ascending=False).reset_index(drop=True)


def select_best_params(grid_df: pd.DataFrame) -> dict:
    filtered = grid_df[grid_df["sharpe"] > 0.5].copy()
    if filtered.empty:
        filtered = grid_df.copy()
    best = filtered.sort_values(["win_rate", "sharpe", "cagr"], ascending=False).iloc[0]
    return {
        "lookback": int(best["lookback"]),
        "ma_window": int(best["ma_window"]),
        "top_k": int(best["top_k"]),
        "rebalance_bars": int(best["rebalance_bars"]),
    }


def plot_research(
    output_dir: Path,
    grid_df: pd.DataFrame,
    curve_df: pd.DataFrame,
    best_params: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 6))
    x = pd.to_datetime(curve_df["date"]).to_numpy()
    plt.plot(x, curve_df["strategy_equity"].to_numpy(), label="Strategy", linewidth=2)
    plt.plot(x, curve_df["benchmark_equity"].to_numpy(), label="Benchmark", linestyle="--")
    plt.title(
        "Equity Curve (Test) | "
        f"L{best_params['lookback']} MA{best_params['ma_window']} "
        f"K{best_params['top_k']} R{best_params['rebalance_bars']}"
    )
    plt.xlabel("Date")
    plt.ylabel("Equity (Start=1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "equity_curve_test.png", dpi=140)
    plt.close()

    plt.figure(figsize=(11, 4))
    x = pd.to_datetime(curve_df["date"]).to_numpy()
    plt.fill_between(x, curve_df["drawdown"].to_numpy(), 0, alpha=0.3)
    plt.title("Drawdown (Test)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(output_dir / "drawdown_test.png", dpi=140)
    plt.close()


def compute_sector_correlation(sector_returns: pd.DataFrame) -> pd.DataFrame:
    if sector_returns.empty:
        return pd.DataFrame()
    return sector_returns.corr(method="pearson")


def plot_sector_correlation(output_dir: Path, corr_df: pd.DataFrame) -> None:
    if corr_df.empty:
        return
    labels = corr_df.columns.tolist()
    values = corr_df.values

    plt.figure(figsize=(12, 10))
    im = plt.imshow(values, cmap="RdBu_r", vmin=-1.0, vmax=1.0, aspect="auto")
    plt.xticks(range(len(labels)), labels, rotation=75, ha="right", fontsize=8)
    plt.yticks(range(len(labels)), labels, fontsize=8)
    plt.title("Sector Return Correlation Matrix")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlation")
    plt.tight_layout()
    plt.savefig(output_dir / "sector_correlation_heatmap.png", dpi=150)
    plt.close()


def top_correlation_pairs(corr_df: pd.DataFrame, top_n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if corr_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    pairs: List[dict] = []
    cols = corr_df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = cols[i]
            b = cols[j]
            c = float(corr_df.iloc[i, j])
            pairs.append({"sector_a": a, "sector_b": b, "correlation": c})
    if not pairs:
        return pd.DataFrame(), pd.DataFrame()
    pair_df = pd.DataFrame(pairs).sort_values("correlation", ascending=False).reset_index(drop=True)
    highest = pair_df.head(top_n).copy()
    lowest = pair_df.tail(top_n).sort_values("correlation", ascending=True).reset_index(drop=True)
    return highest, lowest


def build_last_year_sector_equity(sector_returns: pd.DataFrame) -> pd.DataFrame:
    if sector_returns.empty:
        return pd.DataFrame()
    end_date = pd.to_datetime(sector_returns.index.max())
    start_date = end_date - pd.Timedelta(days=365)
    last_year = sector_returns[sector_returns.index >= start_date].copy()
    if last_year.empty:
        return pd.DataFrame()
    equity = (1.0 + last_year.fillna(0.0)).cumprod()
    return equity


def compute_weighted_sector_price(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    sector_map: Dict[str, List[str]],
) -> pd.DataFrame:
    if close_df.empty or volume_df.empty:
        return pd.DataFrame()
    rows = {}
    for sector, symbols in sector_map.items():
        members = [s for s in symbols if s in close_df.columns and s in volume_df.columns]
        if not members:
            continue
        c = close_df[members]
        v = volume_df[members]
        turnover = (c * v).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        total_turnover = turnover.sum(axis=1)
        valid = total_turnover > 0
        weighted_price = pd.Series(np.nan, index=close_df.index, dtype=float)
        if valid.any():
            weights = turnover.loc[valid].div(total_turnover.loc[valid], axis=0)
            weighted_price.loc[valid] = (weights * c.loc[valid]).sum(axis=1)
        rows[sector] = weighted_price
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).dropna(how="all")


def build_weighted_sector_ma(
    weighted_price_df: pd.DataFrame,
    ma_windows: List[int],
) -> Dict[str, pd.DataFrame]:
    output: Dict[str, pd.DataFrame] = {}
    if weighted_price_df.empty:
        return output
    for sector in weighted_price_df.columns:
        s = weighted_price_df[sector].dropna()
        if s.empty:
            continue
        df = pd.DataFrame({"date": s.index, "weighted_price": s.values})
        for w in ma_windows:
            df[f"ma_{w}"] = s.rolling(w).mean().values
        output[sector] = df
    return output


def plot_weighted_sector_ma(
    output_dir: Path,
    weighted_ma_data: Dict[str, pd.DataFrame],
    ma_windows: List[int],
) -> None:
    out = output_dir / "sector_weighted_ma"
    out.mkdir(parents=True, exist_ok=True)
    for sector, df in weighted_ma_data.items():
        if df.empty:
            continue
        x = pd.to_datetime(df["date"]).to_numpy()
        plt.figure(figsize=(11, 5))
        plt.plot(x, df["weighted_price"].to_numpy(), label="Weighted Price", linewidth=2, color="#1f77b4")
        for w in ma_windows:
            col = f"ma_{w}"
            if col in df.columns:
                plt.plot(x, df[col].to_numpy(), label=f"MA{w}", linewidth=1.2)
        plt.title(f"{sector} | Turnover-Weighted Price & MAs")
        plt.xlabel("Date")
        plt.ylabel("Weighted Price")
        plt.legend(fontsize=8, ncol=3)
        plt.tight_layout()
        plt.savefig(out / f"{sanitize_filename(sector)}_weighted_ma.png", dpi=150)
        plt.close()
        df.to_csv(out / f"{sanitize_filename(sector)}_weighted_ma.csv", index=False)


def plot_last_year_sector_equity(output_dir: Path, equity_df: pd.DataFrame) -> None:
    if equity_df.empty:
        return
    plt.figure(figsize=(13, 7))
    x = pd.to_datetime(equity_df.index).to_numpy()
    for col in equity_df.columns:
        plt.plot(x, equity_df[col].to_numpy(), linewidth=1.6, label=col)
    plt.title("Sector Equity Curves (Last 1 Year)")
    plt.xlabel("Date")
    plt.ylabel("Equity (Start=1)")
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(output_dir / "sector_equity_curve_last_1y.png", dpi=150)
    plt.close()


def collect_touch_returns(
    close_series: pd.Series,
    ema_window: int,
    threshold: float,
    hold_days: int = HOLD_DAYS,
) -> pd.Series:
    close_series = close_series.dropna()
    if close_series.empty or len(close_series) <= max(ema_window, hold_days) + 1:
        return pd.Series(dtype=float)

    ema = close_series.ewm(span=ema_window, adjust=False).mean()
    dist = (close_series - ema).abs() / ema
    touch = dist <= threshold
    first_touch = touch & (~touch.shift(1).fillna(False))

    future_ret = close_series.shift(-hold_days) / close_series - 1.0
    trades = future_ret[first_touch].dropna()
    return trades


def run_touch_study_by_sector(
    close_df: pd.DataFrame,
    sector_map: Dict[str, List[str]],
    ema_windows: List[int],
    thresholds: List[float],
    hold_days: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    for sector, symbols in sector_map.items():
        members = [s for s in symbols if s in close_df.columns]
        if not members:
            continue
        for ema_w in ema_windows:
            for thr in thresholds:
                trade_returns = []
                for symbol in members:
                    s = close_df[symbol].dropna()
                    if s.empty:
                        continue
                    r = collect_touch_returns(s, ema_w, thr, hold_days=hold_days)
                    if not r.empty:
                        trade_returns.extend(r.tolist())

                if not trade_returns:
                    rows.append(
                        {
                            "sector": sector,
                            "ema_window": ema_w,
                            "threshold": thr,
                            "trades": 0,
                            "win_rate": np.nan,
                            "avg_profit": np.nan,
                            "median_profit": np.nan,
                        }
                    )
                    continue

                ret_ser = pd.Series(trade_returns)
                rows.append(
                    {
                        "sector": sector,
                        "ema_window": ema_w,
                        "threshold": thr,
                        "trades": int(len(ret_ser)),
                        "win_rate": float((ret_ser > 0).mean()),
                        "avg_profit": float(ret_ser.mean()),
                        "median_profit": float(ret_ser.median()),
                    }
                )
    return pd.DataFrame(rows)


def sanitize_filename(name: str) -> str:
    allowed = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            allowed.append(ch)
        elif ch == " ":
            allowed.append("_")
    txt = "".join(allowed).strip("_")
    return txt or "sector"


def plot_sector_touch_study(
    output_dir: Path,
    touch_df: pd.DataFrame,
    thresholds: List[float],
    ema_windows: List[int],
) -> None:
    out = output_dir / "touch_study"
    out.mkdir(parents=True, exist_ok=True)
    sectors = sorted(touch_df["sector"].dropna().unique().tolist())
    threshold_labels = [f"{t*100:.2f}%" for t in thresholds]

    for sector in sectors:
        sdf = touch_df[touch_df["sector"] == sector].copy()
        if sdf.empty:
            continue

        pf_pivot = sdf.pivot(index="ema_window", columns="threshold", values="avg_profit").reindex(
            index=ema_windows,
            columns=thresholds,
        )

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        vmax = np.nanmax(np.abs(pf_pivot.values)) if np.isfinite(np.nanmax(np.abs(pf_pivot.values))) else 0.05
        vmax = max(vmax, 0.01)
        im = ax.imshow(
            pf_pivot.values,
            aspect="auto",
            cmap="RdYlGn",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(f"{sector} | Avg Profit (Hold {HOLD_DAYS} bars)")
        ax.set_xticks(range(len(threshold_labels)))
        ax.set_xticklabels(threshold_labels, rotation=45, ha="right")
        ax.set_yticks(range(len(ema_windows)))
        ax.set_yticklabels([f"EMA{w}" for w in ema_windows])
        ax.set_xlabel("Touch Threshold")
        ax.set_ylabel("EMA Window")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f"Touch-Entry Study | {sector}", y=1.03, fontsize=12)
        plt.tight_layout()
        plt.savefig(out / f"{sanitize_filename(sector)}_touch_study.png", dpi=150, bbox_inches="tight")
        plt.close()


def main() -> None:
    sector_path = Path("sectors.json")
    output_dir = Path("research_outputs")

    sector_map = load_sector_map(sector_path)
    symbols = sorted({s for arr in sector_map.values() for s in arr})
    close_df, volume_df = fetch_close_volume_data(symbols=symbols, period="5y", interval="1d")
    if close_df.empty:
        print("No market data fetched. Please retry later.")
        return

    sector_returns = build_sector_returns(close_df, sector_map)
    if len(sector_returns) < 300:
        print("Not enough sector return history for robust research.")
        return

    split_idx = int(len(sector_returns) * 0.7)
    train_returns = sector_returns.iloc[:split_idx].copy()
    test_returns = sector_returns.iloc[split_idx:].copy()

    grid_df = run_grid_search(train_returns)
    if grid_df.empty:
        print("Grid search failed to produce results.")
        return

    best_params = select_best_params(grid_df)
    test_strat, test_bench = backtest_sector_momentum(
        test_returns,
        lookback=best_params["lookback"],
        ma_window=best_params["ma_window"],
        top_k=best_params["top_k"],
        rebalance_bars=best_params["rebalance_bars"],
        positive_score_only=True,
    )
    metrics = calc_metrics(test_strat, test_bench)

    curve_df = pd.DataFrame(
        {
            "date": test_strat.index,
            "strategy_equity": (1.0 + test_strat.fillna(0.0)).cumprod().values,
            "benchmark_equity": (1.0 + test_bench.fillna(0.0)).cumprod().values,
        }
    )
    curve_df["drawdown"] = curve_df["strategy_equity"] / curve_df["strategy_equity"].cummax() - 1.0

    output_dir.mkdir(parents=True, exist_ok=True)
    grid_df.to_csv(output_dir / "grid_search_results.csv", index=False)
    curve_df.to_csv(output_dir / "test_equity_curve.csv", index=False)
    plot_research(output_dir, grid_df, curve_df, best_params)

    touch_df = run_touch_study_by_sector(
        close_df=close_df,
        sector_map=sector_map,
        ema_windows=EMA_WINDOWS,
        thresholds=TOUCH_THRESHOLDS,
        hold_days=HOLD_DAYS,
    )
    touch_df.to_csv(output_dir / "touch_study_by_sector.csv", index=False)
    if not touch_df.empty:
        plot_sector_touch_study(output_dir, touch_df, TOUCH_THRESHOLDS, EMA_WINDOWS)

    corr_df = compute_sector_correlation(sector_returns)
    corr_df.to_csv(output_dir / "sector_correlation_matrix.csv", index=True)
    plot_sector_correlation(output_dir, corr_df)
    top_corr, low_corr = top_correlation_pairs(corr_df, top_n=8)
    if not top_corr.empty:
        top_corr.to_csv(output_dir / "sector_correlation_top_pairs.csv", index=False)
    if not low_corr.empty:
        low_corr.to_csv(output_dir / "sector_correlation_low_pairs.csv", index=False)

    sector_risk_df = compute_sector_risk_metrics(sector_returns)
    if not sector_risk_df.empty:
        sector_risk_df.to_csv(output_dir / "sector_risk_metrics.csv", index=False)

    sector_equity_1y = build_last_year_sector_equity(sector_returns)
    if not sector_equity_1y.empty:
        sector_equity_1y.to_csv(output_dir / "sector_equity_curve_last_1y.csv", index=True)
        plot_last_year_sector_equity(output_dir, sector_equity_1y)

    weighted_sector_price = compute_weighted_sector_price(close_df, volume_df, sector_map)
    weighted_ma_data = build_weighted_sector_ma(weighted_sector_price, SECTOR_MA_WINDOWS)
    if weighted_ma_data:
        plot_weighted_sector_ma(output_dir, weighted_ma_data, SECTOR_MA_WINDOWS)

    print("=== Research Summary ===")
    print("Best params (train):", best_params)
    print("Test metrics:")
    for k, v in metrics.items():
        if "rate" in k or "cagr" in k or "drawdown" in k or "return" in k:
            print(f"  {k}: {v:.2%}")
        else:
            print(f"  {k}: {v:.3f}")
    print("Touch-entry study:")
    print(f"  ema_windows: {EMA_WINDOWS}")
    print(f"  thresholds: {[f'{x*100:.2f}%' for x in TOUCH_THRESHOLDS]}")
    print(f"  hold_days: {HOLD_DAYS}")
    if not top_corr.empty:
        print("Top correlation pairs:")
        for row in top_corr.itertuples(index=False):
            print(f"  {row.sector_a} <-> {row.sector_b}: {row.correlation:.3f}")
    if not low_corr.empty:
        print("Low correlation pairs:")
        for row in low_corr.itertuples(index=False):
            print(f"  {row.sector_a} <-> {row.sector_b}: {row.correlation:.3f}")
    if not sector_risk_df.empty:
        print("Sector risk metrics saved: sharpe, sortino, max_drawdown")
    if not sector_equity_1y.empty:
        print(f"Last 1Y sector equity curves generated for {len(sector_equity_1y.columns)} sectors.")
    if weighted_ma_data:
        print(
            "Turnover-weighted sector MA charts generated "
            f"for {len(weighted_ma_data)} sectors (MA windows: {SECTOR_MA_WINDOWS})."
        )
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
