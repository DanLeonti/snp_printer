"""
Dual Momentum (GEM - Global Equities Momentum) Backtest

Requires: pip install yfinance
Fetches monthly data for SPY, EFA, AGG and applies Antonacci's dual momentum rules.
"""

import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Run: pip install yfinance")
    sys.exit(1)


def get_monthly_closes(ticker, start="1996-01-01", end="2026-03-08"):
    """Fetch daily data and extract month-end closing prices."""
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        return {}
    # Get last trading day of each month
    monthly = {}
    for date, row in df.iterrows():
        date_str = date.strftime("%Y-%m-%d")
        ym = (date.year, date.month)
        # Keep updating — last one per month wins (= month-end close)
        close_val = row["Close"]
        # yfinance may return a Series; extract scalar
        if hasattr(close_val, "item"):
            close_val = close_val.item()
        monthly[ym] = float(close_val)
    return monthly


def main():
    print("Fetching data for SPY, EFA, AGG...")
    spy_monthly = get_monthly_closes("SPY")
    efa_monthly = get_monthly_closes("EFA")
    agg_monthly = get_monthly_closes("AGG")

    if not spy_monthly or not efa_monthly or not agg_monthly:
        print("ERROR: Could not fetch data for one or more tickers.")
        return

    # Find common months where all three have data
    common_months = sorted(
        set(spy_monthly.keys()) & set(efa_monthly.keys()) & set(agg_monthly.keys())
    )

    if len(common_months) < 13:
        print("Not enough common data (need at least 13 months).")
        return

    initial_capital = 10_000.0
    capital = initial_capital
    current_holding = None  # "SPY", "EFA", "AGG", or None
    first_trade_month = None
    trades = []
    peak_value = initial_capital
    max_drawdown = 0.0

    # Equity curve data
    equity_dates = []
    equity_strategy = []
    equity_buyhold = []

    # Start from month 12 (need 12-month lookback)
    for i in range(12, len(common_months)):
        current_ym = common_months[i]
        lookback_ym = common_months[i - 12]

        spy_ret_12m = (spy_monthly[current_ym] / spy_monthly[lookback_ym]) - 1
        efa_ret_12m = (efa_monthly[current_ym] / efa_monthly[lookback_ym]) - 1

        # Step 1: Relative momentum — pick the winner
        if spy_ret_12m >= efa_ret_12m:
            winner = "SPY"
            winner_ret = spy_ret_12m
        else:
            winner = "EFA"
            winner_ret = efa_ret_12m

        # Step 2: Absolute momentum — if winner has positive return, invest; else bonds
        if winner_ret > 0:
            target = winner
        else:
            target = "AGG"

        # Rebalance if needed
        if target != current_holding:
            if first_trade_month is None:
                first_trade_month = i

            # Get price for the target asset this month
            price_map = {"SPY": spy_monthly, "EFA": efa_monthly, "AGG": agg_monthly}

            trades.append(
                (
                    f"{current_ym[0]}-{current_ym[1]:02d}",
                    f"{'SELL ' + current_holding if current_holding else 'START'} -> BUY {target}",
                )
            )
            current_holding = target

        # Track portfolio value using SPY as reference for value
        # In real GEM, capital grows with the held asset
        # We need to track month-over-month returns
        if i > 12 and current_holding:
            prev_ym = common_months[i - 1]
            price_map = {"SPY": spy_monthly, "EFA": efa_monthly, "AGG": agg_monthly}
            prev_price = price_map[current_holding].get(prev_ym)
            curr_price = price_map[current_holding].get(current_ym)
            if prev_price and curr_price and prev_price > 0:
                monthly_ret = curr_price / prev_price
                capital *= monthly_ret

        # Drawdown tracking
        peak_value = max(peak_value, capital)
        dd = (peak_value - capital) / peak_value * 100
        max_drawdown = max(max_drawdown, dd)

        # Record equity curve data point
        equity_dates.append(f"{current_ym[0]}-{current_ym[1]:02d}")
        equity_strategy.append(capital)

    strategy_roi = (capital - initial_capital) / initial_capital * 100

    # Buy-and-hold SPY from first trade month
    if first_trade_month is not None:
        bh_start_ym = common_months[first_trade_month]
        bh_end_ym = common_months[-1]
        bh_start_price = spy_monthly[bh_start_ym]
        bh_end_price = spy_monthly[bh_end_ym]
        bh_roi = (bh_end_price / bh_start_price - 1) * 100
        bh_value = initial_capital * (1 + bh_roi / 100)
        start_label = f"{bh_start_ym[0]}-{bh_start_ym[1]:02d}"
    else:
        bh_roi = 0
        bh_value = initial_capital
        start_label = "N/A"

    end_label = f"{common_months[-1][0]}-{common_months[-1][1]:02d}"

    print()
    print("=" * 55)
    print("   Dual Momentum (GEM) Backtest Results")
    print("=" * 55)
    print(f"  Period:             {start_label} to {end_label}")
    print(f"  Initial capital:    ${initial_capital:,.2f}")
    print()
    print(f"  Strategy final value:   ${capital:,.2f}")
    print(f"  Strategy ROI:           {strategy_roi:,.2f}%")
    print()
    print(f"  Buy & Hold final value: ${bh_value:,.2f}")
    print(f"  Buy & Hold ROI:         {bh_roi:,.2f}%")
    print()
    diff = strategy_roi - bh_roi
    winner = "Strategy" if diff > 0 else "Buy & Hold"
    print(f"  Difference:             {diff:+,.2f}% ({winner} wins)")
    print("-" * 55)
    print(f"  Total rebalances:       {len(trades)}")
    print(f"  Max drawdown:           {max_drawdown:.2f}%")
    print(f"  Current holding:        {current_holding}")
    print("=" * 55)
    print()
    print("  Rebalance log (last 20):")
    for date, action in trades[-20:]:
        print(f"    {date}  {action}")

    # --- Build buy-and-hold equity curve ---
    if first_trade_month is not None:
        bh_start_price = spy_monthly[common_months[12]]
        for i in range(12, len(common_months)):
            ym = common_months[i]
            bh_val = initial_capital * (spy_monthly[ym] / bh_start_price)
            equity_buyhold.append(bh_val)

    # --- Plot equity curves ---
    if equity_dates and equity_buyhold:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(range(len(equity_dates)), equity_strategy, color="#1f77b4",
                linewidth=1.5, label="Dual Momentum (GEM)")
        ax.plot(range(len(equity_dates)), equity_buyhold, color="#999999",
                linewidth=1.5, label="Buy & Hold (SPY)")

        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x:,.0f}"))
        ax.set_ylabel("Portfolio Value (log scale)")
        ax.set_xlabel("Date")
        ax.set_title("Dual Momentum (GEM) vs Buy & Hold (SPY)")

        # X-axis: show label every ~24 months
        step = max(1, len(equity_dates) // 15)
        tick_positions = list(range(0, len(equity_dates), step))
        tick_labels = [equity_dates[j] for j in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)

        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        chart_path = Path(__file__).parent / "dual_momentum_chart.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"\n  Chart saved to {chart_path}")


if __name__ == "__main__":
    main()
