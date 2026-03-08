import csv
from pathlib import Path
from collections import defaultdict


def main():
    data_path = Path(__file__).parent / "spy_daily_close.csv"

    dates = []
    prices = []
    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(row["date"])
            prices.append(float(row["price"]))

    # Group trading days by year-month
    month_days = defaultdict(list)  # (year, month) -> [(idx, date, price)]
    for i, (date, price) in enumerate(zip(dates, prices)):
        year, month, day = date.split("-")
        key = (int(year), int(month))
        month_days[key].append((i, date, price))

    sorted_months = sorted(month_days.keys())

    initial_capital = 10_000.0
    cash = initial_capital
    shares = 0.0
    in_market = False
    trades = []
    wins = 0
    losses = 0
    entry_price = 0.0
    peak_value = initial_capital
    max_drawdown = 0.0
    first_trade_idx = None
    days_in_market = 0

    # For each month, identify entry (5th-to-last trading day) and exit (3rd trading day of next month)
    for mi in range(len(sorted_months) - 1):
        curr_month = sorted_months[mi]
        next_month = sorted_months[mi + 1]

        curr_days = month_days[curr_month]
        next_days = month_days[next_month]

        if len(curr_days) < 5:
            continue
        if len(next_days) < 3:
            continue

        # Entry: 5th-to-last trading day of current month
        entry_idx, entry_date, entry_px = curr_days[-5]
        # Exit: 3rd trading day of next month
        exit_idx, exit_date, exit_px = next_days[2]

        # Update drawdown for days before entry
        portfolio_value = cash + shares * entry_px
        peak_value = max(peak_value, portfolio_value)
        dd = (peak_value - portfolio_value) / peak_value * 100
        max_drawdown = max(max_drawdown, dd)

        if not in_market:
            shares = cash / entry_px
            entry_price = entry_px
            cash = 0.0
            in_market = True
            trades.append(("BUY", entry_date, entry_px))
            if first_trade_idx is None:
                first_trade_idx = entry_idx

        # Count trading days in this holding period
        days_in_market += exit_idx - entry_idx

        # Sell at close of 3rd trading day of next month
        cash = shares * exit_px
        shares = 0.0
        in_market = False
        trades.append(("SELL", exit_date, exit_px))

        if exit_px > entry_price:
            wins += 1
        else:
            losses += 1

        # Update drawdown
        portfolio_value = cash
        peak_value = max(peak_value, portfolio_value)
        dd = (peak_value - portfolio_value) / peak_value * 100
        max_drawdown = max(max_drawdown, dd)

    # Final values
    last_price = prices[-1]
    last_date = dates[-1]
    portfolio_value = cash + shares * last_price
    strategy_roi = (portfolio_value - initial_capital) / initial_capital * 100

    # Buy-and-hold from first trade
    if first_trade_idx is not None:
        bh_price = prices[first_trade_idx]
        bh_date = dates[first_trade_idx]
    else:
        bh_price = prices[0]
        bh_date = dates[0]

    bh_shares = initial_capital / bh_price
    bh_value = bh_shares * last_price
    bh_roi = (bh_value - initial_capital) / initial_capital * 100

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_days = len(prices)
    exposure = days_in_market / total_days * 100 if total_days > 0 else 0

    print("=" * 55)
    print("   Turn-of-Month (Ultimo) Backtest Results")
    print("=" * 55)
    print(f"  Period:             {bh_date} to {last_date}")
    print(f"  Initial capital:    ${initial_capital:,.2f}")
    print()
    print(f"  Strategy final value:   ${portfolio_value:,.2f}")
    print(f"  Strategy ROI:           {strategy_roi:,.2f}%")
    print()
    print(f"  Buy & Hold final value: ${bh_value:,.2f}")
    print(f"  Buy & Hold ROI:         {bh_roi:,.2f}%")
    print()
    diff = strategy_roi - bh_roi
    winner = "Strategy" if diff > 0 else "Buy & Hold"
    print(f"  Difference:             {diff:+,.2f}% ({winner} wins)")
    print("-" * 55)
    print(f"  Total round-trip trades: {total_trades}")
    print(f"  Win rate:               {win_rate:.1f}%")
    print(f"  Max drawdown:           {max_drawdown:.2f}%")
    print(f"  Market exposure:        {exposure:.1f}%")
    status = "Invested" if in_market else "Cash"
    print(f"  Final position:         {status}")
    print("=" * 55)
    print()
    print("  Trade log (last 20):")
    for action, date, price in trades[-20:]:
        print(f"    {date}  {action:4s}  @ ${price:,.2f}")


if __name__ == "__main__":
    main()
