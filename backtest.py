import csv
from pathlib import Path


def main():
    data_path = Path(__file__).parent / "spy_daily_close.csv"

    rows = []
    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["sma_50"] == "" or row["sma_200"] == "":
                continue
            rows.append(
                {
                    "date": row["date"],
                    "price": float(row["price"]),
                    "sma_50": float(row["sma_50"]),
                    "sma_200": float(row["sma_200"]),
                }
            )

    if len(rows) < 2:
        print("Not enough data with both SMAs available.")
        return

    initial_capital = 10_000.0
    cash = initial_capital
    shares = 0.0
    in_market = False
    first_signal_date = None
    first_signal_price = None
    trades = []

    for i in range(1, len(rows)):
        prev = rows[i - 1]
        curr = rows[i]

        prev_above = prev["sma_50"] > prev["sma_200"]
        curr_above = curr["sma_50"] > curr["sma_200"]

        if not prev_above and curr_above:
            # Golden cross — buy
            if not in_market:
                shares = cash / curr["price"]
                cash = 0.0
                in_market = True
                if first_signal_date is None:
                    first_signal_date = curr["date"]
                    first_signal_price = curr["price"]
                trades.append(("BUY", curr["date"], curr["price"]))

        elif prev_above and not curr_above:
            # Death cross — sell
            if in_market:
                cash = shares * curr["price"]
                shares = 0.0
                in_market = False
                if first_signal_date is None:
                    first_signal_date = curr["date"]
                    first_signal_price = curr["price"]
                trades.append(("SELL", curr["date"], curr["price"]))

    if first_signal_date is None:
        print("No crossover signals found.")
        return

    # Final portfolio value
    last_price = rows[-1]["price"]
    last_date = rows[-1]["date"]
    portfolio_value = cash + shares * last_price
    strategy_roi = (portfolio_value - initial_capital) / initial_capital * 100

    # Buy-and-hold from first signal date
    bh_shares = initial_capital / first_signal_price
    bh_value = bh_shares * last_price
    bh_roi = (bh_value - initial_capital) / initial_capital * 100

    print("=" * 55)
    print("   SMA Golden/Death Cross Backtest Results")
    print("=" * 55)
    print(f"  First signal date:  {first_signal_date}")
    print(f"  Last date:          {last_date}")
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
    print("=" * 55)
    print()
    print(f"  Total trades: {len(trades)}")
    status = "Invested" if in_market else "Cash"
    print(f"  Final position: {status}")
    print()
    print("  Trade log:")
    for action, date, price in trades:
        print(f"    {date}  {action:4s}  @ ${price:,.2f}")


if __name__ == "__main__":
    main()
