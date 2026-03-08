import csv
from pathlib import Path


def compute_rsi(prices, period=2):
    """Compute RSI using Wilder's smoothing method."""
    rsi = [None] * len(prices)
    if len(prices) < period + 1:
        return rsi

    # Initial average gain/loss from first `period` changes
    gains = []
    losses = []
    for i in range(1, period + 1):
        change = prices[i] - prices[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - 100.0 / (1.0 + rs)

    # Wilder's smoothing for subsequent values
    for i in range(period + 1, len(prices)):
        change = prices[i] - prices[i - 1]
        gain = max(change, 0)
        loss = max(-change, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def compute_sma(prices, period):
    """Compute simple moving average."""
    sma = [None] * len(prices)
    for i in range(period - 1, len(prices)):
        sma[i] = sum(prices[i - period + 1 : i + 1]) / period
    return sma


def main():
    data_path = Path(__file__).parent / "spy_daily_close.csv"

    dates = []
    prices = []
    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(row["date"])
            prices.append(float(row["price"]))

    rsi = compute_rsi(prices, period=2)
    sma200 = compute_sma(prices, period=200)

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

    for i in range(200, len(prices)):
        if rsi[i] is None or sma200[i] is None:
            continue

        portfolio_value = cash + shares * prices[i]
        peak_value = max(peak_value, portfolio_value)
        dd = (peak_value - portfolio_value) / peak_value * 100
        max_drawdown = max(max_drawdown, dd)

        if not in_market:
            # Buy when RSI(2) < 10 and price above SMA(200)
            if rsi[i] < 10 and prices[i] > sma200[i]:
                shares = cash / prices[i]
                entry_price = prices[i]
                cash = 0.0
                in_market = True
                trades.append(("BUY", dates[i], prices[i]))
                if first_trade_idx is None:
                    first_trade_idx = i
        else:
            # Sell when RSI(2) > 90
            if rsi[i] > 90:
                cash = shares * prices[i]
                shares = 0.0
                in_market = False
                trades.append(("SELL", dates[i], prices[i]))
                if prices[i] > entry_price:
                    wins += 1
                else:
                    losses += 1

    # Final portfolio value
    last_price = prices[-1]
    last_date = dates[-1]
    portfolio_value = cash + shares * last_price
    strategy_roi = (portfolio_value - initial_capital) / initial_capital * 100

    # Update drawdown for final state
    peak_value = max(peak_value, portfolio_value)
    dd = (peak_value - portfolio_value) / peak_value * 100
    max_drawdown = max(max_drawdown, dd)

    # Buy-and-hold from first trade date
    if first_trade_idx is not None:
        bh_price = prices[first_trade_idx]
        bh_date = dates[first_trade_idx]
    else:
        bh_price = prices[200]
        bh_date = dates[200]

    bh_shares = initial_capital / bh_price
    bh_value = bh_shares * last_price
    bh_roi = (bh_value - initial_capital) / initial_capital * 100

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    # Count days in market
    days_in_market = 0
    currently_in = False
    for i in range(200, len(prices)):
        if rsi[i] is None or sma200[i] is None:
            continue
        if not currently_in:
            if rsi[i] < 10 and prices[i] > sma200[i]:
                currently_in = True
                days_in_market += 1
        else:
            days_in_market += 1
            if rsi[i] > 90:
                currently_in = False
    total_days = len(prices) - 200
    exposure = days_in_market / total_days * 100 if total_days > 0 else 0

    print("=" * 55)
    print("   RSI(2) Mean Reversion Backtest Results")
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
