"""
ML Regime-Adaptive Backtest (XGBoost + Technical Indicators)

Requires: pip install xgboost scikit-learn
Uses SPY daily data to predict next-day direction with a regime filter.
Walk-forward: train on rolling 4-year window, predict next 1-year, slide forward.
"""

import csv
import sys
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
except ImportError:
    print("ERROR: Missing dependencies. Run: pip install xgboost scikit-learn numpy")
    sys.exit(1)


def load_data():
    data_path = Path(__file__).parent / "spy_daily_close.csv"
    dates = []
    prices = []
    with open(data_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            dates.append(row["date"])
            prices.append(float(row["price"]))
    return dates, np.array(prices)


def compute_ema(prices, period):
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    mult = 2.0 / (period + 1)
    for i in range(1, len(prices)):
        ema[i] = prices[i] * mult + ema[i - 1] * (1 - mult)
    return ema


def compute_rsi(prices, period=14):
    rsi = np.full(len(prices), np.nan)
    deltas = np.diff(prices)
    gains = np.maximum(deltas, 0)
    losses = np.maximum(-deltas, 0)

    if len(gains) < period:
        return rsi

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rsi[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return rsi


def compute_features(prices):
    """Compute all technical features from price array."""
    n = len(prices)

    ema50 = compute_ema(prices, 50)
    ema200 = compute_ema(prices, 200)
    rsi14 = compute_rsi(prices, 14)

    # MACD
    ema12 = compute_ema(prices, 12)
    ema26 = compute_ema(prices, 26)
    macd = ema12 - ema26
    macd_signal = compute_ema(macd, 9)
    macd_hist = macd - macd_signal

    # Bollinger Bands (20, 2)
    bb_mid = np.full(n, np.nan)
    bb_upper = np.full(n, np.nan)
    bb_lower = np.full(n, np.nan)
    bb_pct = np.full(n, np.nan)
    for i in range(19, n):
        window = prices[i - 19 : i + 1]
        mid = np.mean(window)
        std = np.std(window, ddof=0)
        bb_mid[i] = mid
        bb_upper[i] = mid + 2 * std
        bb_lower[i] = mid - 2 * std
        if std > 0:
            bb_pct[i] = (prices[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i])

    # ATR(14) - using close-to-close as proxy (no high/low data)
    atr = np.full(n, np.nan)
    tr = np.abs(np.diff(prices))
    for i in range(14, len(tr) + 1):
        atr[i] = np.mean(tr[i - 14 : i])

    # Rolling volatility (20-day)
    vol20 = np.full(n, np.nan)
    daily_ret = np.zeros(n)
    daily_ret[1:] = np.diff(prices) / prices[:-1]
    for i in range(20, n):
        vol20[i] = np.std(daily_ret[i - 19 : i + 1], ddof=1)

    # Regime filter: 20-day rolling return
    regime = np.full(n, np.nan)
    for i in range(20, n):
        regime[i] = (prices[i] / prices[i - 20]) - 1

    # Price relative to EMA
    price_to_ema50 = prices / ema50 - 1
    price_to_ema200 = prices / ema200 - 1

    # Next-day return (target)
    next_day_ret = np.zeros(n)
    next_day_ret[:-1] = np.diff(prices) / prices[:-1]
    target = (next_day_ret > 0).astype(int)

    # Build feature matrix
    feature_names = [
        "price_to_ema50",
        "price_to_ema200",
        "rsi14",
        "macd",
        "macd_hist",
        "bb_pct",
        "atr",
        "vol20",
        "regime",
        "daily_ret",
    ]

    features = np.column_stack(
        [price_to_ema50, price_to_ema200, rsi14, macd, macd_hist, bb_pct, atr, vol20, regime, daily_ret]
    )

    return features, target, regime, feature_names


def main():
    dates, prices = load_data()
    features, target, regime, feature_names = compute_features(prices)
    n = len(prices)

    # Find first valid index (where all features are non-NaN)
    start_idx = 200  # Need at least 200 days for EMA(200) to stabilize

    # Walk-forward: train on 4 years (~1008 trading days), predict 1 year (~252 days)
    train_days = 1008
    test_days = 252

    initial_capital = 10_000.0
    cash = initial_capital
    shares = 0.0
    in_market = False
    wins = 0
    losses = 0
    entry_price = 0.0
    peak_value = initial_capital
    max_drawdown = 0.0
    total_predictions = 0
    correct_predictions = 0
    trades = []
    first_trade_idx = None

    print("Running walk-forward XGBoost backtest...")
    print(f"  Features: {', '.join(feature_names)}")
    print(f"  Train window: {train_days} days, Test window: {test_days} days")
    print()

    walk_start = start_idx
    fold = 0

    while walk_start + train_days + test_days <= n:
        train_start = walk_start
        train_end = walk_start + train_days
        test_start = train_end
        test_end = min(train_end + test_days, n - 1)

        # Get training data (skip rows with NaN)
        X_train = features[train_start:train_end]
        y_train = target[train_start:train_end]

        # Remove NaN rows
        valid_train = ~np.any(np.isnan(X_train), axis=1)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        if len(X_train) < 100:
            walk_start += test_days
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0,
            use_label_encoder=False,
        )
        model.fit(X_train_scaled, y_train)

        fold += 1
        fold_trades = 0

        # Test period
        for i in range(test_start, test_end):
            if np.any(np.isnan(features[i])):
                continue

            X_test = scaler.transform(features[i : i + 1])
            pred = model.predict(X_test)[0]
            actual = target[i]

            total_predictions += 1
            if pred == actual:
                correct_predictions += 1

            is_bullish = regime[i] is not None and not np.isnan(regime[i]) and regime[i] > 0

            portfolio_value = cash + shares * prices[i]
            peak_value = max(peak_value, portfolio_value)
            dd = (peak_value - portfolio_value) / peak_value * 100
            max_drawdown = max(max_drawdown, dd)

            if not in_market:
                # Buy when model predicts up AND regime is bullish
                if pred == 1 and is_bullish:
                    shares = cash / prices[i]
                    entry_price = prices[i]
                    cash = 0.0
                    in_market = True
                    fold_trades += 1
                    trades.append(("BUY", dates[i], prices[i]))
                    if first_trade_idx is None:
                        first_trade_idx = i
            else:
                # Sell when model predicts down OR regime turns bearish
                if pred == 0 or not is_bullish:
                    cash = shares * prices[i]
                    shares = 0.0
                    in_market = False
                    fold_trades += 1
                    trades.append(("SELL", dates[i], prices[i]))
                    if prices[i] > entry_price:
                        wins += 1
                    else:
                        losses += 1

        print(
            f"  Fold {fold}: train {dates[train_start]}..{dates[train_end-1]}, "
            f"test {dates[test_start]}..{dates[min(test_end, n-1)-1]}, "
            f"trades: {fold_trades}"
        )

        walk_start += test_days

    # Final portfolio value
    last_price = prices[-1]
    last_date = dates[-1]
    portfolio_value = cash + shares * last_price
    strategy_roi = (portfolio_value - initial_capital) / initial_capital * 100

    peak_value = max(peak_value, portfolio_value)
    dd = (peak_value - portfolio_value) / peak_value * 100
    max_drawdown = max(max_drawdown, dd)

    # Buy-and-hold
    if first_trade_idx is not None:
        bh_price = prices[first_trade_idx]
        bh_date = dates[first_trade_idx]
    else:
        bh_price = prices[start_idx]
        bh_date = dates[start_idx]

    bh_shares = initial_capital / bh_price
    bh_value = bh_shares * last_price
    bh_roi = (bh_value - initial_capital) / initial_capital * 100

    total_trades = wins + losses
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0

    print()
    print("=" * 55)
    print("   ML Regime-Adaptive (XGBoost) Backtest Results")
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
    winner_label = "Strategy" if diff > 0 else "Buy & Hold"
    print(f"  Difference:             {diff:+,.2f}% ({winner_label} wins)")
    print("-" * 55)
    print(f"  Walk-forward folds:     {fold}")
    print(f"  Total round-trip trades: {total_trades}")
    print(f"  Win rate:               {win_rate:.1f}%")
    print(f"  Prediction accuracy:    {accuracy:.1f}%")
    print(f"  Max drawdown:           {max_drawdown:.2f}%")
    status = "Invested" if in_market else "Cash"
    print(f"  Final position:         {status}")
    print("=" * 55)
    print()
    print("  Trade log (last 20):")
    for action, date, price in trades[-20:]:
        print(f"    {date}  {action:4s}  @ ${price:,.2f}")


if __name__ == "__main__":
    main()
