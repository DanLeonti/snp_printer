import yfinance as yf
import pandas as pd

df = yf.download("SPY", start="1996-03-08", end="2026-03-08")
out = pd.DataFrame({
    "date": df.index.strftime("%Y-%m-%d"),
    "ticker": "SPY",
    "price": df["Close"].round(2).values.flatten(),
})
out.to_csv("/root/snp_printer/spy_daily_close.csv", index=False)
print(f"Saved {len(out)} rows")
print(out.head())
print("...")
print(out.tail())
