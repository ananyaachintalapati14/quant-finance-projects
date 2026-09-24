import pandas as pd
import yfinance as yf
import os

os.makedirs("data", exist_ok=True)
TICKERS = ["AAPL", "MSFT", "NVDA", "SPY", "QQQ"]
frames = []

for sym in TICKERS:
    try:
        tk = yf.Ticker(sym)
        exps = tk.options
        if exps:
            ch = tk.option_chain(exps[0])
            df = pd.concat([ch.calls.assign(type="call"), ch.puts.assign(type="put")])
            df["symbol"] = sym
            frames.append(df)
            print(f"Fetched {sym}")
    except Exception as e:
        print(f"{sym} failed: {e}")

if frames:
    pd.concat(frames, ignore_index=True).to_parquet("data/snapshot.parquet", index=False)
    print("Saved data/snapshot.parquet")