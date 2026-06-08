import numpy as np
import pandas as pd
import yfinance as yf
def download_prices(ticker, start="2015-01-01"):
    print(f"Downloading data for: {ticker}")

    raw = yf.download(ticker, start=start, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data found for '{ticker}'. Check the ticker symbol.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Close"])
    df = df.sort_index()

    print(f"Got {len(df)} trading days  ({df.index[0].date()} to {df.index[-1].date()})")
    return df

def add_log_returns(df):
    df = df.copy()
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna(subset=["LogReturn"])
    return df

def get_data(ticker, start="2015-01-01"):
    df = download_prices(ticker, start=start)
    df = add_log_returns(df)
    print(f"Ready. Shape: {df.shape}\n")
    return df

if __name__ == "__main__":
    df = get_data("^NSEI")
    print(df.tail())
