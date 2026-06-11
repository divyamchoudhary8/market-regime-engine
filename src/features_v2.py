import numpy as np
import pandas as pd
import yfinance as yf
from src.features import (
    add_realized_vol,
    add_volume_ratio,
    add_momentum,
    add_sp500_return,
    add_day_of_week,
    add_india_vix
)
def add_rsi(df, window=14):
    df = df.copy()
    
    delta = df["Close"].diff()
    
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    
    return df
def add_macd(df):
    df = df.copy()
    
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    
    return df
def add_atr(df, window=14):
    df = df.copy()
    
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    df["ATR"] = true_range.rolling(window).mean()
    
    return df
def add_atr(df, window=14):
    df = df.copy()
    
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    df["ATR"] = true_range.rolling(window).mean()
    
    return df
def add_ma_distance(df, windows=[20, 50]):
    df = df.copy()
    
    for window in windows:
        ma = df["Close"].rolling(window).mean()
        df[f"MA{window}_dist"] = (df["Close"] - ma) / ma
        
    return df
def add_macro_features(df):
    df = df.copy()
    
    macro_tickers = {
        "USDINR=X": "USDINR",
        "BZ=F":     "Brent",
        "GC=F":     "Gold"
    }
    
    for ticker, name in macro_tickers.items():
        try:
            raw = yf.download(
                ticker,
                start=df.index[0],
                end=df.index[-1],
                auto_adjust=True,
                progress=False
            )
            
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            
            if raw.empty:
                print(f"{name} not available, filling with 0")
                df[f"{name}_return"] = 0
                continue
            
            returns = np.log(raw["Close"] / raw["Close"].shift(1))
            returns.name = f"{name}_return"
            
            df = df.join(returns, how="left")
            df[f"{name}_return"] = df[f"{name}_return"].shift(1)
            df[f"{name}_return"] = df[f"{name}_return"].ffill().fillna(0)
            
        except Exception as e:
            print(f"Could not download {name}: {e}")
            df[f"{name}_return"] = 0
    
    return df
def build_features_v2(df):
    df = add_realized_vol(df)
    df = add_volume_ratio(df)
    df = add_momentum(df)
    df = add_sp500_return(df)
    df = add_day_of_week(df)
    df = add_india_vix(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_atr(df)
    df = add_ma_distance(df)
    df = add_macro_features(df)
    df = df.dropna()
    return df


if __name__ == "__main__":
    from src.ingest import get_data
    df = get_data("^NSEI")
    df = build_features_v2(df)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(df.tail(3))

