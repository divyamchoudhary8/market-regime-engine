import numpy as np
import pandas as pd
import yfinance as yf

def add_realized_vol(df, window=20):
    df = df.copy()
    df["RealizedVol"] = df["LogReturn"].rolling(window).std() * np.sqrt(252)
    return df

def add_volume_ratio(df, window=20):
    df = df.copy()
    df["VolumeRatio"] = df["Volume"] / df["Volume"].rolling(window).mean()
    return df

def add_momentum(df, window=10):
    df = df.copy()
    df["Momentum"] = df["LogReturn"].rolling(window).sum()
    return df

def add_sp500_return(df):
    df = df.copy()
    
    sp500 = yf.download("^GSPC", start=df.index[0], 
                         end=df.index[-1], 
                         auto_adjust=True, 
                         progress=False)
    
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
    
    sp500_returns = np.log(sp500["Close"] / sp500["Close"].shift(1))
    sp500_returns.name = "SP500Return"
    
    df = df.join(sp500_returns, how="left")
    df["SP500Return"] = df["SP500Return"].shift(1)
    df["SP500Return"] = df["SP500Return"].fillna(0)

    return df

def add_day_of_week(df):
    df = df.copy()
    df["DayOfWeek"] = df.index.dayofweek
    return df

def add_india_vix(df):
    df = df.copy()
    
    vix = yf.download("^INDIAVIX", start=df.index[0],
                       end=df.index[-1],
                       auto_adjust=True,
                       progress=False)
    
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    
    if vix.empty:
        df["IndiaVIX"] = 0
        print("India VIX not available, filling with 0")
        return df
    
    vix_close = vix["Close"]
    vix_close.name = "IndiaVIX"
    
    df = df.join(vix_close, how="left")
    df["IndiaVIX"] = df["IndiaVIX"].ffill()
    df["IndiaVIX"] = df["IndiaVIX"].fillna(0)
    
    return df

def build_features(df):
    df = add_realized_vol(df)
    df = add_volume_ratio(df)
    df = add_momentum(df)
    df = add_sp500_return(df)
    df = add_day_of_week(df)
    df = add_india_vix(df)
    df = df.dropna()
    return df

if __name__ == "__main__":
    from src.ingest import get_data
    df = get_data("^NSEI")
    df = build_features(df)
    print(df[["LogReturn", "RealizedVol", "VolumeRatio", "Momentum"]].tail())
    print(f"\nShape: {df.shape}")