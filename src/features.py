import numpy as np
import pandas as pd
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
def build_features(df):
    df = add_realized_vol(df)
    df = add_volume_ratio(df)
    df = add_momentum(df)
    df = df.dropna()
    return df
if __name__ == "__main__":
    from src.ingest import get_data
    df = get_data("^NSEI")
    df = build_features(df)
    print(df[["LogReturn", "RealizedVol", "VolumeRatio", "Momentum"]].tail())
    print(f"\nShape: {df.shape}")