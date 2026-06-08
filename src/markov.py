import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
def label_regimes(df, bull_threshold=0.02, bear_threshold=-0.02):
    df = df.copy()

    conditions = [
        df["Momentum"] > bull_threshold,
        df["Momentum"] < bear_threshold,
    ]
    choices = ["Bull", "Bear"]

    df["Regime"] = np.select(conditions, choices, default="Sideways")

    return df
def build_transition_matrix(df):
    regimes = ["Bull", "Sideways", "Bear"]
    
    matrix = pd.DataFrame(0.0, index=regimes, columns=regimes)
    
    for i in range(len(df) - 1):
        today = df["Regime"].iloc[i]
        tomorrow = df["Regime"].iloc[i + 1]
        matrix.loc[today, tomorrow] += 1
    
    matrix = matrix.div(matrix.sum(axis=1), axis=0)
    
    return matrix
def plot_regimes(df, ticker):
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(df.index, df["Close"], color="black", linewidth=0.8)

    colors = {"Bull": "#c8f5c8", "Sideways": "#e8e8e8", "Bear": "#f5c8c8"}

    regime_start = df.index[0]
    current_regime = df["Regime"].iloc[0]

    for i in range(1, len(df)):
        if df["Regime"].iloc[i] != current_regime or i == len(df) - 1:
            ax.axvspan(regime_start, df.index[i],
                      alpha=0.4, color=colors[current_regime])
            regime_start = df.index[i]
            current_regime = df["Regime"].iloc[i]

    bull_patch = mpatches.Patch(color="#c8f5c8", alpha=0.4, label="Bull")
    side_patch = mpatches.Patch(color="#e8e8e8", alpha=0.4, label="Sideways")
    bear_patch = mpatches.Patch(color="#f5c8c8", alpha=0.4, label="Bear")

    ax.legend(handles=[bull_patch, side_patch, bear_patch], loc="upper left")
    ax.set_title(f"{ticker} — Detected Regimes (Markov Chain)", fontweight="bold")
    ax.set_ylabel("Price")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def run_markov(ticker, start="2015-01-01"):
    from src.ingest import get_data
    from src.features import build_features

    df = get_data(ticker, start=start)
    df = build_features(df)
    df = label_regimes(df)

    matrix = build_transition_matrix(df)

    print("\nTransition Matrix:")
    print(matrix.round(3))

    print(f"\nRegime counts:")
    print(df["Regime"].value_counts())

    plot_regimes(df, ticker)

    return df, matrix


if __name__ == "__main__":
    df, matrix = run_markov("^NSEI")
