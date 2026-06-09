import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from hmmlearn import hmm
def fit_hmm(df, n_states=3):
    returns = df["LogReturn"].values.reshape(-1, 1)

    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        n_iter=1000,
        random_state=42
    )

    model.fit(returns)

    print(f"Model fitted. Converged: {model.monitor_.converged}")
    print(f"\nLearned regime means (daily return):")
    for i, mean in enumerate(model.means_):
        print(f"  State {i}: mean = {mean[0]:.5f}, "
              f"vol = {np.sqrt(model.covars_[i][0][0]):.5f}")

    return model
def get_state_probs(model, df):
    returns = df["LogReturn"].values.reshape(-1, 1)
    
    state_probs = model.predict_proba(returns)
    hidden_states = model.predict(returns)
    
    probs_df = pd.DataFrame(
        state_probs,
        columns=[f"State{i}" for i in range(state_probs.shape[1])],
        index=df.index
    )
    
    probs_df["HMMState"] = hidden_states
    
    return probs_df
def label_states(model):
    means = model.means_.flatten()
    order = np.argsort(means)
    
    labels = {}
    labels[order[0]] = "Bear"
    labels[order[1]] = "Sideways"
    labels[order[2]] = "Bull"
    
    return labels
def plot_hmm_regimes(df, probs_df, labels, ticker):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]})

    color_map = {"Bull": "#c8f5c8", "Sideways": "#e8e8e8", "Bear": "#f5c8c8"}
    line_map  = {"Bull": "#2E6FA7", "Sideways": "#888888", "Bear": "#c0392b"}

    regime_start = df.index[0]
    current_state = probs_df["HMMState"].iloc[0]
    current_label = labels[current_state]

    for i in range(1, len(probs_df)):
        state = probs_df["HMMState"].iloc[i]
        label = labels[state]
        if label != current_label or i == len(probs_df) - 1:
            ax1.axvspan(regime_start, df.index[i],
                       alpha=0.35, color=color_map[current_label])
            regime_start = df.index[i]
            current_label = label
            current_state = state

    ax1.plot(df.index, df["Close"], color="black", linewidth=0.8)
    bull_patch = mpatches.Patch(color="#c8f5c8", alpha=0.4, label="Bull")
    side_patch = mpatches.Patch(color="#e8e8e8", alpha=0.4, label="Sideways")
    bear_patch = mpatches.Patch(color="#f5c8c8", alpha=0.4, label="Bear")
    ax1.legend(handles=[bull_patch, side_patch, bear_patch], loc="upper left")
    ax1.set_title(f"{ticker} — Detected Regimes (HMM)", fontweight="bold")
    ax1.set_ylabel("Price")
    ax1.grid(alpha=0.3)

    bull_state  = [k for k, v in labels.items() if v == "Bull"][0]
    bear_state  = [k for k, v in labels.items() if v == "Bear"][0]

    ax2.fill_between(probs_df.index,
                     probs_df[f"State{bull_state}"],
                     alpha=0.6, color="#2E6FA7", label="P(Bull)")
    ax2.fill_between(probs_df.index,
                     probs_df[f"State{bear_state}"],
                     alpha=0.6, color="#c0392b", label="P(Bear)")
    ax2.set_ylabel("Probability")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper left")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()
def run_hmm(ticker, start="2015-01-01"):
    from src.ingest import get_data
    from src.features import build_features

    df = get_data(ticker, start=start)
    df = build_features(df)

    model = fit_hmm(df)
    probs_df = get_state_probs(model, df)
    labels = label_states(model)

    print(f"\nState labels: {labels}")
    print(f"\nRegime counts:")
    counts = probs_df["HMMState"].map(labels).value_counts()
    print(counts)

    plot_hmm_regimes(df, probs_df, labels, ticker)

    return model, probs_df, labels


if __name__ == "__main__":
    model, probs_df, labels = run_hmm("^NSEI")
