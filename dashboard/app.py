import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingest import get_data
from src.features import build_features
from src.hmm_model import fit_hmm, get_state_probs, label_states
from src.ml_model import build_feature_matrix, walk_forward_validation, evaluate

st.set_page_config(
    page_title="Market Regime Engine",
    page_icon="📈",
    layout="wide"
)
st.title("📈 General Market Regime & Direction Forecasting Engine")
st.caption("Detects market regimes and forecasts next-day direction for any ticker")

with st.sidebar:
    st.header("⚙️ Settings")
    
    ticker = st.text_input(
        "Ticker symbol",
        value="^NSEI",
        help="Any Yahoo Finance ticker: ^NSEI, ^GSPC, ^NSEBANK, GC=F"
    )
    
    start_date = st.selectbox(
        "History",
        options=["2015-01-01", "2018-01-01", "2020-01-01"],
        index=0
    )
    
    n_states = st.selectbox(
        "Number of regimes",
        options=[2, 3, 4],
        index=1
    )
    
    run_button = st.button("🚀 Run Analysis", type="primary")
    
    st.divider()
    st.markdown("**Example tickers:**")
    st.markdown("🇮🇳 `^NSEI` — Nifty 50")
    st.markdown("🇮🇳 `^NSEBANK` — Bank Nifty")
    st.markdown("🇺🇸 `^GSPC` — S&P 500")
    st.markdown("🥇 `GC=F` — Gold")
    st.markdown("🛢️ `CL=F` — Crude Oil")
if run_button:
    with st.spinner(f"Downloading data for {ticker}..."):
        try:
            df = get_data(ticker, start=start_date)
            df = build_features(df)
        except Exception as e:
            st.error(f"Could not load data for '{ticker}'. Check the ticker symbol.")
            st.stop()

    st.success(f"Loaded {len(df)} trading days for {ticker}")

    tab1, tab2, tab3 = st.tabs(["📊 Regimes", "🤖 ML Prediction", "📋 Scorecard"])

    with tab1:
        st.subheader("Hidden Markov Model — Detected Regimes")

        with st.spinner("Fitting HMM..."):
            hmm_model = fit_hmm(df, n_states=n_states)
            probs_df = get_state_probs(hmm_model, df)
            labels = label_states(hmm_model)

        col1, col2, col3 = st.columns(3)

        regime_counts = probs_df["HMMState"].map(labels).value_counts()
        total = len(probs_df)

        with col1:
            bull_pct = regime_counts.get("Bull", 0) / total * 100
            st.metric("🟢 Bull days", f"{bull_pct:.1f}%")
        with col2:
            side_pct = regime_counts.get("Sideways", 0) / total * 100
            st.metric("⬜ Sideways days", f"{side_pct:.1f}%")
        with col3:
            bear_pct = regime_counts.get("Bear", 0) / total * 100
            st.metric("🔴 Bear days", f"{bear_pct:.1f}%")

        fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                         gridspec_kw={"height_ratios": [3, 1]})

        color_map = {"Bull": "#c8f5c8", "Sideways": "#e8e8e8", "Bear": "#f5c8c8"}

        regime_start = df.index[0]
        current_label = labels[probs_df["HMMState"].iloc[0]]

        for i in range(1, len(probs_df)):
            label = labels[probs_df["HMMState"].iloc[i]]
            if label != current_label or i == len(probs_df) - 1:
                ax1.axvspan(regime_start, df.index[i],
                           alpha=0.35, color=color_map[current_label])
                regime_start = df.index[i]
                current_label = label

        ax1.plot(df.index, df["Close"], color="black", linewidth=0.8)
        patches = [mpatches.Patch(color=c, alpha=0.4, label=l)
                   for l, c in color_map.items()]
        ax1.legend(handles=patches, loc="upper left")
        ax1.set_title(f"{ticker} — Detected Regimes", fontweight="bold")
        ax1.set_ylabel("Price")
        ax1.grid(alpha=0.3)

        bull_state = [k for k, v in labels.items() if v == "Bull"]
        bear_state = [k for k, v in labels.items() if v == "Bear"]

        if bull_state:
            ax2.fill_between(probs_df.index,
                             probs_df[f"State{bull_state[0]}"],
                             alpha=0.6, color="#2E6FA7", label="P(Bull)")
        if bear_state:
            ax2.fill_between(probs_df.index,
                             probs_df[f"State{bear_state[0]}"],
                             alpha=0.6, color="#c0392b", label="P(Bear)")

        ax2.set_ylabel("Probability")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper left")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

        today_state = probs_df["HMMState"].iloc[-1]
        today_label = labels[today_state]
        today_prob = probs_df[f"State{today_state}"].iloc[-1]

        color = {"Bull": "🟢", "Sideways": "⬜", "Bear": "🔴"}
        st.info(f"**Current regime: {color[today_label]} {today_label}** "
                f"(confidence: {today_prob:.1%})")
    with tab2:
        st.subheader("ML Model — Next Day Direction Forecast")

        with st.spinner("Running walk-forward validation (this takes 1-2 mins)..."):
            features = build_feature_matrix(df, probs_df)
            
            if len(features) < 600:
                st.warning("Not enough data for walk-forward validation. Use an earlier start date.")
                st.stop()
            
            results = walk_forward_validation(features)

        last_features = features.drop(columns=["Target"]).iloc[-1:]
        
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        
        X = features.drop(columns=["Target"])
        y = features["Target"]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        final_model = LogisticRegression(random_state=42, max_iter=1000)
        final_model.fit(X_scaled, y)
        
        last_scaled = scaler.transform(last_features)
        prob_up = final_model.predict_proba(last_scaled)[0][1]
        direction = "UP ↑" if prob_up > 0.5 else "DOWN ↓"
        confidence = max(prob_up, 1 - prob_up)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Tomorrow's direction", direction)
        with col2:
            st.metric("Confidence", f"{confidence:.1%}")
        with col3:
            st.metric("P(Up)", f"{prob_up:.1%}")

        if prob_up > 0.5:
            st.success(f"Model predicts **{direction}** with {confidence:.1%} confidence")
        else:
            st.error(f"Model predicts **{direction}** with {confidence:.1%} confidence")

        st.divider()
        st.subheader("Rolling P(Up) over time")
        
        fig2, ax = plt.subplots(figsize=(12, 3))
        ax.plot(results.index,
                results["Probability"].rolling(20).mean(),
                color="#2E6FA7", linewidth=1)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
        ax.fill_between(results.index,
                        results["Probability"].rolling(20).mean(),
                        0.5, alpha=0.2, color="#2E6FA7")
        ax.set_ylabel("P(Up tomorrow)")
        ax.set_ylim(0.44, 0.58)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        st.caption("⚠️ This is a research model, not financial advice.")
    with tab3:
        st.subheader("Model Validation Scorecard")

        accuracy = accuracy_score(results["Actual"], results["Predicted"])
        baseline = results["Actual"].mean()
        edge = accuracy - baseline

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model accuracy", f"{accuracy:.1%}")
        with col2:
            st.metric("Baseline (always up)", f"{baseline:.1%}")
        with col3:
            delta_color = "normal" if edge >= 0 else "inverse"
            st.metric("Edge over baseline", f"{edge:.1%}",
                     delta=f"{edge:.1%}", delta_color=delta_color)

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cumulative accuracy")
            fig3, ax = plt.subplots(figsize=(6, 3))
            correct = (results["Actual"] == results["Predicted"])
            ax.plot(results.index, correct.expanding().mean(),
                   color="#1F7A4D", linewidth=1, label="Model")
            ax.axhline(baseline, color="red", linestyle="--",
                      linewidth=0.8, label=f"Baseline {baseline:.1%}")
            ax.set_ylabel("Accuracy")
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig3)
            plt.close()

        with col2:
            st.subheader("Prediction breakdown")
            from sklearn.metrics import classification_report
            report = classification_report(
                results["Actual"],
                results["Predicted"],
                target_names=["Down", "Up"],
                output_dict=True
            )
            report_df = pd.DataFrame(report).transpose().round(2)
            st.dataframe(report_df)

        st.divider()
        st.subheader("📝 Honest interpretation")

        if edge >= 0.02:
            msg = f"""The model beats the baseline by **{edge:.1%}**. 
            This is a meaningful edge for daily index prediction."""
        elif edge >= 0:
            msg = f"""The model matches the baseline closely (+{edge:.1%}). 
            The signal is weak but the methodology is sound."""
        else:
            msg = f"""The model is below baseline by **{abs(edge):.1%}**. 
            This is consistent with efficient markets — daily index direction 
            contains very little predictable signal. The value is in the 
            methodology, not in claiming alpha that doesn't exist."""

        st.info(msg)

        st.caption(f"Validated on {len(results)} trading days using "
                  f"walk-forward validation — trained on past, tested on future, "
                  f"never shuffled.")
