import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
def build_feature_matrix(df, probs_df):
    features = pd.DataFrame(index=df.index)
    
    features["LogReturn"] = df["LogReturn"]
    features["RealizedVol"] = df["RealizedVol"]
    features["VolumeRatio"] = df["VolumeRatio"]
    features["Momentum"] = df["Momentum"]
    
    for col in probs_df.columns:
        if col.startswith("State"):
            features[col] = probs_df[col]
    
    features["Target"] = (df["LogReturn"].shift(-1) > 0).astype(int)
    
    features = features.dropna()
    
    return features
def walk_forward_validation(features, min_train_size=500):
    X = features.drop(columns=["Target"])
    y = features["Target"]
    
    predictions = []
    probabilities = []
    actuals = []
    dates = []
    
    for i in range(min_train_size, len(features)):
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test = X.iloc[i:i+1]
        y_test = y.iloc[i:i+1]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        
        pred = model.predict(X_test_scaled)[0]
        prob = model.predict_proba(X_test_scaled)[0][1]
        
        predictions.append(pred)
        probabilities.append(prob)
        actuals.append(y_test.iloc[0])
        dates.append(X_test.index[0])
    
    results = pd.DataFrame({
        "Actual": actuals,
        "Predicted": predictions,
        "Probability": probabilities
    }, index=dates)
    
    return results
def evaluate(results, ticker):
    accuracy = accuracy_score(results["Actual"], results["Predicted"])
    baseline = results["Actual"].mean()
    
    print(f"\n{'='*40}")
    print(f"Results for {ticker}")
    print(f"{'='*40}")
    print(f"Model accuracy:    {accuracy:.1%}")
    print(f"Baseline (always predict up): {baseline:.1%}")
    print(f"Edge over baseline: {accuracy - baseline:.1%}")
    print(f"\n{classification_report(results['Actual'], results['Predicted'], target_names=['Down','Up'])}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    axes[0].plot(results.index, results["Probability"].rolling(20).mean(),
                 color="#2E6FA7", linewidth=1)
    axes[0].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Rolling 20-day P(Up) forecast")
    axes[0].set_ylabel("Probability")
    axes[0].grid(alpha=0.3)
    
    correct = (results["Actual"] == results["Predicted"])
    cumulative_accuracy = correct.expanding().mean()
    axes[1].plot(results.index, cumulative_accuracy,
                 color="#1F7A4D", linewidth=1)
    axes[1].axhline(baseline, color="red", linestyle="--",
                    linewidth=0.8, label=f"Baseline {baseline:.1%}")
    axes[1].set_title("Cumulative accuracy over time")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return accuracy
def run_ml(ticker, start="2015-01-01"):
    from src.ingest import get_data
    from src.features import build_features
    from src.hmm_model import fit_hmm, get_state_probs

    df = get_data(ticker, start=start)
    df = build_features(df)
    
    model_hmm = fit_hmm(df)
    probs_df = get_state_probs(model_hmm, df)
    
    features = build_feature_matrix(df, probs_df)
    
    print(f"\nRunning walk-forward validation...")
    print(f"Training on first 500 days, predicting the rest ({len(features)-500} days)")
    
    results = walk_forward_validation(features)
    accuracy = evaluate(results, ticker)
    
    return results, accuracy


if __name__ == "__main__":
    results, accuracy = run_ml("^NSEI")
