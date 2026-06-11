import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report


def walk_forward_v2(features, min_train_size=500):
    X = features.drop(columns=["Target"])
    y = features["Target"]

    lr_preds,  lr_probs  = [], []
    lgb_preds, lgb_probs = [], []
    actuals, dates = [], []

    total = len(features) - min_train_size
    for i in range(min_train_size, len(features)):
        if (i - min_train_size) % 100 == 0:
            print(f"  Progress: {i - min_train_size}/{total} days")

        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test  = X.iloc[i:i+1]
        y_test  = y.iloc[i:i+1]

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # Logistic Regression
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X_train_sc, y_train)
        lr_preds.append(lr.predict(X_test_sc)[0])
        lr_probs.append(lr.predict_proba(X_test_sc)[0][1])

        # LightGBM
        lgb = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        lgb.fit(X_train_sc, y_train)
        lgb_preds.append(lgb.predict(X_test_sc)[0])
        lgb_probs.append(lgb.predict_proba(X_test_sc)[0][1])

        actuals.append(y_test.iloc[0])
        dates.append(X_test.index[0])

    results = pd.DataFrame({
        "Actual":   actuals,
        "LR_pred":  lr_preds,
        "LR_prob":  lr_probs,
        "LGB_pred": lgb_preds,
        "LGB_prob": lgb_probs,
    }, index=dates)

    return results


def compare_models(results, ticker):
    lr_acc  = accuracy_score(results["Actual"], results["LR_pred"])
    lgb_acc = accuracy_score(results["Actual"], results["LGB_pred"])
    baseline = results["Actual"].mean()

    print(f"\n{'='*45}")
    print(f"Model Comparison — {ticker}")
    print(f"{'='*45}")
    print(f"Baseline (always Up):   {baseline:.1%}")
    print(f"Logistic Regression:    {lr_acc:.1%}  "
          f"(edge: {lr_acc - baseline:+.1%})")
    print(f"LightGBM:               {lgb_acc:.1%}  "
          f"(edge: {lgb_acc - baseline:+.1%})")
    print(f"LightGBM vs LR:         {lgb_acc - lr_acc:+.1%}")

    print(f"\nLogistic Regression report:")
    print(classification_report(results["Actual"], results["LR_pred"],
                                 target_names=["Down", "Up"]))

    print(f"LightGBM report:")
    print(classification_report(results["Actual"], results["LGB_pred"],
                                 target_names=["Down", "Up"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    correct_lr  = (results["Actual"] == results["LR_pred"])
    correct_lgb = (results["Actual"] == results["LGB_pred"])

    axes[0].plot(results.index, correct_lr.expanding().mean(),
                 color="#2E6FA7", linewidth=1,
                 label=f"Logistic Regression ({lr_acc:.1%})")
    axes[0].plot(results.index, correct_lgb.expanding().mean(),
                 color="#E67E22", linewidth=1,
                 label=f"LightGBM ({lgb_acc:.1%})")
    axes[0].axhline(baseline, color="red", linestyle="--",
                    linewidth=0.8, label=f"Baseline {baseline:.1%}")
    axes[0].set_title("Cumulative accuracy — LR vs LightGBM")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(results.index,
                 results["LR_prob"].rolling(20).mean(),
                 color="#2E6FA7", linewidth=1, label="LR P(Up)")
    axes[1].plot(results.index,
                 results["LGB_prob"].rolling(20).mean(),
                 color="#E67E22", linewidth=1, label="LightGBM P(Up)")
    axes[1].axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Rolling 20-day P(Up) forecast")
    axes[1].set_ylabel("Probability")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return lr_acc, lgb_acc


def explain_with_shap(features):
    print("\nComputing SHAP values on full dataset...")

    X = features.drop(columns=["Target"])
    y = features["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    model = HistGradientBoostingClassifier(
        max_iter=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_scaled_df, y)

    sample = shap.sample(X_scaled_df, 100, random_state=42)
    explainer = shap.Explainer(model.predict_proba, sample)
    shap_values = explainer(X_scaled_df[:500])
    sv = shap_values.values[:, :, 1]
    X_scaled_df = X_scaled_df[:500]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    plt.sca(axes[0])
    shap.summary_plot(
        sv, X_scaled_df,
        plot_type="bar",
        show=False,
        max_display=15
    )
    axes[0].set_title("Feature Importance (SHAP)", fontweight="bold")

    plt.sca(axes[1])
    shap.summary_plot(
        sv, X_scaled_df,
        show=False,
        max_display=15
    )
    axes[1].set_title("SHAP Impact Direction", fontweight="bold")

    plt.tight_layout()
    plt.show()

    feature_importance = pd.DataFrame({
        "Feature":   X.columns,
        "Mean_SHAP": np.abs(sv).mean(axis=0)
    }).sort_values("Mean_SHAP", ascending=False)

    print("\nTop 10 most important features:")
    print(feature_importance.head(10).to_string(index=False))

    return feature_importance


def run_ml_v2(ticker, start="2015-01-01"):
    from src.ingest import get_data
    from src.features_v2 import build_features_v2
    from src.hmm_model import fit_hmm, get_state_probs
    from src.ml_model import build_feature_matrix

    df = get_data(ticker, start=start)
    df = build_features_v2(df)

    hmm_model = fit_hmm(df)
    probs_df  = get_state_probs(hmm_model, df)

    features = build_feature_matrix(df, probs_df)

    print(f"\nFeatures: {features.shape[1]-1} columns")
    print(f"Running walk-forward validation "
          f"({len(features)-500} predictions)...")

    results = walk_forward_v2(features)
    lr_acc, lgb_acc = compare_models(results, ticker)
    importance = explain_with_shap(features)

    return results, lr_acc, lgb_acc, importance


if __name__ == "__main__":
    results, lr_acc, lgb_acc, importance = run_ml_v2("^NSEI")