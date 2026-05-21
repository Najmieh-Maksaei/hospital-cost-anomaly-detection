import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def simulate_hospital_costs(n_days=365, random_state=42):
    """
    Simulate daily hospital cost data with:
    - baseline cost level
    - weekly seasonality
    - mild increasing trend
    - random noise
    - injected abnormal high-cost days
    """
    rng = np.random.default_rng(random_state)

    dates = pd.date_range(start="2025-01-01", periods=n_days, freq="D")

    baseline = 10000
    weekly_pattern = 800 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    trend = 5 * np.arange(n_days)
    noise = rng.normal(0, 700, n_days)

    costs = baseline + weekly_pattern + trend + noise

    anomaly_days = [80, 120, 121, 122, 250, 300]
    costs[anomaly_days] += rng.normal(5000, 800, len(anomaly_days))

    data = pd.DataFrame({
        "date": dates,
        "daily_cost": costs
    })

    data["true_anomaly"] = 0
    data.loc[anomaly_days, "true_anomaly"] = 1

    return data


def add_features(data, window=7):
    """
    Add rolling mean and rolling standard deviation features.
    """
    df = data.copy()

    df["rolling_mean_7"] = df["daily_cost"].rolling(window=window).mean()
    df["rolling_std_7"] = df["daily_cost"].rolling(window=window).std()

    df = df.dropna().reset_index(drop=True)

    return df


def detect_anomalies_clustering(data, n_clusters=2):
    """
    Detect anomalies using KMeans clustering.
    The cluster with the highest average daily cost is labeled as anomalous.
    """
    df = data.copy()

    features = df[["daily_cost", "rolling_mean_7", "rolling_std_7"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(X_scaled)

    cluster_means = df.groupby("cluster")["daily_cost"].mean()
    anomaly_cluster = cluster_means.idxmax()

    df["predicted_anomaly"] = (df["cluster"] == anomaly_cluster).astype(int)

    return df


def plot_anomalies(data, output_path):
    """
    Plot daily hospital costs and highlight detected anomalies.
    """
    plt.figure(figsize=(12, 6))

    plt.plot(data["date"], data["daily_cost"], label="Daily cost")

    anomalies = data[data["predicted_anomaly"] == 1]
    plt.scatter(
        anomalies["date"],
        anomalies["daily_cost"],
        label="Detected anomaly"
    )

    plt.xlabel("Date")
    plt.ylabel("Daily cost")
    plt.title("Detected Anomalies in Daily Hospital Costs")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path)
    plt.close()


def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    data = simulate_hospital_costs()
    data.to_csv("data/synthetic_hospital_costs.csv", index=False)

    data_features = add_features(data)
    results = detect_anomalies_clustering(data_features)

    results.to_csv("results/anomaly_detection_results.csv", index=False)

    plot_anomalies(
        results,
        output_path="results/figures/detected_anomalies.png"
    )

    print("Project completed successfully.")
    print("Results saved in the results/ folder.")


if __name__ == "__main__":
    main()
