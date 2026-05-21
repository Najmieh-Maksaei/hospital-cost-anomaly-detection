# hospital-cost-anomaly-detection
# Hospital Cost Anomaly Detection

This project presents a simple and reproducible Python pipeline for detecting anomalies in synthetic daily hospital cost data.

The project is inspired by real-world healthcare cost monitoring problems, where unexpected increases in daily costs may indicate unusual activity, operational changes, billing issues, or structural changes in the system.

## Objective

The goal is to simulate daily hospital cost data and identify abnormal days using rolling statistical features and clustering-based anomaly detection.

## Methodology

The pipeline includes:

- Synthetic data generation for daily hospital costs
- Weekly seasonality and trend simulation
- Artificial injection of abnormal high-cost days
- Rolling mean and rolling standard deviation features
- KMeans clustering for anomaly detection
- Visualization of detected anomalies

## Tools

- Python
- NumPy
- pandas
- scikit-learn
- matplotlib

## How to Run

Install the required packages:

```bash
pip install -r requirements.txt