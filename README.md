# hospital-cost-anomaly-detection

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
```

Run the script:

```bash
python main.py
```

## Outputs

The script creates:

```text
data/synthetic_hospital_costs.csv
results/anomaly_detection_results.csv
results/figures/detected_anomalies.png
```

## Project Context

This project uses synthetic data and is intended as a simplified, reproducible example of anomaly detection in healthcare cost monitoring.

It does not use real patient or hospital data.

## Author

Najmieh Maksaei  
Postdoctoral Researcher in Statistics  
Université Paris 1 Panthéon-Sorbonne
