# 🌍 Explainable Spatiotemporal Machine Learning for Air Quality Forecasting & Risk Assessment

An end-to-end, IEEE-grade spatiotemporal machine learning framework and interactive web dashboard for multi-horizon $PM_{2.5}$ forecasting and pollution-risk assessment across Central Pollution Control Board (CPCB) monitoring stations in India.

---

## 🚀 Key Features
* **Multi-Horizon Forecasting:** Predicts hourly pollutant trends ($PM_{2.5}$) up to 24 hours ahead using optimized LightGBM regression.
* **Spatiotemporal Feature Engineering:** Incorporates rolling means, temporal lag variables (1h, 24h), and cyclical trigonometric time encodings (`Hour_Sin`, `Hour_Cos`).
* **Interactive Streamlit Dashboard (`app.py`):**
  * **Single Station & Next-Day Outlook:** Real-time metrics, a 2-step time-travel hourly inspector, next-day hourly trends, and prominent AI-driven health advisories.
  * **India Map Severity Overview:** Interactive nationwide mapping of stations categorized by CPCB risk tiers.
  * **Model Evaluation Tab:** Comprehensive regression metrics ($R^2$, RMSE, MAE), CPCB risk-tier confusion matrix, and classification performance reports.
* **Model Explainability:** Global and local SHAP (SHapley Additive exPlanations) integration to interpret feature contributions.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Machine Learning:** LightGBM, Scikit-learn, Pandas, NumPy
* **Explainability:** SHAP
* **Visualization:** Matplotlib, Seaborn, Streamlit Maps
* **Deployment:** Streamlit

---

## 📊 Performance & Evaluation Metrics
Evaluated across a rigorous multi-year benchmark of 50 monitoring stations:
* **$R^2$ Score:** `0.8842` (Explains 88.42% of data variance)
* **Root Mean Squared Error (RMSE):** `16.73 µg/m³`
* **Mean Absolute Error (MAE):** `7.36 µg/m³`
* **Risk-Tier Classification Accuracy:** `82.28%` (Across 6 CPCB health risk categories from *Good* to *Severe*)

---

## ⚙️ Installation & Local Setup

To run this dashboard locally for faculty review or development:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/abhi-114/air-quality-spatiotemporal-forecasting.git](https://github.com/abhi-114/air-quality-spatiotemporal-forecasting.git)
   cd air-quality-spatiotemporal-forecasting