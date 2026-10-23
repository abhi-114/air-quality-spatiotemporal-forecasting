import matplotlib.pyplot as plt
import pandas as pd
import pickle
import shap

print("Loading model and test data...")
with open("data/processed/lgb_model.pkl", "rb") as f:
  model = pickle.load(f)

df = pd.read_csv("data/processed/engineered_air_quality.csv")
features = [
    "PM2.5",
    "PM10",
    "NO2",
    "SO2",
    "CO",
    "Hour",
    "DayOfWeek",
    "Month",
    "Hour_Sin",
    "Hour_Cos",
    "PM2.5_Lag_1h",
    "PM2.5_Lag_24h",
    "PM2.5_Rolling_Mean_24h",
]

X_sample = df[features].sample(
    1000, random_state=42
)  # Sample for fast SHAP computation

print("Calculating SHAP values...")
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)

# Plot Global Feature Importance Summary
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, show=False)
plt.title(
    "SHAP Global Feature Importance for Air Quality Forecasting", fontsize=14
)
plt.tight_layout()
plt.savefig("data/processed/shap_summary_plot.png")
print(
    "✅ SHAP summary plot saved as 'data/processed/shap_summary_plot.png'!"
)