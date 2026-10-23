import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

print("Loading engineered dataset...")
df = pd.read_csv("data/processed/engineered_air_quality.csv")

# Define Features
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

# Multi-horizons to predict: +1 hour, +6 hours, +24 hours
horizons = {
    "+1 Hour": "Target_PM2.5_t+1",
    "+24 Hours": "Target_PM2.5_t+24",  # Note: ensure t+24 target exists in features.py or create it here
}

# If t+24 wasn't explicitly shifted in features.py, let's create it on the fly:
df["Target_PM2.5_t+24"] = df.groupby("Station_ID")["PM2.5"].shift(-24)
df_clean = df.dropna(subset=["Target_PM2.5_t+24"]).copy()

X = df_clean[features]

for label, target_col in horizons.items():
  print(f"\n--- Training Model for Horizon: {label} ---")
  y = df_clean[target_col]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, shuffle=False
  )

  model = lgb.LGBMRegressor(
      n_estimators=100, learning_rate=0.05, random_state=42
  )
  model.fit(X_train, y_train)

  preds = model.predict(X_test)
  rmse = np.sqrt(mean_squared_error(y_test, preds))
  mae = mean_absolute_error(y_test, preds)

  print(f"[{label}] RMSE: {rmse:.2f} | MAE: {mae:.2f}")

  # Pollution-Risk Assessment Classification Logic (Based on CPCB PM2.5 breakpoints)
  # 0-30: Good, 30-60: Satisfactory, 60-90: Moderate, 90-120: Poor, 120-250: Very Poor, >250: Severe
  risk_preds = pd.cut(
      preds,
      bins=[-np.inf, 30, 60, 90, 120, 250, np.inf],
      labels=[
          "Good",
          "Satisfactory",
          "Moderate",
          "Poor",
          "Very Poor",
          "Severe",
      ],
  )
  print(f"Risk Assessment Sample Breakdown for {label}:")
  print(risk_preds.value_counts().head(3))

print(
    "\n✅ Multi-horizon forecasting and risk categorization script executed"
    " successfully!"
)