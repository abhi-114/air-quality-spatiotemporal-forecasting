import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

print("Loading engineered dataset...")
df = pd.read_csv("data/processed/engineered_air_quality.csv")

# Define Features and Target (Predicting PM2.5 1 hour ahead)
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
target = "Target_PM2.5_t+1"

X = df[features]
y = df[target]

# Train-Test Split (Chronological split is best for time series)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("Training LightGBM Multi-Horizon Forecasting Model...")
model = lgb.LGBMRegressor(
    n_estimators=100, learning_rate=0.05, random_state=42
)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)

print("\n--- Model Performance Results ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# Save model object for explainability step
import pickle

with open("data/processed/lgb_model.pkl", "wb") as f:
  pickle.dump(model, f)
print("✅ Model saved successfully!")