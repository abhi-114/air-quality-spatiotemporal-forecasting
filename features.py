import numpy as np
import pandas as pd

print("Loading cleaned master dataset...")
df = pd.read_csv("data/processed/master_air_quality.csv")
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Sort values by station and time
df = df.sort_values(by=["Station_ID", "Timestamp"])

print("Engineering temporal and lag features...")

# 1. Temporal Features
df["Hour"] = df["Timestamp"].dt.hour
df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
df["Month"] = df["Timestamp"].dt.month

# Cyclical Encoding for Time (helps machine learning models understand hours/months smoothly)
df["Hour_Sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
df["Hour_Cos"] = np.cos(2 * np.pi * df["Hour"] / 24)

# 2. Lag Features & Rolling Windows for PM2.5 (Our primary target)
# We group by station so lags don't bleed across different cities/stations
target = "PM2.5"

df[f"{target}_Lag_1h"] = df.groupby("Station_ID")[target].shift(1)
df[f"{target}_Lag_24h"] = df.groupby("Station_ID")[target].shift(24)

# Rolling Averages (e.g., 24-hour moving average)
df[f"{target}_Rolling_Mean_24h"] = (
    df.groupby("Station_ID")[target]
    .shift(1)
    .rolling(window=24, min_periods=1)
    .mean()
)

# Multi-Horizon Targets (e.g., predicting PM2.5 1 hour ahead and 24 hours ahead)
df["Target_PM2.5_t+1"] = df.groupby("Station_ID")[target].shift(-1)
df["Target_PM2.5_t+24"] = df.groupby("Station_ID")[target].shift(-24)

# Drop rows with NaN resulting from shifts/lags
df_model = df.dropna().copy()

output_path = "data/processed/engineered_air_quality.csv"
df_model.to_csv(output_path, index=False)
print(f"✅ Feature engineering complete! Saved to {output_path}")