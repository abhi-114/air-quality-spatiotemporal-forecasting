import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import pickle

print("Loading test data and model for visualization...")
df = pd.read_csv("data/processed/engineered_air_quality.csv")

# Load the model trained earlier
with open("data/processed/lgb_model.pkl", "rb") as f:
  model = pickle.load(f)

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

# Pick a specific prominent station to plot (e.g., Anand Vihar, Delhi or station 301)
sample_station_id = df["Station_ID"].unique()[
    0
]  # Selects the first station in dataset
df_station = df[df["Station_ID"] == sample_station_id].tail(
    168
)  # Last 1 week of hourly data (24 * 7)

X_plot = df_station[features]
actuals = df_station["Target_PM2.5_t+1"]

# Predict
predictions = model.predict(X_plot)

# Plotting Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(
    df_station["Timestamp"],
    actuals,
    label="Actual PM2.5",
    color="black",
    linewidth=1.5,
)
plt.plot(
    df_station["Timestamp"],
    predictions,
    label="Predicted PM2.5 (+1h)",
    color="crimson",
    linestyle="--",
    linewidth=2,
)

station_name = df_station["Station_Name"].iloc[0]
plt.title(
    f"Multi-Horizon PM2.5 Forecasting vs Actuals — Station: {station_name}",
    fontsize=14,
)
plt.xlabel("Timestamp", fontsize=12)
plt.ylabel("PM2.5 Concentration ($\mu g/m^3$)", fontsize=12)
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.xticks(rotation=45)
plt.tight_layout()

output_plot_path = "data/processed/actual_vs_predicted_trend.png"
plt.savefig(output_plot_path)
print(f"✅ Evaluation trend plot saved successfully to '{output_plot_path}'!")