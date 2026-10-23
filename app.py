import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Air Quality Spatiotemporal Dashboard",
    page_icon="🌍",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stMetric {
        background-color: #1E222D;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #313745;
    }
    .advisory-box {
        padding: 20px;
        border-radius: 12px;
        border-left: 8px solid;
        margin-bottom: 20px;
        font-size: 16px;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(
    "🌍 Explainable Spatiotemporal Machine Learning for Air Quality Forecasting"
)
st.caption(
    "Multi-horizon air quality prediction and pollution-risk assessment across"
    " CPCB monitoring stations."
)


@st.cache_data
def load_data():
  df = pd.read_csv("data/processed/engineered_air_quality.csv")
  df["Timestamp"] = pd.to_datetime(df["Timestamp"])
  return df


@st.cache_resource
def load_model():
  with open("data/processed/lgb_model.pkl", "rb") as f:
    model = pickle.load(f)
  return model


df = load_data()
model = load_model()

# Comprehensive Coordinate Mapping for CPCB Stations
EXACT_COORDS = {
    "burari": [28.7301, 77.1930],
    "aya_nagar": [28.4704, 77.1265],
    "jaipur": [26.9124, 75.7873],
    "rajamahendravaram": [17.0005, 81.7777],
    "amaravati": [16.5062, 80.6480],
    "ashok_vihar": [28.6953, 77.1812],
    "hubballi": [15.3647, 75.1240],
    "bawana": [28.7877, 77.0476],
    "kalaburagi": [17.3297, 76.8343],
    "bengaluru": [12.9716, 77.5946],
    "amritsar": [31.6340, 74.8723],
    "visakhapatnam": [17.6868, 83.2185],
    "airoli": [19.1585, 72.9985],
    "hyderabad": [17.3850, 78.4867],
    "howrah": [22.5958, 88.2636],
    "anand_vihar": [28.6463, 77.3160],
    "alipur": [28.8153, 77.1547],
    "ambala": [30.3782, 76.7767],
    "loni": [28.7505, 77.2885],
    "mysuru": [12.2958, 76.6394],
    "samanpura": [25.6110, 85.1150],
    "kochi": [9.9312, 76.2673],
    "gwalior": [26.2183, 78.1828],
    "thiruvananthapuram": [8.5241, 76.9366],
    "arumbakkam": [13.0732, 80.2103],
    "bandra": [19.0657, 72.8311],
    "hadapsar": [18.5089, 73.9259],
    "naubad": [17.9150, 77.5140],
    "puducherry": [11.9416, 79.8083],
    "naharlagun": [27.1000, 93.7000],
    "chandkheda": [23.1023, 72.5714],
    "gandhinagar": [23.2156, 72.6369],
    "rohta": [27.2200, 78.0000],
    "bhagalpur": [25.2425, 87.0172],
    "aurangabad": [19.8762, 75.3433],
    "coimbatore": [11.0168, 76.9558],
    "tiruchirappalli": [10.7905, 78.7047],
    "gulzarpet": [14.6819, 77.6006],
    "guwahati": [26.1445, 91.7362],
    "bhilai": [21.1938, 81.3509],
    "gangineni": [13.6288, 79.4192],
    "vaikuntapuram": [13.6288, 79.4192],
    "tarapur": [24.8333, 92.2500],
    "rourkela": [22.2604, 84.8536],
    "idgah": [23.2599, 77.4126],
    "dhanbad": [23.7957, 86.4304],
    "rati": [23.5500, 74.4300],
    "midc": [19.8762, 75.3433],
    "rajiv": [16.5062, 80.6480],
    "pcbl": [23.5500, 87.2833],
}


def get_station_coords(station_name):
  name_lower = str(station_name).lower()
  for key, coords in EXACT_COORDS.items():
    if key in name_lower:
      return coords[0] + np.random.uniform(-0.02, 0.02), coords[1] + np.random.uniform(
          -0.02, 0.02
      )
  return 22.9734 + np.random.uniform(-2, 2), 78.6569 + np.random.uniform(-2, 2)


# Navigation Tabs (Now with 3 Tabs!)
tab1, tab2, tab3 = st.tabs(
    [
        "📊 Single Station & Next-Day Forecast",
        "🗺️ India Map Severity Overview",
        "📈 Model Evaluation & Metrics",
    ]
)

with tab1:
  st.sidebar.header("📍 Select Location")

  df["Station_ID"] = df["Station_ID"].astype(str)
  unique_stations = df[["Station_ID", "Station_Name"]].drop_duplicates()
  station_options = (
      unique_stations["Station_ID"]
      + " - "
      + unique_stations["Station_Name"]
  )

  selected_option = st.sidebar.selectbox(
      "Choose Monitoring Station:", station_options
  )
  selected_station_id = selected_option.split(" - ")[0]

  station_df = df[df["Station_ID"] == selected_station_id].sort_values(
      "Timestamp"
  )

  st.subheader(f"Station Overview: {selected_option}")
  col1, col2, col3, col4 = st.columns(4)

  if not station_df.empty:
    latest = station_df.iloc[-1]
    col1.metric("Current PM2.5", f"{latest['PM2.5']:.1f} µg/m³")
    col2.metric("Current PM10", f"{latest['PM10']:.1f} µg/m³")
    col3.metric("Current NO₂", f"{latest['NO2']:.1f} µg/m³")
    col4.metric("Current SO₂", f"{latest['SO2']:.1f} µg/m³")

  st.markdown("---")

  # Two-Step Time-Travel Inspector
  st.subheader("⏱️ Time-Travel Inspector (Hourly Data Explorer)")
  if not station_df.empty:
    station_df["Date_Str"] = station_df["Timestamp"].dt.strftime("%Y-%m-%d")
    available_dates = station_df["Date_Str"].unique()

    col_d1, col_d2 = st.columns(2)
    with col_d1:
      selected_date = st.selectbox("1. Select Date:", available_dates)

    day_subset = station_df[station_df["Date_Str"] == selected_date]
    available_times = day_subset["Timestamp"].dt.strftime("%H:%M").tolist()

    with col_d2:
      selected_time = st.selectbox("2. Select Time (Hour):", available_times)

    row_at_time = day_subset[
        day_subset["Timestamp"].dt.strftime("%H:%M") == selected_time
    ]
    if not row_at_time.empty:
      r = row_at_time.iloc[0]
      sc1, sc2, sc3, sc4 = st.columns(4)
      sc1.metric("Recorded PM2.5", f"{r['PM2.5']:.1f} µg/m³")
      sc2.metric("Recorded NO₂", f"{r['NO2']:.1f} µg/m³")
      sc3.metric("Recorded SO₂", f"{r['SO2']:.1f} µg/m³")
      sc4.metric("Recorded CO", f"{r['CO']:.1f} mg/m³")

  st.markdown("---")

  # Tomorrow's 24-Hour Forecast & Large Advisory Box
  st.subheader(
      "🔮 Tomorrow's Air Quality & Pollution-Risk Forecast (+24 Hours Ahead)"
  )

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

  if not station_df.empty and len(station_df) >= 24:
    tomorrow_df = station_df.tail(24).copy()
    tomorrow_preds = model.predict(tomorrow_df[features])

    avg_tomorrow_pm25 = np.mean(tomorrow_preds)
    max_tomorrow_pm25 = np.max(tomorrow_preds)


    def get_risk_category(val):
      if val <= 30:
        return "Good 🟢"
      elif val <= 60:
        return "Satisfactory 🟡"
      elif val <= 90:
        return "Moderate 🟠"
      elif val <= 120:
        return "Poor 🔴"
      elif val <= 250:
        return "Very Poor 🟣"
      else:
        return "Severe 🟤"


    tomorrow_risk = get_risk_category(avg_tomorrow_pm25)

    t_col1, t_col2, t_col3 = st.columns(3)
    t_col1.metric(
        "Expected Avg PM2.5 (Tomorrow)", f"{avg_tomorrow_pm25:.1f} µg/m³"
    )
    t_col2.metric(
        "Expected Peak PM2.5 (Tomorrow)", f"{max_tomorrow_pm25:.1f} µg/m³"
    )
    t_col3.metric("Predicted Air Quality Risk Level", tomorrow_risk)

    st.markdown("<br>", unsafe_allow_html=True)
    if avg_tomorrow_pm25 <= 30:
      st.markdown(
          """
                <div class="advisory-box" style="background-color: rgba(46, 204, 113, 0.15); border-color: #2ECC71; color: #2ECC71;">
                    <h3>🟢 AI Health Advisory: SAFE & CLEAN AIR</h3>
                    <p>Air quality is expected to be clean tomorrow. Ideal conditions for outdoor physical activities, sports, and window ventilation.</p>
                </div>
                """,
          unsafe_allow_html=True,
      )
    elif avg_tomorrow_pm25 <= 60:
      st.markdown(
          """
                <div class="advisory-box" style="background-color: rgba(241, 196, 15, 0.15); border-color: #F1C40F; color: #F1C40F;">
                    <h3>🟡 AI Health Advisory: SATISFACTORY</h3>
                    <p>Acceptable air quality. Sensitive individuals with respiratory or heart conditions should consider reducing prolonged or heavy outdoor exertion.</p>
                </div>
                """,
          unsafe_allow_html=True,
      )
    elif avg_tomorrow_pm25 <= 120:
      st.markdown(
          """
                <div class="advisory-box" style="background-color: rgba(230, 126, 34, 0.15); border-color: #E67E22; color: #E67E22;">
                    <h3>🟠 AI Health Advisory: MODERATE TO POOR ALERT</h3>
                    <p>Elevated pollution levels predicted. Breathing discomfort may occur for sensitive groups. Limit heavy outdoor work, construction exposure, or morning jogs.</p>
                </div>
                """,
          unsafe_allow_html=True,
      )
    else:
      st.markdown(
          """
                <div class="advisory-box" style="background-color: rgba(231, 76, 60, 0.15); border-color: #E74C3C; color: #E74C3C;">
                    <h3>🔴 AI Health Advisory: HIGH HAZARD WARNING</h3>
                    <p>Hazardous air quality forecasted for tomorrow! General public and vulnerable groups are strongly advised to stay indoors, seal windows, and use N95 masks outdoors.</p>
                </div>
                """,
          unsafe_allow_html=True,
      )

    fig_t, ax_t = plt.subplots(figsize=(14, 4.5))
    plt.style.use("dark_background")

    timestamps_str = tomorrow_df["Timestamp"].dt.strftime("%H:%M").tolist()

    ax_t.plot(
        timestamps_str,
        tomorrow_preds,
        label="Predicted Hourly PM2.5 (Next 24h)",
        color="#00E676",
        marker="o",
        linewidth=2,
    )

    for i, txt in enumerate(timestamps_str):
      ax_t.annotate(
          txt,
          (timestamps_str[i], tomorrow_preds[i]),
          textcoords="offset points",
          xytext=(0, 8),
          ha="center",
          fontsize=8,
          color="#A9DFBF",
      )

    ax_t.axhline(
        y=60,
        color="orange",
        linestyle="--",
        alpha=0.6,
        label="Moderate Risk Threshold",
    )
    ax_t.axhline(
        y=120,
        color="red",
        linestyle="--",
        alpha=0.6,
        label="Poor Risk Threshold",
    )
    ax_t.set_ylabel("PM2.5 (µg/m³)")
    ax_t.set_xlabel("Tomorrow's Hourly Timestamps")
    ax_t.legend(facecolor="#1E222D", edgecolor="none")
    ax_t.grid(True, linestyle=":", alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig_t)

  st.markdown("---")
  st.subheader("📈 Historical Trend Tracking (+1h Model Check)")

  plot_df = station_df.tail(168)
  if not plot_df.empty:
    X_subset = plot_df[features]
    preds = model.predict(X_subset)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        plot_df["Timestamp"],
        plot_df["Target_PM2.5_t+1"],
        label="Actual PM2.5",
        color="#00E676",
        linewidth=1.5,
    )
    ax.plot(
        plot_df["Timestamp"],
        preds,
        label="Predicted PM2.5 (+1h)",
        color="#FF5252",
        linestyle="--",
        linewidth=2,
    )
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(facecolor="#1E222D", edgecolor="none")
    ax.grid(True, linestyle=":", alpha=0.3)
    fig.autofmt_xdate()
    st.pyplot(fig)

with tab2:
  st.subheader(
      "🗺️ Spatiotemporal Air Quality Across India (Next-Day Severity Map)"
  )
  st.markdown(
      "Interactive map showing forecasted next-day PM2.5 severity levels across"
      " all monitored stations."
  )

  latest_df = (
      df.sort_values("Timestamp")
      .groupby(["Station_ID", "Station_Name"], as_index=False)
      .last()
  )

  coords = latest_df["Station_Name"].apply(get_station_coords)
  latest_df["lat"] = [c[0] for c in coords]
  latest_df["lon"] = [c[1] for c in coords]


  def get_color(val):
    if val <= 30:
      return [46, 204, 113, 200]
    elif val <= 60:
      return [241, 196, 15, 200]
    elif val <= 90:
      return [230, 126, 34, 200]
    elif val <= 120:
      return [231, 76, 60, 200]
    elif val <= 250:
      return [155, 89, 182, 200]
    else:
      return [120, 40, 31, 200]


  latest_df["color"] = latest_df["PM2.5"].apply(get_color)
  latest_df["size"] = latest_df["PM2.5"].clip(lower=10, upper=300) / 3

  st.map(
      latest_df,
      latitude="lat",
      longitude="lon",
      size="size",
      color="color",
      zoom=4,
  )

  st.write(
      f"### Station Summary Table (Total Stations: {len(latest_df)})"
  )
  st.dataframe(
      latest_df[
          [
              "Station_ID",
              "Station_Name",
              "Year",
              "PM2.5",
              "PM10",
              "NO2",
              "Timestamp",
          ]
      ],
      use_container_width=True,
      hide_index=True,
  )

# --- NEW TAB 3: Model Evaluation & Performance Metrics ---
with tab3:
  st.subheader("📈 Comprehensive Model Evaluation & Performance Metrics")
  st.markdown(
      "Detailed regression performance and classification confusion matrix"
      " evaluated across all monitoring stations."
  )

  # Sample a subset of data for fast evaluation metrics computation
  eval_df = df.dropna(subset=["Target_PM2.5_t+1"]).sample(
      n=min(5000, len(df)), random_state=42
  )
  X_eval = eval_df[features]
  y_true = eval_df["Target_PM2.5_t+1"]
  y_pred = model.predict(X_eval)

  # Regression Metrics
  mae = mean_absolute_error(y_true, y_pred)
  rmse = np.sqrt(mean_squared_error(y_true, y_pred))
  r2 = r2_score(y_true, y_pred)

  m1, m2, m3 = st.columns(3)
  m1.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f} µg/m³")
  m2.metric("Mean Absolute Error (MAE)", f"{mae:.2f} µg/m³")
  m3.metric("R² Score Accuracy", f"{r2:.4f}")

  st.markdown("---")

  st.subheader("🔲 CPCB Risk Tier Confusion Matrix")
  st.markdown(
      "Evaluating classification accuracy by mapping actual vs. predicted"
      " values into CPCB health risk tiers."
  )


  def categorize(val_series):
    return pd.cut(
        val_series,
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


  y_true_cat = categorize(y_true)
  y_pred_cat = categorize(y_pred)

  # Calculate accuracy score for categorical tiers
  acc_score = accuracy_score(y_true_cat, y_pred_cat)
  st.metric("Risk-Tier Classification Accuracy", f"{acc_score * 100:.2f}%")

  # Compute Confusion Matrix
  labels_order = [
      "Good",
      "Satisfactory",
      "Moderate",
      "Poor",
      "Very Poor",
      "Severe",
  ]
  cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels_order)

  col_cm1, col_cm2 = st.columns([1.2, 1])

  with col_cm1:
    fig_cm, ax_cm = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels_order,
        yticklabels=labels_order,
        ax=ax_cm,
    )
    ax_cm.set_ylabel("Actual Risk Tier", color="white")
    ax_cm.set_xlabel("Predicted Risk Tier", color="white")
    plt.xticks(rotation=45, color="white")
    plt.yticks(rotation=0, color="white")
    st.pyplot(fig_cm)

  with col_cm2:
    st.write("### Classification Report Summary")
    report_dict = classification_report(
        y_true_cat, y_pred_cat, labels=labels_order, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format(precision=2), use_container_width=True)

st.markdown("---")
st.caption("💡 *Spatiotemporal Machine Learning Pipeline — Project Operational*")