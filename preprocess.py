import glob
import os
import numpy as np
import pandas as pd


def clean_cpcb_file(filepath):
  print(f"Processing: {filepath}")

  # Read CSV/Excel safely
  try:
    if filepath.endswith(".xlsx") or filepath.endswith(".xls"):
      df = pd.read_excel(filepath)
    else:
      df = pd.read_csv(filepath, low_memory=False)
  except Exception as e:
    print(f"  ❌ Error reading {filepath}: {e}")
    return None

  # Find the column that represents the timestamp/date
  timestamp_col = None
  for col in df.columns:
    if "timestamp" in str(col).lower() or "date" in str(col).lower():
      timestamp_col = col
      break

  if timestamp_col is None:
    print(f"  ⚠️ Timestamp column not found in {filepath}")
    return None

  # Rename found timestamp column to standard 'Timestamp'
  df = df.rename(columns={timestamp_col: "Timestamp"})

  # 1. Handle CPCB Wide-Format Issue (repeating timestamp blocks horizontally)
  expected_cols = [
      "Timestamp",
      "PM2.5",
      "PM10",
      "NO",
      "NO2",
      "NOx",
      "NH3",
      "SO2",
      "CO",
      "Ozone",
  ]
  valid_cols = []
  for base in expected_cols:
    matched = [c for c in df.columns if str(c).strip().startswith(base)]
    if matched:
      valid_cols.append(matched[0])  # Take the first occurrence block

  df = df[valid_cols]
  # Clean column names
  df.columns = [str(c).split()[0].strip() for c in df.columns]

  # 2. Standardize Timestamp & Set as DatetimeIndex
  df["Timestamp"] = pd.to_datetime(
      df["Timestamp"], errors="coerce", dayfirst=True
  )
  df = df.dropna(subset=["Timestamp"])
  df = df.sort_values("Timestamp")

  if len(df) == 0:
    print(f"  ⚠️ Warning: 0 valid timestamp rows found in {filepath}")
    return None

  # Set Timestamp as index for time-based interpolation
  df = df.set_index("Timestamp")

  # 3. Handle Negative Error Codes & Non-numeric values
  pollutant_cols = df.columns
  for col in pollutant_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df[col] < 0, col] = np.nan  # CPCB uses negative values for errors

  # 4. Interpolate missing hourly data (up to 6 hours gap) and fill remaining with median
  df[pollutant_cols] = df[pollutant_cols].interpolate(method="time", limit=6)
  df[pollutant_cols] = df[pollutant_cols].fillna(df[pollutant_cols].median())

  # Reset index back so 'Timestamp' becomes a column again
  df = df.reset_index()

  return df


# Main execution for batch processing all files in data/raw/
if __name__ == "__main__":
  os.makedirs("data/processed", exist_ok=True)
  all_station_data = []

  files = glob.glob("data/raw/*.csv") + glob.glob("data/raw/*.xlsx")

  if not files:
    print(
        "⚠️ No files found! Please place your station files inside the"
        " 'data/raw/' folder."
    )
  else:
    for file in files:
      filename = os.path.basename(file)
      parts = filename.split("_")

      try:
        year = parts[2]
        station_id = parts[4]
        station_name = "_".join(parts[5:-2])
      except IndexError:
        year = "Unknown"
        station_id = "000"
        station_name = filename

      df_station = clean_cpcb_file(file)
      if df_station is not None and not df_station.empty:
        df_station["Station_ID"] = station_id
        df_station["Station_Name"] = station_name
        df_station["Year"] = year
        all_station_data.append(df_station)

    if all_station_data:
      master_df = pd.concat(all_station_data, ignore_index=True)
      output_path = "data/processed/master_air_quality.csv"
      master_df.to_csv(output_path, index=False)
      print(f"\n✅ Successfully saved master dataset to {output_path}")
      print(master_df.info())
    else:
      print("❌ No data was successfully processed across all files.")