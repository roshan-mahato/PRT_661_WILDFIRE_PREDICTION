"""
join_firms_weather.py

Purpose:
Join your FIRMS hotspot CSV with the weather_data.csv fetched earlier,
matching each fire record to its weather conditions by:
  - location (lat_round, lon_round — same grid used when fetching weather)
  - date
  - hour

Weather fields included (from fetch_weather_csv.py):
temperature_2m, relative_humidity_2m, wind_speed_10m, wind_direction_10m,
precipitation, wind_gusts_10m, soil_moisture_0_to_7cm, surface_pressure,
vapour_pressure_deficit, et0_fao_evapotranspiration

Output: firms_weather_merged.csv — one row per fire hotspot, with all
weather columns above attached. This is the file your ML teammate can
use to build the training feature matrix.

Before running:
- Make sure fetch_weather_csv.py has finished (all calls done, not partial).
- Update FIRMS_CSV_PATH and WEATHER_CSV_PATH below if needed.
"""

import pandas as pd

# ---------- CONFIG ----------
FIRMS_CSV_PATH = "data/fire_archive_SV-C2_792465.csv"
WEATHER_CSV_PATH = "data/all_weather_data.csv"
OUTPUT_CSV_PATH = "data/firms_weather_merged.csv"
GRID_SIZE_DEGREES = 0.5   # must match what was used in fetch_weather_csv.py


def round_to_grid(value, grid_size):
    return round(round(value / grid_size) * grid_size, 3)


def load_and_prepare_firms(path):
    df = pd.read_csv(path)
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.strftime("%Y-%m-%d")
    df["acq_hour"] = df["acq_time"].astype(str).str.zfill(4).str[:2].astype(int)
    df["lat_round"] = df["latitude"].apply(lambda v: round_to_grid(v, GRID_SIZE_DEGREES))
    df["lon_round"] = df["longitude"].apply(lambda v: round_to_grid(v, GRID_SIZE_DEGREES))
    return df


def load_and_prepare_weather(path):
    df = pd.read_csv(path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df["weather_date"] = df["datetime_utc"].dt.strftime("%Y-%m-%d")
    df["weather_hour"] = df["datetime_utc"].dt.hour
    return df


def main():
    print("Loading FIRMS data...")
    firms_df = load_and_prepare_firms(FIRMS_CSV_PATH)
    print(f"Loaded {len(firms_df)} hotspot records.")

    print("Loading weather data...")
    weather_df = load_and_prepare_weather(WEATHER_CSV_PATH)
    print(f"Loaded {len(weather_df)} weather rows.")

    print("Merging...")
    merged = firms_df.merge(
        weather_df,
        left_on=["lat_round", "lon_round", "acq_date", "acq_hour"],
        right_on=["lat_round", "lon_round", "weather_date", "weather_hour"],
        how="left",  # keep all fire records, even if a weather match is missing
    )

    matched = merged["temperature_2m"].notna().sum()
    print(f"Matched weather data for {matched}/{len(merged)} hotspot records "
          f"({matched/len(merged)*100:.1f}%).")

    if matched < len(merged):
        missing = len(merged) - matched
        print(f"Note: {missing} records have no weather match yet — this is "
              f"expected if fetch_weather_csv.py hasn't finished all calls.")

    merged.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"Saved merged dataset to {OUTPUT_CSV_PATH} ({len(merged)} rows).")


if __name__ == "__main__":
    main()