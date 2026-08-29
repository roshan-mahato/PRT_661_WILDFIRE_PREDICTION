
import os
import time
from datetime import timedelta
 
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
 
# ---------- CONFIG ----------
FIRMS_CSV_PATH = "data/fire_nrt_SV-C2_792465.csv"
OUTPUT_CSV_PATH = "data/all_weather_data.csv"
PROGRESS_LOG_PATH = "data/fetch_progress.log"
 
GRID_SIZE_DEGREES = 0.5   # ~55km cells — better spatial precision, ~4 days to complete with the daily budget below
CHUNK_DAYS = 14
REQUEST_DELAY_SECONDS = 1
DAILY_CALL_LIMIT = 9900   # pushed close to the 10,000/day free limit, small safety buffer only
 
HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "precipitation",
    "wind_gusts_10m",              # gusts matter more than average wind for fire spread
    "soil_moisture_0_to_7cm",      # feeds into Drought Factor estimation
    "cape",                        # convective available potential energy — dry lightning/ignition risk
    "vapour_pressure_deficit",     # air dryness, closely tied to fire risk
    "et0_fao_evapotranspiration",  # rate of fuel drying
]  # exactly 10 — the threshold before extra fields start costing more than 1 call each
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
 
cache_session = requests_cache.CachedSession(".cache", expire_after=86400)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
 
 
def round_to_grid(value, grid_size):
    """Round a coordinate to the nearest grid_size step. The round(..., 3)
    removes floating-point noise (e.g. 152.49999999999997 instead of 152.5)
    that would otherwise create fake duplicate locations."""
    return round(round(value / grid_size) * grid_size, 3)
 
 
def load_firms_data(path):
    df = pd.read_csv(path)
    df["acq_date"] = pd.to_datetime(df["acq_date"]).dt.strftime("%Y-%m-%d")
    df["lat_round"] = df["latitude"].apply(lambda v: round_to_grid(v, GRID_SIZE_DEGREES))
    df["lon_round"] = df["longitude"].apply(lambda v: round_to_grid(v, GRID_SIZE_DEGREES))
    return df
 
 
def get_locations_and_dates(df):
    df["acq_date_dt"] = pd.to_datetime(df["acq_date"])
    return df.groupby(["lat_round", "lon_round"])["acq_date_dt"].apply(list).reset_index()
 
 
def build_date_chunks(dates, chunk_days):
    """Only build chunks around days that actually have fire records,
    skipping long empty gaps between fire clusters."""
    dates = sorted(set(dates))
    chunks = []
    i = 0
    while i < len(dates):
        start = dates[i]
        window_end = start + timedelta(days=chunk_days - 1)
        j = i
        while j < len(dates) and dates[j] <= window_end:
            j += 1
        end = dates[j - 1]
        chunks.append((start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
        i = j
    return chunks
 
 
def call_key(lat, lon, start_date, end_date):
    """A unique string identifying one API call, used for the progress log."""
    return f"{lat},{lon},{start_date},{end_date}"
 
 
def load_completed_calls():
    """Read the progress log and return the set of already-completed calls."""
    if not os.path.exists(PROGRESS_LOG_PATH):
        return set()
    with open(PROGRESS_LOG_PATH, "r") as f:
        return set(line.strip() for line in f if line.strip())
 
 
def mark_call_complete(key):
    """Append one completed call to the progress log immediately (not buffered),
    so progress is safe even if the script stops right after this call."""
    with open(PROGRESS_LOG_PATH, "a") as f:
        f.write(key + "\n")
 
 
def fetch_weather_chunk(lat, lon, start_date, end_date):
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": HOURLY_VARS,
        "timezone": "UTC",  # matches FIRMS acq_time (UTC); convert to AEST later in ETL
    }
    responses = openmeteo.weather_api(ARCHIVE_URL, params=params)
    response = responses[0]
    hourly = response.Hourly()
 
    hourly_data = {
        "datetime_utc": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }
    for i, var_name in enumerate(HOURLY_VARS):
        hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()
 
    df = pd.DataFrame(hourly_data)
    df["lat_round"] = lat
    df["lon_round"] = lon
    return df
 
 
def append_to_output_csv(df):
    """Append rows to the output CSV. Writes the header only if the file
    doesn't exist yet (first write of the whole job)."""
    file_exists = os.path.exists(OUTPUT_CSV_PATH)
    df.to_csv(OUTPUT_CSV_PATH, mode="a", header=not file_exists, index=False)
 
 
def main():
    print("Loading FIRMS CSV...")
    df = load_firms_data(FIRMS_CSV_PATH)
    print(f"Loaded {len(df)} hotspot records.")
 
    locations = get_locations_and_dates(df)
    print(f"Found {len(locations)} unique locations.")
 
    all_calls = []
    for _, row in locations.iterrows():
        chunks = build_date_chunks(row["acq_date_dt"], CHUNK_DAYS)
        for start_date, end_date in chunks:
            all_calls.append((row["lat_round"], row["lon_round"], start_date, end_date))
 
    total_calls_needed = len(all_calls)
    print(f"Total API calls needed overall: {total_calls_needed}")
 
    completed = load_completed_calls()
    print(f"Already completed (from previous runs): {len(completed)}")
 
    remaining_calls = [
        c for c in all_calls if call_key(c[0], c[1], c[2], c[3]) not in completed
    ]
    print(f"Remaining calls to fetch: {len(remaining_calls)}")
 
    if not remaining_calls:
        print("Nothing left to fetch. All data already collected.")
        return
 
    calls_this_run = 0
    for lat, lon, start_date, end_date in remaining_calls:
        if calls_this_run >= DAILY_CALL_LIMIT:
            done_total = len(completed) + calls_this_run
            print(f"\nReached today's call budget ({DAILY_CALL_LIMIT}).")
            print(f"Progress: {done_total}/{total_calls_needed} calls done.")
            print(f"Run this script again tomorrow — it will resume automatically "
                  f"from call {done_total + 1}.")
            return
 
        key = call_key(lat, lon, start_date, end_date)
        try:
            chunk_df = fetch_weather_chunk(lat, lon, start_date, end_date)
            append_to_output_csv(chunk_df)
            mark_call_complete(key)
            calls_this_run += 1
        except Exception as e:
            print(f"FAILED for ({lat}, {lon}, {start_date}-{end_date}): {e}")
            # Not marked complete, so it will be retried on the next run
 
        done_total = len(completed) + calls_this_run
        if calls_this_run % 20 == 0:
            print(f"Progress: {done_total}/{total_calls_needed} calls done "
                  f"({calls_this_run} this run)")
 
        time.sleep(REQUEST_DELAY_SECONDS)
 
    done_total = len(completed) + calls_this_run
    print(f"\nAll done for now. Progress: {done_total}/{total_calls_needed} calls done.")
    if done_total >= total_calls_needed:
        print("Fetching complete! All weather data has been saved to "
              f"{OUTPUT_CSV_PATH}")
    else:
        print("Run this script again to continue fetching the rest.")
 
 
if __name__ == "__main__":
    main()