
import os
from datetime import timedelta
 
import pandas as pd
 
# ---------- CONFIG ----------
FIRMS_CSV_PATH = "data/fire_archive_SV-C2_792465.csv"
WEATHER_CSV_PATH = "data/all_weather_data.csv"
PROGRESS_LOG_PATH = "data/fetch_progress.log"
REMAINING_CALLS_CSV_PATH = "data/remaining_calls.csv"
MISSING_ROWS_CSV_PATH = "data/missing_weather_rows.csv"
GRID_SIZE_DEGREES = 0.5   # must match fetch_weather_csv.py
CHUNK_DAYS = 14
 
 
def round_to_grid(value, grid_size):
    return round(round(value / grid_size) * grid_size, 3)
 
 
def build_date_chunks(dates, chunk_days):
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
    return f"{lat},{lon},{start_date},{end_date}"
 
 
# ---------- CHECK 1: CALL-LEVEL ----------
def check_call_level_completeness(firms_df):
    print("=" * 60)
    print("CHECK 1: API call-level completeness")
    print("=" * 60)
 
    locations = firms_df.groupby(["lat_round", "lon_round"])["acq_date_dt"].apply(list).reset_index()
 
    all_calls = []
    for _, row in locations.iterrows():
        chunks = build_date_chunks(row["acq_date_dt"], CHUNK_DAYS)
        for start_date, end_date in chunks:
            all_calls.append((row["lat_round"], row["lon_round"], start_date, end_date))
 
    total_expected = len(all_calls)
 
    if os.path.exists(PROGRESS_LOG_PATH):
        with open(PROGRESS_LOG_PATH, "r") as f:
            completed = set(line.strip() for line in f if line.strip())
    else:
        completed = set()
 
    remaining_calls = [
        c for c in all_calls if call_key(c[0], c[1], c[2], c[3]) not in completed
    ]
    completed_count = total_expected - len(remaining_calls)
 
    print(f"Total calls needed:     {total_expected}")
    print(f"Calls completed so far: {completed_count}")
    print(f"Calls remaining:        {len(remaining_calls)}")
 
    if remaining_calls:
        pct = completed_count / total_expected * 100 if total_expected else 0
        print(f">>> Fetch is {pct:.1f}% complete. Run fetch_weather_csv.py again to continue.")
 
        remaining_df = pd.DataFrame(
            remaining_calls, columns=["lat_round", "lon_round", "start_date", "end_date"]
        )
        remaining_df.to_csv(REMAINING_CALLS_CSV_PATH, index=False)
        print(f">>> Saved list of {len(remaining_calls)} remaining/failed calls to "
              f"{REMAINING_CALLS_CSV_PATH}")
    else:
        print(">>> Fetch is COMPLETE at the call level.")
        if os.path.exists(REMAINING_CALLS_CSV_PATH):
            os.remove(REMAINING_CALLS_CSV_PATH)
 
    return len(remaining_calls) == 0
 
 
# ---------- CHECK 2: ROW-LEVEL ----------
def check_row_level_completeness(firms_df, weather_df):
    print("\n" + "=" * 60)
    print("CHECK 2: Row-level weather match completeness")
    print("=" * 60)
 
    weather_df["datetime_utc"] = pd.to_datetime(weather_df["datetime_utc"])
    weather_df["w_date"] = weather_df["datetime_utc"].dt.strftime("%Y-%m-%d")
    weather_df["w_hour"] = weather_df["datetime_utc"].dt.hour
 
    weather_keys = set(
        zip(weather_df["lat_round"], weather_df["lon_round"], weather_df["w_date"], weather_df["w_hour"])
    )
 
    firms_df["acq_hour"] = firms_df["acq_time"].astype(str).str.zfill(4).str[:2].astype(int)
    firms_keys = list(
        zip(firms_df["lat_round"], firms_df["lon_round"], firms_df["acq_date"], firms_df["acq_hour"])
    )
 
    matched = sum(1 for k in firms_keys if k in weather_keys)
    total = len(firms_keys)
    missing = total - matched
 
    print(f"Total FIRMS records:      {total}")
    print(f"Matched with weather:     {matched}")
    print(f"Missing weather match:    {missing}")
    print(f"Match rate:               {matched/total*100:.2f}%")
 
    if missing > 0:
        # show a sample of missing (location, date) combos to help debug
        missing_keys = [k for k in firms_keys if k not in weather_keys]
        missing_df = pd.DataFrame(missing_keys, columns=["lat_round", "lon_round", "acq_date", "acq_hour"])
 
        missing_df.to_csv(MISSING_ROWS_CSV_PATH, index=False)
        print(f"\nSaved all {len(missing_df)} missing (location, date, hour) combos to "
              f"{MISSING_ROWS_CSV_PATH}")
 
        top_missing = (
            missing_df.groupby(["lat_round", "lon_round"])
            .size()
            .reset_index(name="missing_count")
            .sort_values("missing_count", ascending=False)
            .head(10)
        )
        print("\nTop 10 locations with the most missing weather matches:")
        print(top_missing.to_string(index=False))
    else:
        if os.path.exists(MISSING_ROWS_CSV_PATH):
            os.remove(MISSING_ROWS_CSV_PATH)
 
    return missing == 0
 
 
def main():
    print("Loading FIRMS data...")
    firms_df = pd.read_csv(FIRMS_CSV_PATH)
    firms_df["acq_date"] = pd.to_datetime(firms_df["acq_date"]).dt.strftime("%Y-%m-%d")
    firms_df["acq_date_dt"] = pd.to_datetime(firms_df["acq_date"])
    firms_df["lat_round"] = firms_df["latitude"].apply(lambda v: round_to_grid(v, GRID_SIZE_DEGREES))
    firms_df["lon_round"] = firms_df["longitude"].apply(lambda v: round_to_grid(v, GRID_SIZE_DEGREES))
    print(f"Loaded {len(firms_df)} FIRMS records.\n")
 
    call_level_complete = check_call_level_completeness(firms_df)
 
    if not os.path.exists(WEATHER_CSV_PATH):
        print(f"\n{WEATHER_CSV_PATH} not found yet — skipping row-level check.")
        return
 
    print("\nLoading weather data...")
    weather_df = pd.read_csv(WEATHER_CSV_PATH)
    print(f"Loaded {len(weather_df)} weather rows.")
 
    row_level_complete = check_row_level_completeness(firms_df, weather_df)
 
    print("\n" + "=" * 60)
    if call_level_complete and row_level_complete:
        print("READY: All weather data is fetched and matches your FIRMS data. "
              "Safe to proceed to join_firms_weather.py and preprocessing.")
    else:
        print("NOT READY: See above for what's missing.")
    print("=" * 60)
 
 
if __name__ == "__main__":
    main()
 