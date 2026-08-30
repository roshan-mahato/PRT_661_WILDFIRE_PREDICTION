import time
from datetime import datetime, timedelta
 
import pandas as pd
 
from fetch_openmeteo_historical_data import (
    fetch_weather_chunk,
    append_to_output_csv,
    mark_call_complete,
    call_key,
    REQUEST_DELAY_SECONDS,
    DAILY_CALL_LIMIT,
    OUTPUT_CSV_PATH,
)
 
REMAINING_CALLS_CSV_PATH = "data/remaining_calls.csv"
MIN_SPLIT_DAYS = 2   # don't split smaller than this — avoids infinite splitting on a truly broken range
 
 
def try_fetch_with_split_fallback(lat, lon, start_date, end_date, depth=0):
    """Try fetching one call. If it fails and the date range is still big
    enough to split, try two half-sized ranges instead — this helps when
    a large range times out but each half succeeds individually.
    Returns a list of (start_date, end_date) sub-ranges that STILL failed
    after all split attempts (empty list if everything succeeded)."""
    key = call_key(lat, lon, start_date, end_date)
    try:
        chunk_df = fetch_weather_chunk(lat, lon, start_date, end_date)
        append_to_output_csv(chunk_df)
        mark_call_complete(key)
        print(f"{'  ' * depth}OK: ({lat}, {lon}, {start_date} to {end_date}) "
              f"-> {len(chunk_df)} rows")
        return []
    except Exception as e:
        print(f"{'  ' * depth}FAILED: ({lat}, {lon}, {start_date}-{end_date}): {e}")
 
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        total_days = (end - start).days + 1
 
        if total_days <= MIN_SPLIT_DAYS:
            print(f"{'  ' * depth}Range too small to split further ({total_days} day(s)). Giving up on this one.")
            return [(start_date, end_date)]
 
        mid = start + timedelta(days=total_days // 2)
        first_half = (start.strftime("%Y-%m-%d"), (mid - timedelta(days=1)).strftime("%Y-%m-%d"))
        second_half = (mid.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
 
        print(f"{'  ' * depth}Splitting into {first_half} and {second_half}, retrying each...")
 
        still_failed = []
        still_failed += try_fetch_with_split_fallback(lat, lon, first_half[0], first_half[1], depth + 1)
        time.sleep(REQUEST_DELAY_SECONDS)
        still_failed += try_fetch_with_split_fallback(lat, lon, second_half[0], second_half[1], depth + 1)
        return still_failed
 
 
def main():
    print(f"Loading list of remaining/failed calls from {REMAINING_CALLS_CSV_PATH}...")
    calls_df = pd.read_csv(REMAINING_CALLS_CSV_PATH)
    print(f"Found {len(calls_df)} calls to retry.")
 
    if calls_df.empty:
        print("Nothing to retry.")
        return
 
    calls_this_run = 0
    still_failed = []
 
    for _, row in calls_df.iterrows():
        lat, lon, start_date, end_date = row["lat_round"], row["lon_round"], row["start_date"], row["end_date"]
 
        if calls_this_run >= DAILY_CALL_LIMIT:
            print(f"\nReached today's call budget ({DAILY_CALL_LIMIT}). "
                  f"Run this script again to continue retrying.")
            break
 
        failed_subranges = try_fetch_with_split_fallback(lat, lon, start_date, end_date)
        calls_this_run += 1
 
        for sub_start, sub_end in failed_subranges:
            still_failed.append({"lat_round": lat, "lon_round": lon, "start_date": sub_start, "end_date": sub_end})
 
        time.sleep(REQUEST_DELAY_SECONDS)
 
    succeeded = len(calls_df) - len(still_failed)
    print(f"\n{succeeded}/{len(calls_df)} original call(s) fully resolved this run "
          f"(counting split sub-ranges as part of the same original call).")
 
    if still_failed:
        still_failed_df = pd.DataFrame(still_failed)
        still_failed_df.to_csv(REMAINING_CALLS_CSV_PATH, index=False)
        print(f"{len(still_failed)} sub-range(s) still failed — saved to {REMAINING_CALLS_CSV_PATH} for another retry.")
    else:
        print("All listed calls succeeded. Run check_weather_completeness.py "
              "to confirm everything is now complete.")
 
 
if __name__ == "__main__":
    main()