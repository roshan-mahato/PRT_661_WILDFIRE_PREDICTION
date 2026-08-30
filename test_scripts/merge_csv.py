import pandas as pd

ARCHIVE_CSV_PATH = "data/fire_archive_SV-C2_792465.csv"
NRT_CSV_PATH = "data/fire_nrt_SV-C2_792465.csv"
OUTPUT_CSV_PATH = "data/all_firms_dataset.csv"
 
TYPE_UNKNOWN_VALUE = -1   # sentinel for "not yet classified" (NRT rows)
 
 
def load_archive(path):
    df = pd.read_csv(path)
    df["source"] = "archive"
    return df
 
 
def load_nrt(path):
    df = pd.read_csv(path)
    df["source"] = "nrt"
    if "type" not in df.columns:
        df["type"] = TYPE_UNKNOWN_VALUE
    return df
 
 
def main():
    print("Loading archive data...")
    archive_df = load_archive(ARCHIVE_CSV_PATH)
    print(f"Loaded {len(archive_df)} archive rows "
          f"({archive_df['acq_date'].min()} to {archive_df['acq_date'].max()})")
 
    print("Loading NRT data...")
    nrt_df = load_nrt(NRT_CSV_PATH)
    print(f"Loaded {len(nrt_df)} NRT rows "
          f"({nrt_df['acq_date'].min()} to {nrt_df['acq_date'].max()})")
 
    # Sanity check: warn if the two files actually overlap in date range,
    # since that could mean duplicate detections of the same fire
    overlap = nrt_df["acq_date"].min() <= archive_df["acq_date"].max()
    if overlap:
        print(f"\nNOTE: date ranges overlap (archive ends {archive_df['acq_date'].max()}, "
              f"NRT starts {nrt_df['acq_date'].min()}). Rows in the overlap period may "
              f"include duplicate detections from both files.")
 
    combined_df = pd.concat([archive_df, nrt_df], ignore_index=True)
 
    # Drop exact duplicates that could arise from any overlap window
    before = len(combined_df)
    combined_df = combined_df.drop_duplicates(
        subset=["latitude", "longitude", "acq_date", "acq_time", "satellite"]
    )
    if len(combined_df) < before:
        print(f"Dropped {before - len(combined_df)} duplicate rows found in the overlap window.")
 
    combined_df = combined_df.sort_values(["acq_date", "acq_time"]).reset_index(drop=True)
 
    combined_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"\nSaved combined dataset to {OUTPUT_CSV_PATH} ({len(combined_df)} total rows).")
    print(f"Date range: {combined_df['acq_date'].min()} to {combined_df['acq_date'].max()}")
 
    unknown_type_count = (combined_df["type"] == TYPE_UNKNOWN_VALUE).sum()
    print(f"\n{unknown_type_count} rows have type = {TYPE_UNKNOWN_VALUE} (unclassified, from NRT).")
    print("IMPORTANT: update preprocess_data.py's vegetation-fire filter to treat "
          f"type == {TYPE_UNKNOWN_VALUE} as 'keep, unknown' rather than dropping it "
          "(it currently only keeps type == 0).")
 
    print(f"\nNext step: point fetch_weather_csv.py's FIRMS_CSV_PATH at {OUTPUT_CSV_PATH} "
          "and rerun it — it will automatically fetch weather only for the new NRT "
          "date range and append to your existing weather_data.csv.")
 
 
if __name__ == "__main__":
    main()