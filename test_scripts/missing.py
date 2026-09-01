"""
diagnose_missing_coords.py

Purpose:
Quick check to see how many FIRMS rows have missing/invalid latitude or
longitude, since that would explain widespread weather-matching failures
(a row with no coordinates can never match a weather grid cell).

Before running:
    pip install pandas
"""

import pandas as pd

FIRMS_CSV_PATH = "data/firms_combined.csv"   # <-- change to whichever file you're checking

df = pd.read_csv(FIRMS_CSV_PATH)
print(f"Total rows: {len(df)}")

missing_lat = df["latitude"].isna().sum()
missing_lon = df["longitude"].isna().sum()
missing_either = df[["latitude", "longitude"]].isna().any(axis=1).sum()

print(f"Missing latitude:  {missing_lat}")
print(f"Missing longitude: {missing_lon}")
print(f"Missing either:    {missing_either} ({missing_either/len(df)*100:.2f}%)")

# Also check for coordinates that are technically present but outside Australia
# (could indicate bad/corrupted values rather than truly missing ones)
valid = df.dropna(subset=["latitude", "longitude"])
out_of_bounds = valid[
    ~valid["latitude"].between(-44, -10) | ~valid["longitude"].between(112, 154)
]
print(f"Present but outside Australia bounding box: {len(out_of_bounds)}")

if missing_either > 0:
    print("\nSample of rows with missing coordinates:")
    print(df[df[["latitude", "longitude"]].isna().any(axis=1)].head(10))

    if "source" in df.columns:
        print("\nMissing coordinates by source file:")
        print(df[df[["latitude", "longitude"]].isna().any(axis=1)]["source"].value_counts())