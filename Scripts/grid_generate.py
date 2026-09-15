import numpy as np
import pandas as pd

# Bounding box and grid size (from project conventions)
LAT_MIN, LAT_MAX = -44, -10
LON_MIN, LON_MAX = 112, 154
GRID_SIZE = 0.5

def round_to_grid(v, g=GRID_SIZE):
    """Double-round pattern used throughout the pipeline."""
    return round(round(v / g) * g, 3)

def generate_grid():
    lats = np.arange(LAT_MIN, LAT_MAX + GRID_SIZE, GRID_SIZE)
    lons = np.arange(LON_MIN, LON_MAX + GRID_SIZE, GRID_SIZE)

    lats = [round_to_grid(lat) for lat in lats]
    lons = [round_to_grid(lon) for lon in lons]

    grid = [(lat, lon) for lat in lats for lon in lons]

    df = pd.DataFrame(grid, columns=["lat_round", "lon_round"])
    return df

if __name__ == "__main__":
    df = generate_grid()
    print(f"Generated {len(df)} grid cells.")
    df.to_csv("data/australia_grid_points.csv", index=False)
    print("Saved to data/australia_grid_points.csv")