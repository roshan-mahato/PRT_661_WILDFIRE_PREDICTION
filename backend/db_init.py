import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from backend.database import engine
from backend.db_model.base import Base
from backend.db_model.nasa_firms import NASAFirms  # noqa: F401 -- registers the table
from backend.db_model.grid_cell import GridCell  # noqa: F401
from backend.db_model.fetch_run import FetchRun  # noqa: F401
from backend.db_model.weather_daily import WeatherDaily  # noqa: F401
from backend.db_model.feature_engineered import FeatureEngineered  # noqa: F401
from backend.db_model.prediction import Prediction  # noqa: F401

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAND_MASK_CSV = os.path.join(REPO_ROOT, "data", "full_grid_land_mask.csv")

# Tables of the old design: hourly live weather purged after 7 days, and
# predictions that copied their inputs. Replaced by weather_daily /
# feature_engineered / prediction.
LEGACY_TABLES = ["openmeteo", "fire_prediction"]


def create_tables():
    """
    Creates any missing table, and adds columns that were added to a model
    after its table was created (nullable ones only, so existing rows stay
    valid). Existing columns and rows are left alone.
    """
    Base.metadata.create_all(bind=engine)
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            have = {r[1] for r in conn.exec_driver_sql(f"PRAGMA table_info({table.name})")}
            for col in table.columns:
                if col.name not in have and col.nullable:
                    ddl = col.type.compile(dialect=engine.dialect)
                    conn.exec_driver_sql(f"ALTER TABLE {table.name} ADD COLUMN {col.name} {ddl}")
                    print(f"Added column {table.name}.{col.name}")


def init_db(drop_legacy: bool = False):
    print("Initializing database...")
    create_tables()

    if os.path.exists(LAND_MASK_CSV):
        from backend.api.weather import seed_grid_cells
        m = pd.read_csv(LAND_MASK_CSV)
        n = seed_grid_cells({(a, o): bool(land) for a, o, land
                             in zip(m["lat_round"], m["lon_round"], m["is_land"])})
        print(f"Grid cells: {n:,} loaded from {os.path.basename(LAND_MASK_CSV)}")

    if drop_legacy:
        with engine.begin() as conn:
            for t in LEGACY_TABLES:
                conn.exec_driver_sql(f"DROP TABLE IF EXISTS {t}")
        print(f"Dropped legacy tables: {', '.join(LEGACY_TABLES)}")

    print(f"Database initialized at {engine.url.database}")
    print(f"Tables: {', '.join(sorted(Base.metadata.tables.keys()))}")


if __name__ == "__main__":
    init_db(drop_legacy="--drop-legacy" in sys.argv)
