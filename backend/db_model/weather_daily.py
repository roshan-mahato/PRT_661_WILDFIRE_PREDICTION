from sqlalchemy import (
    CheckConstraint, Column, Date, Float, ForeignKey, Index, Integer, String,
    UniqueConstraint, text,
)

from backend.db_model.base import Base


class WeatherDaily(Base):
    """
    Daily weather, history and live, one row per (grid cell, day, fetch run).

    Rows are append-only: a row is never updated or deleted-and-reinserted,
    so a feature or prediction built from it can always be traced back to the
    exact weather it used.

    Aggregation matches training: precipitation is the SUM of the UTC day,
    every other variable the MEAN. The peak columns hold the afternoon fire
    weather the means hide. They may be filled in later on an archive row that
    was stored without them -- that adds values, it never changes one.
    """

    __tablename__ = "weather_daily"

    weather_id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("fetch_run.run_id"), nullable=False)
    cell_id = Column(Integer, ForeignKey("grid_cell.cell_id"), nullable=False)
    date = Column(Date, nullable=False)

    # Copied from fetch_run so the archive series can be indexed on its own.
    source = Column(String, nullable=False)
    # Set when the history fetch had to fill the day: 'forecast' (from the
    # forecast API) or 'interpolated'. NULL for a fully observed day.
    filled_from = Column(String, nullable=True)

    temperature_2m = Column(Float, nullable=False)
    relative_humidity_2m = Column(Float, nullable=False)
    wind_speed_10m = Column(Float, nullable=False)
    wind_direction_10m = Column(Float, nullable=False)
    precipitation = Column(Float, nullable=False)
    wind_gusts_10m = Column(Float, nullable=False)
    soil_moisture_0_to_7cm = Column(Float, nullable=False)
    vapour_pressure_deficit = Column(Float, nullable=False)
    et0_fao_evapotranspiration = Column(Float, nullable=False)

    # Daily peaks of the hourly series (Scripts/prediction_engine.py,
    # DAILY_PEAK_COLS). NULL on rows stored before these existed.
    temperature_2m_max = Column(Float, nullable=True)
    relative_humidity_2m_min = Column(Float, nullable=True)
    wind_speed_10m_max = Column(Float, nullable=True)
    wind_gusts_10m_max = Column(Float, nullable=True)
    vapour_pressure_deficit_max = Column(Float, nullable=True)
    et0_fao_evapotranspiration_sum = Column(Float, nullable=True)   # mm/day
    # Conditions at the hour with the highest FFDI
    peak_fire_hour_utc = Column(Integer, nullable=True)
    peak_fire_temperature_2m = Column(Float, nullable=True)
    peak_fire_relative_humidity_2m = Column(Float, nullable=True)
    peak_fire_wind_speed_10m = Column(Float, nullable=True)
    wind_direction_at_max_wind = Column(Float, nullable=True)

    __table_args__ = (
        UniqueConstraint("cell_id", "date", "run_id", name="uq_weather_cell_date_run"),
        # Exactly one observed row per cell-day: the archive is one series.
        Index("ux_weather_archive_cell_date", "cell_id", "date", unique=True,
              sqlite_where=text("source = 'archive'")),
        Index("idx_weather_run", "run_id"),
        CheckConstraint("source IN ('archive', 'forecast')", name="ck_weather_source"),
    )
