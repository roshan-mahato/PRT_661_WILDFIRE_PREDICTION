from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, UniqueConstraint, Index
from datetime import datetime

from backend.db_model.base import Base


class OpenMeteo(Base):
    """
    Schema for Open-Meteo weather data.
    
    Columns are based on the Open-Meteo API response for weather conditions
    and live weather grid data.
    """

    __tablename__ = 'openmeteo'

    id = Column(Integer, primary_key=True, index=True)
    
    # Grid cell coordinates (0.5-degree grid, see pipeline grid-rounding convention)
    lat_round = Column(Float, nullable=False, index=True)
    lon_round = Column(Float, nullable=False, index=True)
    
    # Temporal information
    datetime_utc = Column(TIMESTAMP, nullable=False, index=True)
    fetched_at = Column(TIMESTAMP, default=datetime.now, nullable=False)
    
    # Weather variables (matches HOURLY_VARS in fetch_openmeteo_live.py)
    temperature_2m = Column(Float, nullable=False)
    relative_humidity_2m = Column(Float, nullable=False)
    wind_speed_10m = Column(Float, nullable=False)
    wind_direction_10m = Column(Float, nullable=False)
    precipitation = Column(Float, nullable=False)
    wind_gusts_10m = Column(Float, nullable=False)
    soil_moisture_0_to_7cm = Column(Float, nullable=False)
    cape = Column(Float, nullable=False)
    vapour_pressure_deficit = Column(Float, nullable=False)
    et0_fao_evapotranspiration = Column(Float, nullable=False)
    
    __table_args__ = (
        Index("idx_wl_grid_cell", "lat_round", "lon_round"),
        Index("idx_wl_datetime_utc", "datetime_utc"),
        Index("idx_wl_grid_cell_datetime", "lat_round", "lon_round", "datetime_utc"),
        UniqueConstraint(
            "lat_round", "lon_round", "datetime_utc", name="uq_wl_weather_observation"
        ),
    )
