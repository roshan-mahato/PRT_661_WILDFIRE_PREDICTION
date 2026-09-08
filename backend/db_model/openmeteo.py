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
    
    # Geographic coordinates
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    
    # Temporal information
    datetime_utc = Column(TIMESTAMP, nullable=False, index=True)
    fetched_at = Column(TIMESTAMP, default=datetime.now, nullable=False)
    
    # Temperature measurements
    temperature_2m = Column(Float, nullable=False)
    
    # Humidity
    relative_humidity_2m = Column(Float, nullable=False)
    
    # Wind conditions
    wind_speed_10m = Column(Float, nullable=False)
    wind_direction_10m = Column(Float, nullable=False)
    wind_gusts_10m = Column(Float, nullable=False)
    
    # Precipitation
    precipitation = Column(Float, nullable=False)
    
    # Soil conditions
    soil_moisture_0_to_7cm = Column(Float, nullable=False)
    
    # Atmospheric parameters
    cape = Column(Float, nullable=False)  # Convective Available Potential Energy
    vapour_pressure_deficit = Column(Float, nullable=False)  # VPD for plant transpiration
    et0_fao_evapotranspiration = Column(Float, nullable=False)  # Reference evapotranspiration

    # Composite indexes for efficient spatial-temporal queries
    __table_args__ = (
        Index('idx_om_location', 'latitude', 'longitude'),
        Index('idx_om_datetime_utc', 'datetime_utc'),
        Index('idx_om_location_datetime', 'latitude', 'longitude', 'datetime_utc'),
        Index('idx_om_fetched_at', 'fetched_at'),
        UniqueConstraint('latitude', 'longitude', 'datetime_utc', name='uq_om_weather_observation'),
    )
