from sqlalchemy import Column, Integer, String, Float, TIMESTAMP, UniqueConstraint, Index
from datetime import datetime

from backend.db_model.base import Base


class NASAFirms(Base):
    """
    Schema for NASA FIRMS (Fire Information for Resource Management System) data.
    
    Columns are based on the NASA FIRMS DATASET, API Response.
    """

    __tablename__ = 'nasa_firms_historical'

    id = Column(Integer, primary_key=True, index=True)
    
    # Geographic coordinates
    latitude = Column(Float, nullable=False, index=True)
    longitude = Column(Float, nullable=False, index=True)
    
    # Thermal measurements (brightness temperature)
    bright_ti4 = Column(Float, nullable=False)
    bright_ti5 = Column(Float, nullable=False)
    
    # Satellite scan characteristics
    scan = Column(Float, nullable=False)
    track = Column(Float, nullable=False)
    
    # Fire detection timing
    acq_date = Column(String(50), nullable=False, index=True)
    acq_time = Column(Integer, nullable=False)
    
    # Satellite and sensor information
    satellite = Column(String(50), nullable=False, index=True)
    instrument = Column(String(50), nullable=False)
    
    # Detection quality indicators
    confidence = Column(String(50), nullable=False)
    version = Column(String(50), nullable=False)
    
    # Fire energy measurement
    frp = Column(Float, nullable=False)  # Fire Radiative Power
    
    # Observation context
    daynight = Column(String(10), nullable=False)

    created_at = Column(TIMESTAMP, default=datetime.now, nullable=False)

    # Composite indexes for efficient spatial-temporal queries
    __table_args__ = (
        Index('idx_location', 'latitude', 'longitude'),
        Index('idx_acq_datetime', 'acq_date', 'acq_time'),
        Index('idx_satellite_detection', 'satellite', 'instrument', 'acq_date'),
        UniqueConstraint('latitude', 'longitude', 'acq_date', 'acq_time', 'satellite', name='uq_fire_detection'),
    )