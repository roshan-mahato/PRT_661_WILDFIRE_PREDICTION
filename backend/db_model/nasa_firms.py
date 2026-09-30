from datetime import datetime

from sqlalchemy import (
    TIMESTAMP,
    Column,
    Date,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
)

from backend.db_model.base import Base


class NASAFirms(Base):
    """
    One VIIRS active-fire detection from NASA FIRMS (S-NPP, NOAA-20, NOAA-21),
    loaded by Scripts/combine_firms.py.

    Detections are stored as delivered (every confidence and type); the
    labelling step decides which ones count as a fire. `source` says whether a
    row came from the standard archive or the near-real-time (NRT) feed:
    archive rows supersede NRT rows for the same satellite and days.
    """

    __tablename__ = "nasa_firms"

    id = Column(Integer, primary_key=True)

    # Detection position, and the 0.5-degree grid cell it falls in
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    lat_round = Column(Float, nullable=False)
    lon_round = Column(Float, nullable=False)

    # Brightness temperatures (K): VIIRS I4 (3.7 um) and I5 (11 um) channels.
    # The FIRMS download calls them `brightness` and `bright_t31`.
    bright_ti4 = Column(Float, nullable=False)
    bright_ti5 = Column(Float, nullable=False)

    # Pixel size along scan / track (km)
    scan = Column(Float, nullable=False)
    track = Column(Float, nullable=False)

    # Overpass: UTC date and time (HHMM, e.g. 414 = 04:14)
    acq_date = Column(Date, nullable=False)
    acq_time = Column(Integer, nullable=False)

    satellite = Column(String(10), nullable=False)    # SNPP, N20, N21
    instrument = Column(String(10), nullable=False)   # VIIRS
    confidence = Column(String(1), nullable=False)    # l(ow), n(ominal), h(igh)
    version = Column(String(20), nullable=False)      # "2" archive, "2.0NRT" near-real-time
    frp = Column(Float, nullable=False)               # Fire Radiative Power (MW)
    daynight = Column(String(1), nullable=False)      # D or N

    # 0 vegetation fire, 1 active volcano, 2 other static land source,
    # 3 offshore. Only the archive classifies detections; NULL on NRT rows.
    type = Column(Integer, nullable=True)

    source = Column(String(10), nullable=False)       # archive or nrt
    created_at = Column(TIMESTAMP, default=datetime.now, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "latitude", "longitude", "acq_date", "acq_time", "satellite",
            name="uq_fire_detection",
        ),
        Index("idx_firms_cell_date", "lat_round", "lon_round", "acq_date"),
        Index("idx_firms_satellite_date", "satellite", "acq_date", "source"),
    )
