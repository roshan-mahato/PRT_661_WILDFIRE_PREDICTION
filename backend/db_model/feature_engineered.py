from sqlalchemy import (
    TIMESTAMP, Boolean, Column, Date, Float, ForeignKey, Index, Integer, String,
)

from backend.db_model.base import Base


class FeatureEngineered(Base):
    """
    Model features for one weather_daily row (1:1, keyed by weather_id).

    KBDI, Drought Factor and days_since_rain come from a recursion over the
    previous days, so this row also holds everything needed to continue it
    the next day -- the recursion state is simply the cell's latest archive
    feature row. That replaces Scripts/fire_state.json.

    cell_id, date and source are copies of the weather row's values, kept so
    the latest row per cell can be found without a join.
    """

    __tablename__ = "feature_engineered"

    weather_id = Column(
        Integer, ForeignKey("weather_daily.weather_id"), primary_key=True
    )
    cell_id = Column(Integer, ForeignKey("grid_cell.cell_id"), nullable=False)
    date = Column(Date, nullable=False)
    source = Column(String, nullable=False)

    # Recursion features
    kbdi = Column(Float, nullable=False)
    drought_factor = Column(Float, nullable=False)
    days_since_rain = Column(Float, nullable=False)
    days_since_cell_start = Column(Integer, nullable=False)
    kbdi_spinup_flag = Column(Boolean, nullable=False)

    # Rain-event state, only needed to continue the recursion
    last_rain_amt = Column(Float, nullable=False)
    in_rain_event = Column(Boolean, nullable=False)
    event_cumulative_rain = Column(Float, nullable=False)

    # Same-day features
    emc = Column(Float, nullable=False)
    ffdi = Column(Float, nullable=False)
    rate_of_spread = Column(Float, nullable=False)
    # FFDI at the day's peak fire-weather hour (>= ffdi, which uses daily
    # means). NULL while the weather row has no peak columns.
    ffdi_max = Column(Float, nullable=True)
    month_sin = Column(Float, nullable=False)
    month_cos = Column(Float, nullable=False)

    computed_at = Column(TIMESTAMP, nullable=False)

    __table_args__ = (
        Index("idx_feature_source_cell_date", "source", "cell_id", "date"),
    )
