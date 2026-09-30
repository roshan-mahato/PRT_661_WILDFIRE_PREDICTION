from sqlalchemy import (
    TIMESTAMP, Boolean, Column, Date, Float, ForeignKey, Index, Integer, String,
    UniqueConstraint, text,
)

from backend.db_model.base import Base


class Prediction(Base):
    """
    Model output for one feature row and one model version.

    The inputs are not copied here: weather_id points at the feature row,
    which points at the weather row, and neither is ever changed -- so any
    prediction can still be explained and reproduced later.

    A cell-day is predicted again by every forecast run that covers it. The
    newest one has is_current = 1; the partial unique index makes SQLite
    guarantee there is only one. Older ones are kept to measure how accuracy
    changes with lead_days (days between the forecast and the day predicted).
    """

    __tablename__ = "prediction"

    prediction_id = Column(Integer, primary_key=True)
    weather_id = Column(
        Integer, ForeignKey("feature_engineered.weather_id"), nullable=False
    )
    run_id = Column(Integer, ForeignKey("fetch_run.run_id"), nullable=False)
    cell_id = Column(Integer, ForeignKey("grid_cell.cell_id"), nullable=False)
    date = Column(Date, nullable=False)
    lead_days = Column(Integer, nullable=False)

    fire_probability = Column(Float, nullable=False)
    fire_predicted = Column(Integer, nullable=False)
    risk_level = Column(String, nullable=False)
    threshold = Column(Float, nullable=False)

    model_version = Column(String, nullable=False)
    predicted_at = Column(TIMESTAMP, nullable=False)
    is_current = Column(Boolean, nullable=False, default=False)

    __table_args__ = (
        UniqueConstraint("weather_id", "model_version", name="uq_prediction_weather_model"),
        Index("ux_prediction_current", "date", "cell_id", unique=True,
              sqlite_where=text("is_current = 1")),
        Index("idx_prediction_run", "run_id"),
        Index("idx_prediction_cell_date", "cell_id", "date"),
    )
