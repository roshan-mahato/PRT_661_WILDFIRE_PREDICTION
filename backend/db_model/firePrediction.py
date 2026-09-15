from sqlalchemy import (
    Boolean, Column, Date, Float, Index, Integer, String, TIMESTAMP, UniqueConstraint
)
from datetime import datetime

from backend.db_model.base import Base


class FirePrediction(Base):
    """
    Schema for generated fire-risk predictions.

    One row per (grid cell, forecast day). Predictions are stored rather than
    recomputed on every request for three reasons: the dashboard can load
    instantly instead of waiting on the KBDI recursion and model inference;
    there is an offline fallback if Open-Meteo or the model is unavailable;
    and keeping the history lets the team compare past predictions against
    what FIRMS actually detected.

    The weather values, fire-danger indices and recursion-derived features
    are all stored alongside the prediction, so a stored row is
    self-explaining -- you can see WHY a cell was rated Extreme without
    re-joining to the weather table, and you can reproduce the prediction
    exactly. That matters because the source data does not survive: the
    `weather_live` table is delete-then-insert on every fetch and purged
    after 7 days, and `fire_state.json` holds only current recursion state.
    """

    __tablename__ = "fire_prediction"

    id = Column(Integer, primary_key=True, index=True)

    # Grid cell and forecast day this prediction applies to
    lat_round = Column(Float, nullable=False, index=True)
    lon_round = Column(Float, nullable=False, index=True)
    acq_date = Column(Date, nullable=False, index=True)

    # Input weather (daily aggregates the prediction was built from).
    # The full set the model consumes is stored, not just the headline three,
    # so a stored row is a complete, reproducible training example: you can
    # re-run it through the model and get the same probability back, and later
    # compare predictions against what FIRMS actually detected with every
    # feature still available for error analysis.
    temperature_2m = Column(Float, nullable=False)
    relative_humidity_2m = Column(Float, nullable=False)
    wind_speed_10m = Column(Float, nullable=False)
    wind_gusts_10m = Column(Float, nullable=True)
    wind_direction_10m = Column(Float, nullable=True)
    precipitation = Column(Float, nullable=True)
    soil_moisture_0_to_7cm = Column(Float, nullable=True)
    vapour_pressure_deficit = Column(Float, nullable=True)
    et0_fao_evapotranspiration = Column(Float, nullable=True)

    # Fire-danger indices
    ffdi = Column(Float, nullable=False)
    kbdi = Column(Float, nullable=False)
    drought_factor = Column(Float, nullable=False)
    kbdi_spinup_flag = Column(Boolean, nullable=False, default=False)

    # Recursion-derived features. These live nowhere else -- fire_state.json
    # holds only current state, not history -- so if they are not stored here
    # they cannot be recovered for a past prediction.
    days_since_rain = Column(Float, nullable=True)
    days_since_cell_start = Column(Float, nullable=True)

    # Model output
    fire_probability = Column(Float, nullable=False)
    fire_predicted = Column(Integer, nullable=False)
    risk_level = Column(String, nullable=False, index=True)

    # Audit trail -- which run produced this row
    predicted_at = Column(TIMESTAMP, default=datetime.now, nullable=False, index=True)
    model_version = Column(String, nullable=True)

    __table_args__ = (
        Index("idx_fp_grid_cell", "lat_round", "lon_round"),
        Index("idx_fp_grid_cell_date", "lat_round", "lon_round", "acq_date"),
        Index("idx_fp_risk_level", "risk_level"),
        Index("idx_fp_predicted_at", "predicted_at"),
        # A cell-day has exactly one current prediction; a re-run supersedes
        # the previous one rather than adding a second row.
        UniqueConstraint("lat_round", "lon_round", "acq_date", name="uq_fp_cell_day"),
    )
