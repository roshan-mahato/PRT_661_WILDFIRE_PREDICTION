"""
Pydantic response schemas for the live fire-risk prediction endpoints.

These define the JSON contract the Streamlit frontend (and any other client)
can rely on, independent of the column names the pandas pipeline happens to
use internally.
"""

from datetime import date as date_type
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class FireRiskPrediction(BaseModel):
    """A fire-risk prediction for one grid cell on one day."""

    lat_round: float = Field(..., description="Grid cell latitude (0.5-degree grid)")
    lon_round: float = Field(..., description="Grid cell longitude (0.5-degree grid)")
    acq_date: date_type = Field(..., description="Date the prediction applies to (UTC)")

    # Input weather (echoed back so the dashboard can show why the risk is what it is)
    temperature_2m: float = Field(..., description="Daily mean temperature (degrees C)")
    relative_humidity_2m: float = Field(..., description="Daily mean relative humidity (%)")
    wind_speed_10m: float = Field(..., description="Daily mean wind speed (km/h)")

    # Fire-danger indices
    ffdi: float = Field(..., description="McArthur Mark 5 Forest Fire Danger Index")
    kbdi: float = Field(..., description="Keetch-Byram Drought Index (0-203.2)")
    drought_factor: float = Field(..., description="Drought Factor (0-10)")
    kbdi_spinup_flag: bool = Field(
        ...,
        description=(
            "True if this grid cell has under 30 days of monitoring history, "
            "meaning KBDI/Drought Factor/FFDI are cold-start estimates and "
            "should be treated as lower-confidence."
        ),
    )

    # Model output
    fire_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted fire probability")
    fire_predicted: int = Field(..., description="1 if probability is above the model threshold")
    risk_level: Literal["Low", "Moderate", "High", "Extreme"] = Field(
        ..., description="Probability banded into a human-readable risk level"
    )


class FireRiskResponse(BaseModel):
    """Envelope returned by the prediction endpoints."""

    count: int = Field(..., description="Number of (grid cell, day) predictions returned")
    hours_back: int = Field(..., description="Lookback window used to pull weather data")
    spinup_count: int = Field(
        ..., description="How many returned predictions are cold-start (kbdi_spinup_flag=True)"
    )
    predictions: List[FireRiskPrediction]


class HealthResponse(BaseModel):
    """Service health / readiness."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    database_reachable: bool
    detail: Optional[str] = None
