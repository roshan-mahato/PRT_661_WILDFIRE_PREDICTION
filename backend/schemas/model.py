"""
Pydantic response schemas for the model-information endpoints.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class FeatureImportance(BaseModel):
    """How much one input feature contributes to the model's decisions."""

    feature: str = Field(..., description="Feature name as used by the model")
    importance: float = Field(
        ..., ge=0.0, le=100.0, description="Share of total importance (%), all features sum to 100"
    )


class FeatureImportanceResponse(BaseModel):
    """Envelope returned by GET /model/feature-importance."""

    model_name: str = Field(..., description="Model family, e.g. LightGBM")
    importance_type: str = Field(
        ..., description="How importance was measured (e.g. 'gain' = total loss reduction)"
    )
    trained_at: Optional[str] = Field(None, description="When the model was trained, if recorded")
    features: List[FeatureImportance] = Field(..., description="Sorted highest importance first")
