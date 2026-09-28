"""
FastAPI routes describing the trained model itself (not its predictions).
"""

import json

import numpy as np
from fastapi import APIRouter, HTTPException

from Scripts.prediction_engine import MODEL_PATH, load_model_bundle
from backend.schemas.model import FeatureImportance, FeatureImportanceResponse

router = APIRouter(prefix="/model", tags=["model"])

METADATA_PATH = MODEL_PATH.replace(".joblib", "_metadata.json")


def _trained_at() -> str | None:
    """Reads the training timestamp from the metadata file saved beside the model."""
    try:
        with open(METADATA_PATH) as f:
            return json.load(f).get("trained_at")
    except (OSError, ValueError):
        return None


@router.get(
    "/feature-importance",
    response_model=FeatureImportanceResponse,
    summary="Feature importance of the trained model",
)
def read_feature_importance() -> FeatureImportanceResponse:
    """
    Returns each input feature's share of the model's total importance.

    For tree boosters this uses *gain* (how much each feature's splits reduced
    the loss) rather than LightGBM's default split count -- split count
    over-rewards continuous features like lat/lon that simply offer many split
    points, while gain reflects what actually drove the predictions.
    """
    try:
        bundle = load_model_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Model file not available: {exc}") from exc

    model, features = bundle["model"], bundle["features"]

    if hasattr(model, "booster_"):
        raw = model.booster_.feature_importance(importance_type="gain")
        importance_type = "gain"
    elif hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
        importance_type = "impurity"
    elif hasattr(model, "coef_"):
        raw = np.abs(np.ravel(model.coef_))
        importance_type = "abs_coefficient"
    else:
        raise HTTPException(
            status_code=501,
            detail=f"Feature importance is not available for {type(model).__name__}.",
        )

    raw = np.asarray(raw, dtype=float)
    total = raw.sum()
    shares = raw / total * 100 if total > 0 else np.zeros_like(raw)

    ranked = sorted(zip(features, shares), key=lambda pair: pair[1], reverse=True)
    return FeatureImportanceResponse(
        model_name=bundle.get("model_name") or type(model).__name__,
        importance_type=importance_type,
        trained_at=_trained_at(),
        features=[FeatureImportance(feature=f, importance=round(float(v), 2)) for f, v in ranked],
    )
