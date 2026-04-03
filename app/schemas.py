"""
app/schemas.py — Pydantic models defining the API request/response contract.

WHY Pydantic schemas:
FastAPI uses these models to automatically validate incoming data, serialize
outgoing data, and generate the OpenAPI (Swagger) docs at /docs.
Without schemas, we'd be passing raw dicts with no type safety or documentation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Top3Prediction(BaseModel):
    """A single disease prediction with its confidence score."""
    disease:    str   = Field(..., description="Predicted plant disease class name.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence (0–1).")


class PredictionResponse(BaseModel):
    """
    Full API response for a /predict request.

    WHY Optional fields: the response shape differs between success and error states.
    When success=False, disease/confidence/top_3 are None and error explains why.
    When success=True, error is None.
    """
    success:    bool                    = Field(..., description="Whether a valid prediction was made.")
    disease:    Optional[str]           = Field(None, description="Top predicted disease class.")
    confidence: Optional[float]         = Field(None, description="Confidence of top prediction (0–1).")
    top_3:      Optional[List[Top3Prediction]] = Field(None, description="Top 3 predictions ranked by confidence.")
    error:      Optional[str]           = Field(None, description="Error message if success=False.")


class HealthResponse(BaseModel):
    """Response for the GET /health endpoint."""
    status: str = Field(..., description="Service status string, e.g. 'ok'.")
    app:    str = Field(..., description="Application name.")
