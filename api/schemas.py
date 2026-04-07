"""
FastAPI request/response schemas.

WHY SEPARATE FROM src/datasets/schema.py?
  - src/datasets/schema.py defines the INTERNAL data contract (used everywhere)
  - api/schemas.py defines the EXTERNAL API contract (what clients send/receive)
  This separation lets us evolve the API without touching internal code.

API VERSIONING STRATEGY:
  /v1/score — current stable version
  /v2/score — future version (can run both simultaneously during migration)
"""

from typing import Optional
from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    """What the API client sends. Mirrors the PaySim/real-world transaction fields."""
    step: int = Field(..., example=501, description="Time step (hour of simulation)")
    type: str = Field(..., example="TRANSFER", description="PAYMENT|TRANSFER|CASH_OUT|CASH_IN|DEBIT")
    amount: float = Field(..., gt=0, example=181.00)
    nameOrig: str = Field(..., example="C1231006815")
    oldbalanceOrg: float = Field(..., ge=0, example=181.00)
    newbalanceOrig: float = Field(..., ge=0, example=0.00)
    nameDest: str = Field(..., example="C1666544295")
    oldbalanceDest: float = Field(..., ge=0, example=0.00)
    newbalanceDest: float = Field(..., ge=0, example=0.00)

    model_config = {
        "json_schema_extra": {
            "example": {
                "step": 501,
                "type": "TRANSFER",
                "amount": 181.00,
                "nameOrig": "C1231006815",
                "oldbalanceOrg": 181.00,
                "newbalanceOrig": 0.00,
                "nameDest": "C1666544295",
                "oldbalanceDest": 0.00,
                "newbalanceDest": 0.00,
            }
        }
    }


class ScoreResponse(BaseModel):
    """What the API returns to the client."""
    step: int
    name_orig: str
    name_dest: str
    amount: float
    type: str

    risk_score: float = Field(..., description="P(fraud), range [0.0, 1.0]")
    decision: str = Field(..., description="APPROVE | REVIEW | BLOCK")

    top_features: dict[str, float] = Field(
        default_factory=dict,
        description="Top SHAP feature contributions. Positive = toward fraud."
    )

    latency_ms: Optional[float] = Field(None, description="Server-side inference time in ms")


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
