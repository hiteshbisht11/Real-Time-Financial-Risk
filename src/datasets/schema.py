"""
Transaction schema definitions.

WHY: Every component in the system needs to agree on what a "transaction" looks like.
     Pydantic gives us runtime validation + type hints + JSON serialization for free.
     This is the contract between producers (Kafka) and consumers (inference, monitoring).
"""

from enum import Enum
from pydantic import BaseModel, Field, model_validator
from typing import Optional


class TransactionType(str, Enum):
    PAYMENT = "PAYMENT"
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    CASH_IN = "CASH_IN"
    DEBIT = "DEBIT"


class RiskDecision(str, Enum):
    APPROVE = "APPROVE"      # risk_score < 0.3
    REVIEW = "REVIEW"        # 0.3 <= risk_score < 0.7
    BLOCK = "BLOCK"          # risk_score >= 0.7


class TransactionEvent(BaseModel):
    """
    Raw transaction event — exactly as it arrives from the stream.
    Mirrors the PaySim CSV schema so our adapter is a thin wrapper.
    """
    step: int = Field(..., description="Hour of simulation (1-743). Proxy for timestamp.")
    type: TransactionType
    amount: float = Field(..., gt=0)
    name_orig: str = Field(..., alias="nameOrig")
    old_balance_orig: float = Field(..., alias="oldbalanceOrg", ge=0)
    new_balance_orig: float = Field(..., alias="newbalanceOrig", ge=0)
    name_dest: str = Field(..., alias="nameDest")
    old_balance_dest: float = Field(..., alias="oldbalanceDest", ge=0)
    new_balance_dest: float = Field(..., alias="newbalanceDest", ge=0)
    is_fraud: Optional[int] = Field(None, alias="isFraud")       # label, not available in prod
    is_flagged_fraud: Optional[int] = Field(None, alias="isFlaggedFraud")

    model_config = {"populate_by_name": True}


class EnrichedTransaction(BaseModel):
    """
    Transaction AFTER feature engineering — what the model actually sees.
    Separating raw vs enriched lets us version features independently of the schema.
    """
    # Identity (not used as model features but needed for logging/tracing)
    step: int
    name_orig: str
    name_dest: str
    type: TransactionType
    amount: float

    # --- Engineered features ---
    # Balance error: how much the balance books don't add up.
    # Fraudulent transactions often have systematic balance discrepancies.
    error_balance_orig: float   # (oldBal - newBal) - amount  ->  >0 means money disappeared
    error_balance_dest: float   # (newBal - oldBal) - amount  ->  >0 means money appeared from nowhere

    # One-hot encoded transaction types (5 types -> 4 dummy vars; PAYMENT is baseline)
    type_TRANSFER: int
    type_CASH_OUT: int
    type_CASH_IN: int
    type_DEBIT: int

    # Velocity features (computed from Redis / in-memory counter)
    # These capture "how much did this account transact recently?"
    orig_tx_count_1h: int = 0
    orig_amount_sum_1h: float = 0.0
    dest_tx_count_1h: int = 0

    # Label (None in prod, present during training/monitoring)
    is_fraud: Optional[int] = None


class RiskScore(BaseModel):
    """
    Output of the inference service — the full risk assessment for one transaction.
    Everything needed to make a downstream decision AND explain it.
    """
    step: int
    name_orig: str
    name_dest: str
    amount: float
    type: TransactionType

    risk_score: float = Field(..., ge=0.0, le=1.0,
                              description="P(fraud) from the model, 0=safe, 1=fraud")
    decision: RiskDecision

    # SHAP-based top contributing features
    top_features: dict[str, float] = Field(
        default_factory=dict,
        description="Feature name -> SHAP value (positive = pushed score toward fraud)"
    )

    # Groundtruth when available (used in monitoring/offline eval)
    is_fraud: Optional[int] = None

    @model_validator(mode="after")
    def decision_matches_score(self) -> "RiskScore":
        """Guard: decision must be consistent with the numeric score."""
        s = self.risk_score
        expected = (
            RiskDecision.BLOCK if s >= 0.7 else
            RiskDecision.REVIEW if s >= 0.3 else
            RiskDecision.APPROVE
        )
        if self.decision != expected:
            raise ValueError(
                f"decision={self.decision} is inconsistent with risk_score={s:.3f} "
                f"(expected {expected})"
            )
        return self
