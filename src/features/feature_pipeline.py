"""
Feature Engineering Pipeline.

TWO MODES:
  1. Batch mode  — transforms a pandas DataFrame for training/eval
  2. Online mode — transforms a single TransactionEvent for real-time inference

WHY TWO MODES?
  Training uses batch pandas operations (fast on 6M rows).
  Inference receives one event at a time and must be <10ms.
  Both modes produce IDENTICAL feature vectors — this is the "training-serving skew" problem.
  Keeping them in one file guarantees they stay in sync.

FEATURE GROUPS:
  A. Balance Error Features   — the most predictive signals (from EDA)
  B. Transaction Type Dummies — LightGBM handles categoricals natively, but
                                 explicit dummies let us use Logistic Regression too
  C. Velocity Features        — count/sum of transactions in the last N steps
                                 (requires Redis in production; in-memory dict in training)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.datasets.schema import EnrichedTransaction, TransactionEvent

logger = logging.getLogger(__name__)

# The exact feature columns the model expects — ORDER MATTERS for numpy arrays
MODEL_FEATURE_COLS = [
    "amount",
    "error_balance_orig",
    "error_balance_dest",
    "type_TRANSFER",
    "type_CASH_OUT",
    "type_CASH_IN",
    "type_DEBIT",
    "orig_tx_count_1h",
    "orig_amount_sum_1h",
    "dest_tx_count_1h",
]


# ---------------------------------------------------------------------------
# Batch mode (training / offline evaluation)
# ---------------------------------------------------------------------------

def engineer_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw PaySim DataFrame into model-ready feature matrix.

    Parameters
    ----------
    df : raw DataFrame from PaySimAdapter.batch_load()

    Returns
    -------
    DataFrame with MODEL_FEATURE_COLS + 'isFraud' column

    TEACHING NOTES:
      errorBalanceOrig = (oldbalanceOrg - newbalanceOrig) - amount
        When this is 0: the books balance perfectly (normal transaction).
        When this is non-zero: money appeared/disappeared — strong fraud signal.
        Why? Fraudsters often drain accounts entirely, leaving 0 in destination,
        which creates a systematic imbalance.

      errorBalanceDest = (newbalanceDest - oldbalanceDest) - amount
        If I send you $100, your balance should increase by exactly $100.
        If it didn't, something fishy happened.

      Velocity features:
        "How many times did this account transact in the last hour?"
        A legitimate user rarely sends 50 transfers in one hour.
        Fraudsters automate transactions → velocity spikes.
        In batch mode we approximate this with a rolling window on step.
    """
    out = df.copy()

    # A. Balance Error Features
    out["error_balance_orig"] = (
        (out["oldbalanceOrg"] - out["newbalanceOrig"]) - out["amount"]
    )
    out["error_balance_dest"] = (
        (out["newbalanceDest"] - out["oldbalanceDest"]) - out["amount"]
    )

    # B. Transaction Type One-Hot Dummies
    # PAYMENT is the baseline (all zeros = PAYMENT)
    for t in ["TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]:
        out[f"type_{t}"] = (out["type"] == t).astype(int)

    # C. Velocity Features — approximate with per-step count (batch proxy)
    # In production, Redis stores exact rolling 1-hour counts per account.
    # Here, we count transactions per account per step (1-hour bucket).
    step_orig_counts = (
        out.groupby(["step", "nameOrig"]).cumcount()
    )
    step_orig_amounts = (
        out.groupby(["step", "nameOrig"])["amount"].cumsum() - out["amount"]
    )
    step_dest_counts = (
        out.groupby(["step", "nameDest"]).cumcount()
    )

    out["orig_tx_count_1h"] = step_orig_counts
    out["orig_amount_sum_1h"] = step_orig_amounts
    out["dest_tx_count_1h"] = step_dest_counts

    logger.info(
        f"Engineered features for {len(out):,} rows. "
        f"Fraud rate: {out['isFraud'].mean()*100:.3f}%"
    )
    return out


def get_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (X, y) arrays from an engineered DataFrame.
    Always call engineer_features_batch() first.
    """
    X = df[MODEL_FEATURE_COLS].values.astype(np.float32)
    y = df["isFraud"].values.astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Online mode (real-time inference)
# ---------------------------------------------------------------------------

class OnlineFeaturePipeline:
    """
    Transforms a single TransactionEvent into model features in real-time.

    Maintains in-memory velocity counters keyed by account ID.
    In production, replace _velocity_store with a Redis client.

    Usage:
        pipeline = OnlineFeaturePipeline()
        enriched = pipeline.transform(event)
        features = pipeline.to_feature_vector(enriched)  # numpy array for model
    """

    def __init__(self, redis_client=None):
        # If no Redis provided, fall back to in-memory dict (good for testing)
        self._redis = redis_client
        # In-memory fallback: {account_id: {"count": int, "amount_sum": float, "step": int}}
        self._velocity_store: dict[str, dict] = {}

    def _get_velocity(self, account_id: str, current_step: int) -> tuple[int, float]:
        """
        Returns (tx_count, amount_sum) for the account in the current step window.

        Redis version (production):
            key = f"vel:{account_id}:{step}"
            INCR + EXPIREAT for automatic TTL cleanup.
        """
        if self._redis is not None:
            return self._get_velocity_redis(account_id, current_step)
        return self._get_velocity_memory(account_id, current_step)

    def _get_velocity_memory(self, account_id: str, current_step: int) -> tuple[int, float]:
        """In-memory fallback for development/testing."""
        v = self._velocity_store.get(account_id, {})
        if v.get("step") != current_step:
            return 0, 0.0
        return v.get("count", 0), v.get("amount_sum", 0.0)

    def _update_velocity(self, account_id: str, amount: float, current_step: int) -> None:
        """Increment velocity counters after a transaction is processed."""
        if self._redis is not None:
            self._update_velocity_redis(account_id, amount, current_step)
            return
        v = self._velocity_store.get(account_id, {})
        if v.get("step") != current_step:
            v = {"step": current_step, "count": 0, "amount_sum": 0.0}
        v["count"] += 1
        v["amount_sum"] += amount
        self._velocity_store[account_id] = v

    def _get_velocity_redis(self, account_id: str, current_step: int) -> tuple[int, float]:
        """
        Redis-backed velocity lookup.
        Keys expire after 2 steps (2 simulated hours) automatically.
        """
        count_key = f"vel:count:{account_id}:{current_step}"
        amount_key = f"vel:amount:{account_id}:{current_step}"
        count = int(self._redis.get(count_key) or 0)
        amount = float(self._redis.get(amount_key) or 0.0)
        return count, amount

    def _update_velocity_redis(self, account_id: str, amount: float, current_step: int) -> None:
        """Atomically update Redis velocity counters with TTL."""
        count_key = f"vel:count:{account_id}:{current_step}"
        amount_key = f"vel:amount:{account_id}:{current_step}"
        pipe = self._redis.pipeline()
        pipe.incr(count_key)
        pipe.incrbyfloat(amount_key, amount)
        pipe.expire(count_key, 7200)   # 2 hours TTL
        pipe.expire(amount_key, 7200)
        pipe.execute()

    def transform(self, event: TransactionEvent) -> EnrichedTransaction:
        """
        Main entry point: raw event → enriched transaction with all features.
        This is what the inference endpoint calls.
        """
        # Balance error features
        error_balance_orig = (event.old_balance_orig - event.new_balance_orig) - event.amount
        error_balance_dest = (event.new_balance_dest - event.old_balance_dest) - event.amount

        # Type dummies
        t = event.type.value
        type_TRANSFER = int(t == "TRANSFER")
        type_CASH_OUT = int(t == "CASH_OUT")
        type_CASH_IN = int(t == "CASH_IN")
        type_DEBIT = int(t == "DEBIT")

        # Velocity features
        orig_count, orig_amount_sum = self._get_velocity(event.name_orig, event.step)
        dest_count, _ = self._get_velocity(event.name_dest, event.step)

        # Update velocity AFTER reading (we don't count the current tx yet)
        self._update_velocity(event.name_orig, event.amount, event.step)
        self._update_velocity(event.name_dest, event.amount, event.step)

        return EnrichedTransaction(
            step=event.step,
            name_orig=event.name_orig,
            name_dest=event.name_dest,
            type=event.type,
            amount=event.amount,
            error_balance_orig=error_balance_orig,
            error_balance_dest=error_balance_dest,
            type_TRANSFER=type_TRANSFER,
            type_CASH_OUT=type_CASH_OUT,
            type_CASH_IN=type_CASH_IN,
            type_DEBIT=type_DEBIT,
            orig_tx_count_1h=orig_count,
            orig_amount_sum_1h=orig_amount_sum,
            dest_tx_count_1h=dest_count,
            is_fraud=event.is_fraud,
        )

    def to_feature_vector(self, enriched: EnrichedTransaction) -> np.ndarray:
        """
        Convert EnrichedTransaction → numpy array in MODEL_FEATURE_COLS order.
        This array is what gets passed directly to model.predict().
        """
        return np.array([
            enriched.amount,
            enriched.error_balance_orig,
            enriched.error_balance_dest,
            enriched.type_TRANSFER,
            enriched.type_CASH_OUT,
            enriched.type_CASH_IN,
            enriched.type_DEBIT,
            enriched.orig_tx_count_1h,
            enriched.orig_amount_sum_1h,
            enriched.dest_tx_count_1h,
        ], dtype=np.float32).reshape(1, -1)
