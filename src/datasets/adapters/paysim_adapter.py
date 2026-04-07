"""
PaySim Dataset Adapter — System + Streaming Focused.

WHY THIS EXISTS:
  The PaySim CSV is a static file. But our production system expects a stream of
  TransactionEvent objects. This adapter bridges that gap in two ways:
    1. batch_load()  — returns a pandas DataFrame for training (offline path)
    2. stream()      — yields TransactionEvents one-by-one, simulating Kafka messages

  Using an adapter pattern means the rest of the system never imports pandas or knows
  about CSV paths. If we swap PaySim for a real Kafka topic, nothing else changes.

TIME-ORDERED SPLIT STRATEGY:
  PaySim has 743 "steps" (each = 1 simulated hour, roughly 30 days total).
  We split by step (not random) because:
    - Random splits leak future information into training (temporal leakage)
    - Real fraud patterns evolve over time; we want the model to generalize forward
  Split: train = steps 1-500, test = steps 501-743
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Generator, Optional

import pandas as pd

from src.datasets.schema import TransactionEvent, TransactionType

logger = logging.getLogger(__name__)

# Default path relative to the repo root
DEFAULT_CSV = Path(__file__).parents[2] / "datasets" / "notebooks" / "raw" / "PS_20174392719_1491204439457_log.csv"

# Time-ordered split boundary (steps are 1-indexed hours of simulation)
TRAIN_MAX_STEP = 500
TEST_MIN_STEP = 501


class PaySimAdapter:
    """
    Loads and serves PaySim data in both batch and streaming modes.

    Usage:
        adapter = PaySimAdapter()
        train_df, test_df = adapter.batch_load()          # for training
        for event in adapter.stream(split="test"):        # for simulation
            process(event)
    """

    def __init__(self, csv_path: Path = DEFAULT_CSV):
        self.csv_path = csv_path
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_raw(self) -> pd.DataFrame:
        """Load CSV once, cache it. Re-use on subsequent calls."""
        if self._df is None:
            logger.info(f"Loading PaySim CSV from {self.csv_path} ...")
            df = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(df):,} transactions")
            self._df = df
        return self._df

    @staticmethod
    def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalise column names to snake_case to match TransactionEvent aliases.
        The CSV uses camelCase (nameOrig, oldbalanceOrg, etc.).
        """
        return df.rename(columns={
            "nameOrig": "nameOrig",          # kept as-is; Pydantic alias handles it
            "oldbalanceOrg": "oldbalanceOrg",
            "newbalanceOrig": "newbalanceOrig",
            "nameDest": "nameDest",
            "oldbalanceDest": "oldbalanceDest",
            "newbalanceDest": "newbalanceDest",
            "isFraud": "isFraud",
            "isFlaggedFraud": "isFlaggedFraud",
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def batch_load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns (train_df, test_df) split by step.

        The DataFrame keeps ALL original columns plus the derived ones needed
        by the feature pipeline. No feature engineering here — that belongs in
        src/features/feature_pipeline.py.

        Returns
        -------
        train_df : steps 1-500  (~95% of rows, ~7.6K frauds)
        test_df  : steps 501-743 (~5% of rows, ~600 frauds)
        """
        df = self._load_raw()
        train_df = df[df["step"] <= TRAIN_MAX_STEP].reset_index(drop=True)
        test_df = df[df["step"] >= TEST_MIN_STEP].reset_index(drop=True)

        logger.info(
            f"Train: {len(train_df):,} rows, {train_df['isFraud'].sum():,} frauds "
            f"({100*train_df['isFraud'].mean():.3f}%)"
        )
        logger.info(
            f"Test:  {len(test_df):,} rows, {test_df['isFraud'].sum():,} frauds "
            f"({100*test_df['isFraud'].mean():.3f}%)"
        )
        return train_df, test_df

    def stream(
        self,
        split: str = "test",
        speed_multiplier: float = 1.0,
        max_events: Optional[int] = None,
    ) -> Generator[TransactionEvent, None, None]:
        """
        Simulate a real-time stream of transactions by replaying the dataset.

        Parameters
        ----------
        split           : "train", "test", or "all"
        speed_multiplier: 1.0 = real-time (1 step = 1 simulated hour, compressed to
                          ~1 ms between events for testing). Increase for faster replay.
        max_events      : stop after N events (useful for integration tests)

        Yields
        ------
        TransactionEvent  (Pydantic-validated, ready for the feature pipeline)

        DESIGN NOTE:
          In production this generator would be replaced by a Kafka consumer.
          The generator interface is identical — swap the implementation, not the API.
        """
        df = self._load_raw()

        if split == "train":
            subset = df[df["step"] <= TRAIN_MAX_STEP]
        elif split == "test":
            subset = df[df["step"] >= TEST_MIN_STEP]
        else:
            subset = df

        subset = subset.sort_values("step")   # ensure temporal order
        count = 0

        for _, row in subset.iterrows():
            if max_events is not None and count >= max_events:
                break

            try:
                event = TransactionEvent(
                    step=int(row["step"]),
                    type=TransactionType(row["type"]),
                    amount=float(row["amount"]),
                    nameOrig=str(row["nameOrig"]),
                    oldbalanceOrg=float(row["oldbalanceOrg"]),
                    newbalanceOrig=float(row["newbalanceOrig"]),
                    nameDest=str(row["nameDest"]),
                    oldbalanceDest=float(row["oldbalanceDest"]),
                    newbalanceDest=float(row["newbalanceDest"]),
                    isFraud=int(row["isFraud"]) if "isFraud" in row else None,
                    isFlaggedFraud=int(row["isFlaggedFraud"]) if "isFlaggedFraud" in row else None,
                )
                yield event
                count += 1

                # Throttle to simulate real-time arrival rate
                if speed_multiplier > 0:
                    time.sleep(0.001 / speed_multiplier)

            except Exception as e:
                logger.warning(f"Skipping malformed row {count}: {e}")
                continue

        logger.info(f"Stream exhausted after {count:,} events")

    def fraud_only_stream(self, split: str = "test") -> Generator[TransactionEvent, None, None]:
        """
        Yields ONLY fraudulent transactions — useful for stress-testing
        the decision engine and alert pipeline.
        """
        for event in self.stream(split=split, speed_multiplier=0):
            if event.is_fraud == 1:
                yield event
