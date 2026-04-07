"""
Data & Model Drift Detection using Evidently.

WHY DRIFT DETECTION MATTERS:
  You trained your model on PaySim steps 1-500.
  Three months later, fraudsters change their pattern (e.g., start using CASH_IN).
  The model was never trained on this — it starts missing fraud silently.
  This is "model drift" (performance degradation) caused by "data drift" (distribution shift).

TWO TYPES OF DRIFT WE MONITOR:
  1. Data Drift: Have the input feature distributions changed?
     E.g., "amount" used to peak at $100-$500; now it peaks at $10K.
     Tool: Evidently DataDriftPreset

  2. Prediction/Score Drift: Has the distribution of risk scores changed?
     E.g., 1 month ago, 0.1% of transactions scored >0.7. Now it's 5%.
     Could mean: more fraud, OR the model is miscalibrated.
     Tool: Evidently TargetDriftPreset

  3. Model Performance Drift (requires labels): Is PR-AUC declining?
     Tool: Evidently ClassificationPreset (needs ground truth labels)

HOW IT WORKS IN PRODUCTION:
  - Every hour, the monitoring service reads the last N decisions from the log
  - Compares the feature distributions to the training set baseline
  - If drift score > threshold, triggers an alert and queues retraining

HOW TO RUN:
  python -m src.monitoring.drift_detector
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).parents[2] / "monitoring" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class DriftDetector:
    """
    Monitors input feature drift and prediction score drift.

    Usage:
        # 1. Set reference dataset (training distribution)
        detector = DriftDetector()
        detector.set_reference(train_df_engineered)

        # 2. Periodically check current data
        is_drifted, report = detector.check_drift(current_window_df)
        if is_drifted:
            trigger_retraining()
    """

    def __init__(self):
        self._reference_df: Optional[pd.DataFrame] = None
        self._reference_scores: Optional[np.ndarray] = None

    def set_reference(
        self,
        reference_df: pd.DataFrame,
        reference_scores: Optional[np.ndarray] = None,
    ) -> None:
        """
        Store the training-time distribution as the baseline for drift comparison.

        Parameters
        ----------
        reference_df     : Feature DataFrame from training set (after engineer_features_batch)
        reference_scores : Model predictions on the reference set (P(fraud) per row)
        """
        from src.features.feature_pipeline import MODEL_FEATURE_COLS
        self._reference_df = reference_df[MODEL_FEATURE_COLS + ["isFraud"]].copy()
        if reference_scores is not None:
            self._reference_df["risk_score"] = reference_scores
        self._reference_scores = reference_scores
        logger.info(
            f"Reference set: {len(self._reference_df):,} rows, "
            f"fraud rate={self._reference_df['isFraud'].mean()*100:.3f}%"
        )

    def check_drift(
        self,
        current_df: pd.DataFrame,
        current_scores: Optional[np.ndarray] = None,
        save_report: bool = True,
    ) -> tuple[bool, dict]:
        """
        Run Evidently drift analysis between reference and current window.

        Returns
        -------
        (is_drifted, report_dict)
          is_drifted: True if any drift threshold exceeded
          report_dict: detailed metrics per feature

        DRIFT THRESHOLD:
          We flag drift if >30% of features show statistically significant drift
          (Evidently's default uses KS test p < 0.05 for numerical, chi2 for categorical).
          This threshold is tunable: lower = more sensitive, higher = fewer false alarms.
        """
        if self._reference_df is None:
            raise RuntimeError("Call set_reference() before check_drift()")

        try:
            from evidently.metric_preset import DataDriftPreset
            from evidently.report import Report
            return self._check_drift_evidently(current_df, current_scores, save_report)
        except ImportError:
            logger.warning("evidently not installed. Using statistical fallback.")
            return self._check_drift_statistical(current_df, current_scores)

    def _check_drift_evidently(
        self,
        current_df: pd.DataFrame,
        current_scores: Optional[np.ndarray],
        save_report: bool,
    ) -> tuple[bool, dict]:
        """Full Evidently drift report."""
        from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
        from evidently.report import Report
        from src.features.feature_pipeline import MODEL_FEATURE_COLS

        current = current_df[MODEL_FEATURE_COLS].copy()
        if current_scores is not None:
            current["risk_score"] = current_scores

        reference = self._reference_df[MODEL_FEATURE_COLS].copy()
        if self._reference_scores is not None:
            reference["risk_score"] = self._reference_scores

        presets = [DataDriftPreset()]
        if "risk_score" in current.columns:
            presets.append(TargetDriftPreset())

        report = Report(metrics=presets)
        report.run(reference_data=reference, current_data=current)

        result = report.as_dict()

        # Parse Evidently output for drift flags
        drift_metrics = result.get("metrics", [])
        drifted_features = []
        for metric in drift_metrics:
            if "DataDriftTable" in str(metric.get("metric", "")):
                for col_result in metric.get("result", {}).get("drift_by_columns", {}).values():
                    if col_result.get("drift_detected"):
                        drifted_features.append(col_result.get("column_name"))

        n_features = len(MODEL_FEATURE_COLS)
        drift_ratio = len(drifted_features) / max(n_features, 1)
        is_drifted = drift_ratio > 0.3   # flag if >30% of features drift

        if save_report:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = REPORTS_DIR / f"drift_report_{timestamp}.html"
            report.save_html(str(report_path))
            logger.info(f"Drift report saved: {report_path}")

        summary = {
            "is_drifted": is_drifted,
            "drift_ratio": drift_ratio,
            "drifted_features": drifted_features,
            "current_size": len(current_df),
            "reference_size": len(self._reference_df),
            "timestamp": datetime.now().isoformat(),
        }

        if is_drifted:
            logger.warning(
                f"DATA DRIFT DETECTED: {len(drifted_features)}/{n_features} features drifted. "
                f"Drifted: {drifted_features}"
            )
        else:
            logger.info(f"No significant drift. ({len(drifted_features)}/{n_features} features shifted)")

        return is_drifted, summary

    def _check_drift_statistical(
        self,
        current_df: pd.DataFrame,
        current_scores: Optional[np.ndarray],
    ) -> tuple[bool, dict]:
        """
        Fallback drift detection using KS test (no Evidently dependency).

        Kolmogorov-Smirnov test: tests if two distributions are the same.
        p < 0.05 means "reject the null hypothesis that they're the same distribution."
        """
        from scipy import stats
        from src.features.feature_pipeline import MODEL_FEATURE_COLS

        drifted = []
        for col in MODEL_FEATURE_COLS:
            if col not in current_df.columns or col not in self._reference_df.columns:
                continue
            ks_stat, p_value = stats.ks_2samp(
                self._reference_df[col].dropna().values,
                current_df[col].dropna().values,
            )
            if p_value < 0.05:
                drifted.append({"feature": col, "ks_stat": round(ks_stat, 4), "p_value": round(p_value, 6)})

        drift_ratio = len(drifted) / max(len(MODEL_FEATURE_COLS), 1)
        is_drifted = drift_ratio > 0.3

        summary = {
            "is_drifted": is_drifted,
            "drift_ratio": drift_ratio,
            "drifted_features": [d["feature"] for d in drifted],
            "details": drifted,
            "timestamp": datetime.now().isoformat(),
        }

        if is_drifted:
            logger.warning(f"DRIFT DETECTED (KS test): {len(drifted)} features drifted")
        return is_drifted, summary


class ScoreMonitor:
    """
    Monitors the distribution of model output scores over time.

    Detects:
      - Score distribution shifts (overall model behavior change)
      - Fraud rate changes (business metric)
      - Latency spikes (operational metric)
    """

    def __init__(self, window_size: int = 10000):
        self.window_size = window_size
        self._score_history: list[float] = []
        self._decision_counts: dict[str, int] = {"APPROVE": 0, "REVIEW": 0, "BLOCK": 0}
        self._latency_history: list[float] = []

    def record(self, risk_score: float, decision: str, latency_ms: float) -> None:
        """Record a single inference result."""
        self._score_history.append(risk_score)
        self._decision_counts[decision] = self._decision_counts.get(decision, 0) + 1
        self._latency_history.append(latency_ms)

        # Keep only the last window_size entries
        if len(self._score_history) > self.window_size:
            self._score_history = self._score_history[-self.window_size:]
            self._latency_history = self._latency_history[-self.window_size:]

    def summary(self) -> dict:
        """Current window statistics."""
        if not self._score_history:
            return {"status": "no data"}

        scores = np.array(self._score_history)
        latencies = np.array(self._latency_history)
        total = sum(self._decision_counts.values())

        return {
            "window_size": len(scores),
            "score_mean": float(np.mean(scores)),
            "score_p95": float(np.percentile(scores, 95)),
            "score_p99": float(np.percentile(scores, 99)),
            "block_rate": self._decision_counts.get("BLOCK", 0) / max(total, 1),
            "review_rate": self._decision_counts.get("REVIEW", 0) / max(total, 1),
            "approve_rate": self._decision_counts.get("APPROVE", 0) / max(total, 1),
            "latency_p50_ms": float(np.percentile(latencies, 50)),
            "latency_p99_ms": float(np.percentile(latencies, 99)),
            "decision_counts": dict(self._decision_counts),
        }

    def alert_if_anomalous(self, baseline_block_rate: float = 0.001) -> Optional[str]:
        """
        Simple threshold-based alert.
        Returns an alert message if something looks wrong, None otherwise.
        """
        summary = self.summary()
        if "block_rate" not in summary:
            return None

        # Alert if block rate is 10x the baseline (sudden surge of blocks)
        if summary["block_rate"] > baseline_block_rate * 10:
            return (
                f"ALERT: Block rate {summary['block_rate']*100:.2f}% is 10x above baseline "
                f"({baseline_block_rate*100:.2f}%). Possible fraud wave or model error."
            )

        # Alert if p99 latency > 100ms
        if summary.get("latency_p99_ms", 0) > 100:
            return f"ALERT: p99 latency {summary['latency_p99_ms']:.1f}ms exceeds 100ms SLA"

        return None
