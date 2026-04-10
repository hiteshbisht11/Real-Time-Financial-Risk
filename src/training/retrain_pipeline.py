"""
Automated Retraining Pipeline.

WHEN IS RETRAINING TRIGGERED?
  1. Scheduled: Every 7 days (cron job / Airflow DAG)
  2. On-demand: When drift_detector.check_drift() returns is_drifted=True
  3. Manual: When a data scientist explicitly kicks it off

RETRAINING STRATEGY:
  "Sliding window" approach:
    - Always use the MOST RECENT N steps as training data
    - Throw away very old data (fraud patterns change)
    - Validate on the most recent held-out window
    - Only deploy if the new model improves PR-AUC by > min_improvement

  WHY NOT RETRAIN ON ALL DATA?
    - Older fraud patterns may no longer be relevant
    - 6M rows takes ~10 minutes to train; recent 500K takes ~1 minute
    - Concept drift: recent patterns are more predictive than old ones

MODEL REGISTRY:
  We use a simple file-based registry (model artifacts + metadata JSON).
  In production, use MLflow or Weights & Biases for:
    - Experiment tracking (hyperparams, metrics, dataset hash)
    - Model comparison across versions
    - A/B testing infrastructure
    - Rollback to any previous version in 1 command

HOW TO RUN:
  python -m src.training.retrain_pipeline --trigger manual
  python -m src.training.retrain_pipeline --trigger drift --drift-score 0.45
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parents[3]))

logger = logging.getLogger(__name__)

MODEL_REGISTRY_DIR = Path(__file__).parents[2] / "models" / "registry"
MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)


class RetrainingPipeline:
    """
    Orchestrates the full retraining cycle:
      Load data -> Engineer features -> Train -> Evaluate -> Promote (if better)

    Usage:
        pipeline = RetrainingPipeline()
        result = pipeline.run(trigger="scheduled")
        if result["promoted"]:
            print(f"New model {result['new_version']} deployed!")
    """

    def __init__(
        self,
        min_improvement: float = 0.005,   # require 0.5% PR-AUC improvement to promote
        training_window_steps: int = 500,  # use last N steps for training
    ):
        self.min_improvement = min_improvement
        self.training_window_steps = training_window_steps

    def _get_current_champion_version(self) -> Optional[str]:
        """Load the currently deployed model version from the registry."""
        champion_file = MODEL_REGISTRY_DIR / "champion.json"
        if not champion_file.exists():
            return None
        with open(champion_file) as f:
            data = json.load(f)
        return data.get("version")

    def _promote_model(
        self,
        new_version: str,
        new_pr_auc: float,
        trigger: str,
        drift_score: Optional[float],
    ) -> None:
        """
        Promote the new model to production (champion).
        Writes to the registry so the inference service picks it up.
        """
        champion_data = {
            "version": new_version,
            "pr_auc": new_pr_auc,
            "promoted_at": datetime.now().isoformat(),
            "trigger": trigger,
            "drift_score": drift_score,
        }
        champion_file = MODEL_REGISTRY_DIR / "champion.json"
        with open(champion_file, "w") as f:
            json.dump(champion_data, f, indent=2)
        logger.info(f"Promoted model {new_version} to champion (PR-AUC={new_pr_auc:.4f})")

    def _save_registry_entry(
        self,
        version: str,
        metrics: dict,
        trigger: str,
        drift_score: Optional[float],
    ) -> None:
        """Log every training run to the registry (even non-promoted ones)."""
        entry = {
            "version": version,
            "metrics": metrics,
            "trained_at": datetime.now().isoformat(),
            "trigger": trigger,
            "drift_score": drift_score,
        }
        entry_file = MODEL_REGISTRY_DIR / f"{version}.json"
        with open(entry_file, "w") as f:
            json.dump(entry, f, indent=2)

    def run(
        self,
        trigger: str = "scheduled",
        drift_score: Optional[float] = None,
        force_promote: bool = False,
    ) -> dict:
        """
        Execute the full retraining cycle.

        Parameters
        ----------
        trigger      : "scheduled" | "drift" | "manual"
        drift_score  : drift ratio from DriftDetector (0-1), logged for tracking
        force_promote: promote even if improvement < min_improvement

        Returns
        -------
        dict with keys: new_version, pr_auc, promoted, champion_pr_auc
        """
        from sklearn.metrics import average_precision_score

        from src.datasets.adapters.paysim_adapter import PaySimAdapter
        from src.features.feature_pipeline import engineer_features_batch, get_feature_matrix
        from src.models.lgbm_model import FraudDetectionModel

        logger.info(f"=== Retraining triggered by: {trigger} | drift_score={drift_score} ===")

        # 1. Determine new version string
        existing = FraudDetectionModel.list_versions()
        if existing:
            last_num = max(int(v.replace("v", "")) for v in existing if v.startswith("v"))
            new_version = f"v{last_num + 1}"
        else:
            new_version = "v1"
        logger.info(f"New version: {new_version}")

        # 2. Load data (use the most recent training window)
        adapter = PaySimAdapter()
        train_df, test_df = adapter.batch_load()

        # Sliding window: keep only recent data for training
        max_step = train_df["step"].max()
        min_step = max(1, max_step - self.training_window_steps)
        train_df = train_df[train_df["step"] >= min_step]
        logger.info(
            f"Training window: steps {min_step}-{max_step} "
            f"({len(train_df):,} rows)"
        )

        # 3. Feature engineering
        train_feat = engineer_features_batch(train_df)
        test_feat = engineer_features_batch(test_df)
        X_train, y_train = get_feature_matrix(train_feat)
        X_test, y_test = get_feature_matrix(test_feat)

        # 4. Train new model
        new_model = FraudDetectionModel(version=new_version)
        new_model.train(X_train, y_train, X_val=X_test, y_val=y_test)

        # 5. Evaluate new model
        new_scores = new_model.predict_proba(X_test)
        new_pr_auc = float(average_precision_score(y_test, new_scores))
        logger.info(f"New model PR-AUC: {new_pr_auc:.4f}")

        # 6. Compare against current champion
        champion_version = self._get_current_champion_version()
        champion_pr_auc = 0.0

        if champion_version is not None:
            try:
                champion_model = FraudDetectionModel.load(version=champion_version)
                champ_scores = champion_model.predict_proba(X_test)
                champion_pr_auc = float(average_precision_score(y_test, champ_scores))
                logger.info(f"Champion model ({champion_version}) PR-AUC: {champion_pr_auc:.4f}")
            except Exception as e:
                logger.warning(f"Could not evaluate champion: {e}")

        # 7. Promotion decision
        improvement = new_pr_auc - champion_pr_auc
        should_promote = force_promote or (improvement >= self.min_improvement)

        if should_promote:
            new_model.save()
            self._promote_model(new_version, new_pr_auc, trigger, drift_score)
            logger.info(
                f"PROMOTED {new_version}: PR-AUC {champion_pr_auc:.4f} -> {new_pr_auc:.4f} "
                f"(+{improvement:.4f})"
            )
        else:
            logger.info(
                f"NOT promoted: improvement {improvement:.4f} < threshold {self.min_improvement}"
            )

        # Always save registry entry
        self._save_registry_entry(
            version=new_version,
            metrics={"pr_auc": new_pr_auc, "improvement": improvement},
            trigger=trigger,
            drift_score=drift_score,
        )

        return {
            "new_version": new_version,
            "pr_auc": new_pr_auc,
            "champion_pr_auc": champion_pr_auc,
            "improvement": improvement,
            "promoted": should_promote,
            "trigger": trigger,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the automated retraining pipeline")
    parser.add_argument("--trigger", default="manual", choices=["scheduled", "drift", "manual"])
    parser.add_argument("--drift-score", type=float, default=None)
    parser.add_argument("--force-promote", action="store_true")
    parser.add_argument("--window", type=int, default=500, help="Training window in steps")
    args = parser.parse_args()

    pipeline = RetrainingPipeline(training_window_steps=args.window)
    result = pipeline.run(
        trigger=args.trigger,
        drift_score=args.drift_score,
        force_promote=args.force_promote,
    )

    logger.info(f"\nResult: {json.dumps(result, indent=2)}")
    if result["promoted"]:
        logger.info(f"\nNew champion: {result['new_version']} (PR-AUC={result['pr_auc']:.4f})")
        logger.info("Restart the inference service to pick up the new model.")
