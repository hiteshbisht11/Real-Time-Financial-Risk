"""
Model Training Script.

Run with:
    python -m src.training.train [--version v1] [--no-save]

This script:
  1. Loads PaySim data via the adapter (time-ordered split)
  2. Engineers features in batch mode
  3. Trains LightGBM with class-imbalance handling
  4. Evaluates on the held-out test set
  5. Prints a classification report + PR-AUC
  6. Saves the model artifact

WHY PR-AUC INSTEAD OF ROC-AUC?
  With 0.13% fraud rate, a model that predicts "all legitimate" gets ROC-AUC = 0.5.
  But PR-AUC is dominated by the positive (fraud) class precision/recall tradeoff.
  PR-AUC = 0.93 means: "across all thresholds, the model catches 93% of fraud
  while keeping false positive rate low." Much more meaningful for imbalanced data.

THRESHOLD SELECTION:
  We find the threshold that maximizes F1 on the test set.
  This is reported but NOT hardcoded — the inference service applies a business
  threshold (0.3 for REVIEW, 0.7 for BLOCK) set separately.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    precision_recall_curve,
)

# Add project root to path so imports work when run as a script
sys.path.insert(0, str(Path(__file__).parents[3]))

from src.datasets.adapters.paysim_adapter import PaySimAdapter
from src.features.feature_pipeline import engineer_features_batch, get_feature_matrix
from src.models.lgbm_model import FraudDetectionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_best_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, float]:
    """
    Find the threshold that maximizes F1 score on the evaluation set.

    Returns (best_threshold, best_f1)

    TEACHING NOTE:
      In real fraud systems, thresholds are business decisions, not ML decisions.
      "What is the cost of a false positive (blocking a good customer)?" vs
      "What is the cost of a false negative (letting fraud through)?"
      Here we use F1 as a starting point, then tune based on business logic.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # F1 = 2 * (P * R) / (P + R), avoid division by zero
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    best_idx = np.argmax(f1_scores[:-1])   # last element has no threshold
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate(
    model: FraudDetectionModel,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Full evaluation: PR-AUC, optimal threshold, classification report."""
    y_scores = model.predict_proba(X_test)
    pr_auc = average_precision_score(y_test, y_scores)

    best_threshold, best_f1 = find_best_threshold(y_test, y_scores)

    y_pred = (y_scores >= threshold).astype(int)
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"])

    logger.info(f"\n{'='*60}")
    logger.info(f"PR-AUC:          {pr_auc:.4f}")
    logger.info(f"Best threshold:  {best_threshold:.4f} (F1={best_f1:.4f})")
    logger.info(f"Eval threshold:  {threshold:.2f}")
    logger.info(f"\nClassification Report:\n{report}")

    feature_imp = model.feature_importance()
    logger.info("\nTop 10 Feature Importances:")
    for feat, imp in list(feature_imp.items())[:10]:
        logger.info(f"  {feat:<30} {imp:>8.1f}")

    return {
        "pr_auc": pr_auc,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "feature_importance": feature_imp,
    }


def train(version: str = "v1", save: bool = True) -> FraudDetectionModel:
    """
    Full training pipeline. Returns the trained model.

    Steps:
      1. Load data
      2. Engineer features
      3. Get train/test matrices
      4. Train LightGBM
      5. Evaluate
      6. Save artifact
    """
    logger.info("=== PaySim Fraud Detection — LightGBM Training ===")

    # Step 1: Load
    adapter = PaySimAdapter()
    train_df, test_df = adapter.batch_load()

    # Step 2: Engineer features
    logger.info("Engineering training features...")
    train_feat = engineer_features_batch(train_df)
    logger.info("Engineering test features...")
    test_feat = engineer_features_batch(test_df)

    # Step 3: Get arrays
    X_train, y_train = get_feature_matrix(train_feat)
    X_test, y_test = get_feature_matrix(test_feat)
    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Step 4: Train
    model = FraudDetectionModel(version=version)
    model.train(X_train, y_train, X_val=X_test, y_val=y_test)

    # Step 5: Evaluate
    metrics = evaluate(model, X_test, y_test, threshold=0.5)

    # Show a SHAP example on 3 fraud cases to demonstrate explainability
    fraud_idx = np.where(y_test == 1)[0][:3]
    if len(fraud_idx) > 0:
        sample_X = X_test[fraud_idx]
        sample_scores = model.predict_proba(sample_X)
        explanations = model.explain(sample_X)
        logger.info("\n=== SHAP Explanations for 3 Fraud Cases ===")
        for i, (score, expl) in enumerate(zip(sample_scores, explanations)):
            logger.info(f"\nCase {i+1}: risk_score={score:.4f}")
            for feat, shap_val in expl.items():
                direction = "-> FRAUD" if shap_val > 0 else "-> legit"
                logger.info(f"  {feat:<30} {shap_val:+.4f}  {direction}")

    # Step 6: Save
    if save:
        path = model.save()
        logger.info(f"\nModel artifact saved: {path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LightGBM fraud detection model")
    parser.add_argument("--version", default="v1", help="Model version string (e.g. v1, v2)")
    parser.add_argument("--no-save", action="store_true", help="Skip saving the model artifact")
    args = parser.parse_args()

    trained_model = train(version=args.version, save=not args.no_save)
    logger.info("\nTraining complete!")
