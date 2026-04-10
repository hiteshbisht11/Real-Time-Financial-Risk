"""
End-to-End Pipeline Demo Script.

This script walks through the entire system WITHOUT needing Kafka/Redis/Docker.
Run this to verify everything works locally after installing dependencies.

STEPS:
  1. Load PaySim data
  2. Engineer features
  3. Train LightGBM
  4. Evaluate (PR-AUC, SHAP)
  5. Simulate real-time scoring on 100 transactions
  6. Run drift detection (compares test vs train distribution)

HOW TO RUN:
  # Install dependencies first
  pip install -r deployment/requirements.txt

  # Run the demo
  python scripts/run_pipeline.py

Expected output:
  - PR-AUC > 0.90
  - Correct APPROVE/REVIEW/BLOCK decisions
  - SHAP explanations showing error_balance features dominating
  - Drift report: no drift expected (same dataset)
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    print("\n" + "="*70)
    print("  REAL-TIME FINANCIAL RISK SCORING — END-TO-END DEMO")
    print("="*70 + "\n")

    # ------------------------------------------------------------------ #
    # STEP 1: Load data
    # ------------------------------------------------------------------ #
    print("STEP 1: Loading PaySim dataset...")
    from src.datasets.adapters.paysim_adapter import PaySimAdapter
    adapter = PaySimAdapter()
    train_df, test_df = adapter.batch_load()
    print(f"  Train: {len(train_df):,} rows | Test: {len(test_df):,} rows\n")

    # ------------------------------------------------------------------ #
    # STEP 2: Feature engineering
    # ------------------------------------------------------------------ #
    print("STEP 2: Engineering features...")
    from src.features.feature_pipeline import engineer_features_batch, get_feature_matrix
    train_feat = engineer_features_batch(train_df)
    test_feat = engineer_features_batch(test_df)
    X_train, y_train = get_feature_matrix(train_feat)
    X_test, y_test = get_feature_matrix(test_feat)
    print(f"  Feature matrix: {X_train.shape[1]} features\n")

    # ------------------------------------------------------------------ #
    # STEP 3: Train LightGBM
    # ------------------------------------------------------------------ #
    print("STEP 3: Training LightGBM model...")
    from src.models.lgbm_model import FraudDetectionModel
    model = FraudDetectionModel(version="demo")
    model.train(X_train, y_train, X_val=X_test, y_val=y_test)
    print()

    # ------------------------------------------------------------------ #
    # STEP 4: Evaluate
    # ------------------------------------------------------------------ #
    print("STEP 4: Evaluating on test set...")
    from sklearn.metrics import average_precision_score, classification_report
    import numpy as np

    y_scores = model.predict_proba(X_test)
    pr_auc = average_precision_score(y_test, y_scores)

    # Use 0.5 threshold for classification report
    y_pred = (y_scores >= 0.5).astype(int)
    report = classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"])

    print(f"  PR-AUC: {pr_auc:.4f}")
    print(f"\n  Classification Report:\n{report}")

    print("  Top Feature Importances:")
    for feat, imp in list(model.feature_importance().items())[:5]:
        bar = "█" * int(imp / max(model.feature_importance().values()) * 30)
        print(f"    {feat:<30} {bar}")
    print()

    # ------------------------------------------------------------------ #
    # STEP 5: Real-time scoring simulation
    # ------------------------------------------------------------------ #
    print("STEP 5: Simulating real-time scoring on 10 test transactions...")
    from src.datasets.schema import RiskDecision
    from src.features.feature_pipeline import OnlineFeaturePipeline

    pipeline = OnlineFeaturePipeline()
    decision_counts = {"APPROVE": 0, "REVIEW": 0, "BLOCK": 0}

    print(f"\n  {'Type':<12} {'Amount':>12}  {'Score':>6}  {'Decision':<8}  {'Top Factor'}")
    print(f"  {'-'*12} {'-'*12}  {'-'*6}  {'-'*8}  {'-'*30}")

    sample_count = 0
    for event in adapter.stream(split="test", speed_multiplier=0, max_events=500):
        enriched = pipeline.transform(event)
        X = pipeline.to_feature_vector(enriched)
        score = float(model.predict_proba(X)[0])

        if score >= 0.7:
            decision = "BLOCK"
        elif score >= 0.3:
            decision = "REVIEW"
        else:
            decision = "APPROVE"

        decision_counts[decision] += 1

        # Only print interesting cases (non-trivial scores) up to 10
        if sample_count < 10 and score > 0.05:
            top_feats = model.explain(X, top_k=1)[0]
            top_feat = next(iter(top_feats)) if top_feats else "n/a"
            top_val = next(iter(top_feats.values())) if top_feats else 0
            label = " [FRAUD]" if event.is_fraud == 1 else ""
            print(
                f"  {event.type.value:<12} {event.amount:>12,.2f}  {score:>6.3f}  "
                f"{decision:<8}  {top_feat}: {top_val:+.3f}{label}"
            )
            sample_count += 1

    print(f"\n  Decision distribution (500 transactions):")
    total = sum(decision_counts.values())
    for dec, count in decision_counts.items():
        pct = count / total * 100
        print(f"    {dec:<8}: {count:>4} ({pct:.1f}%)")
    print()

    # ------------------------------------------------------------------ #
    # STEP 6: Drift detection
    # ------------------------------------------------------------------ #
    print("STEP 6: Running drift detection (train vs test distribution)...")
    from src.monitoring.drift_detector import DriftDetector

    detector = DriftDetector()
    train_scores = model.predict_proba(X_train[:10000])   # sample to save memory
    detector.set_reference(train_feat.iloc[:10000], train_scores)

    test_scores = model.predict_proba(X_test[:5000])
    is_drifted, report = detector.check_drift(test_feat.iloc[:5000], test_scores, save_report=False)

    print(f"  Drift detected: {is_drifted}")
    print(f"  Drift ratio: {report['drift_ratio']:.2%}")
    if report['drifted_features']:
        print(f"  Drifted features: {report['drifted_features']}")
    else:
        print("  No features drifted (expected: same dataset)")

    print("\n" + "="*70)
    print("  DEMO COMPLETE")
    print(f"  PR-AUC: {pr_auc:.4f} (Logistic Regression baseline was 0.827)")
    print(f"  Next step: python -m src.training.train --version v1  (save for API)")
    print(f"  Then:      uvicorn api.main:app --reload --port 8000")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
