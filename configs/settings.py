"""
Central configuration — loaded from environment variables with defaults.

WHY ENV VARS?
  The 12-Factor App methodology says: "store config in the environment."
  Same Docker image runs in dev (MODEL_VERSION=v1) and prod (MODEL_VERSION=v3)
  by setting different env vars — no code changes, no image rebuilds.

Usage:
    from configs.settings import settings
    print(settings.model_version)
    print(settings.block_threshold)
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Model
    model_version: str = os.getenv("MODEL_VERSION", "v1")

    # Decision thresholds
    review_threshold: float = float(os.getenv("REVIEW_THRESHOLD", "0.3"))
    block_threshold: float = float(os.getenv("BLOCK_THRESHOLD", "0.7"))

    # Kafka
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_transactions_topic: str = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")
    kafka_decisions_topic: str = os.getenv("KAFKA_DECISIONS_TOPIC", "fraud-decisions")
    kafka_consumer_group: str = os.getenv("KAFKA_CONSUMER_GROUP", "fraud-scoring-service")

    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_velocity_ttl_seconds: int = int(os.getenv("REDIS_VELOCITY_TTL", "7200"))  # 2 hours

    # Monitoring
    drift_check_window_size: int = int(os.getenv("DRIFT_WINDOW_SIZE", "50000"))
    drift_threshold: float = float(os.getenv("DRIFT_THRESHOLD", "0.3"))

    # Retraining
    min_pr_auc_improvement: float = float(os.getenv("MIN_PR_AUC_IMPROVEMENT", "0.005"))
    training_window_steps: int = int(os.getenv("TRAINING_WINDOW_STEPS", "500"))


settings = Settings()
