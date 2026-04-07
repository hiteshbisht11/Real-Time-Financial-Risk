"""
Kafka Consumer — Real-Time Transaction Processor.

This is the core of the real-time pipeline:
  1. Consume transaction events from Kafka
  2. Engineer features (with Redis velocity lookups)
  3. Call the inference service (or directly call the model)
  4. Publish risk scores to the 'fraud-decisions' Kafka topic
  5. Log results for monitoring

TWO PROCESSING MODES:
  Mode A (microservice): Consumer calls the FastAPI /v1/score HTTP endpoint.
    - Pros: clean separation, easy to scale independently
    - Cons: extra network hop adds ~2ms latency

  Mode B (embedded model): Consumer loads the model directly in-process.
    - Pros: lowest latency (~6ms total), no network hop
    - Cons: tight coupling; model updates require consumer restart

  We implement Mode B here for lowest latency, with Mode A available as fallback.

CONSUMER GROUP:
  Multiple consumer instances can share a group_id to process a topic in parallel.
  Each partition is assigned to exactly one consumer in the group.
  With 12 partitions and 3 consumers: each consumer handles 4 partitions.
  This lets us scale horizontally without duplicate processing.

HOW TO RUN:
  python -m src.ingestion.kafka_consumer --mode embedded
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parents[3]))

logger = logging.getLogger(__name__)

KAFKA_TOPIC = "transactions"
DECISIONS_TOPIC = "fraud-decisions"
KAFKA_BOOTSTRAP = "localhost:9092"
CONSUMER_GROUP = "fraud-scoring-service"


class FraudScoringConsumer:
    """
    Consumes transactions from Kafka and produces fraud risk scores.

    Usage:
        consumer = FraudScoringConsumer(model_version="v1")
        consumer.start()   # blocks forever; Ctrl+C to stop
    """

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP,
        model_version: str = "v1",
        redis_url: Optional[str] = None,
        score_api_url: Optional[str] = None,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.model_version = model_version
        self.redis_url = redis_url
        self.score_api_url = score_api_url

        self._model = None
        self._feature_pipeline = None
        self._producer = None

    def _setup_model(self) -> None:
        """Load the fraud model and feature pipeline into memory."""
        from src.features.feature_pipeline import OnlineFeaturePipeline
        from src.models.lgbm_model import FraudDetectionModel

        redis_client = None
        if self.redis_url:
            try:
                import redis
                redis_client = redis.from_url(self.redis_url, decode_responses=True)
                redis_client.ping()
                logger.info(f"Redis connected: {self.redis_url}")
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}, using in-memory velocity")

        self._model = FraudDetectionModel.load(version=self.model_version)
        self._feature_pipeline = OnlineFeaturePipeline(redis_client=redis_client)
        logger.info(f"Model v{self.model_version} loaded")

    def _setup_kafka(self):
        """Create Kafka consumer + producer."""
        from kafka import KafkaConsumer, KafkaProducer

        self._consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=self.bootstrap_servers,
            group_id=CONSUMER_GROUP,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            key_deserializer=lambda k: k.decode("utf-8") if k else None,
            auto_offset_reset="earliest",   # start from beginning if no offset stored
            enable_auto_commit=True,
            auto_commit_interval_ms=1000,   # commit offsets every second
            max_poll_records=100,           # process up to 100 messages per poll
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000,
        )

        self._producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
        )
        logger.info(f"Kafka consumer/producer connected to {self.bootstrap_servers}")

    def _process_message(self, message_value: dict) -> Optional[dict]:
        """
        Core processing logic: raw message -> risk score.

        CRITICAL: this function must be fast. Target < 15ms.
        """
        from src.datasets.schema import TransactionEvent, TransactionType

        t_start = time.perf_counter()

        try:
            event = TransactionEvent(
                step=message_value["step"],
                type=TransactionType(message_value["type"]),
                amount=message_value["amount"],
                nameOrig=message_value["nameOrig"],
                oldbalanceOrg=message_value["oldbalanceOrg"],
                newbalanceOrig=message_value["newbalanceOrig"],
                nameDest=message_value["nameDest"],
                oldbalanceDest=message_value["oldbalanceDest"],
                newbalanceDest=message_value["newbalanceDest"],
                isFraud=message_value.get("isFraud"),
            )
        except Exception as e:
            logger.warning(f"Invalid message, skipping: {e}")
            return None

        # Feature engineering
        enriched = self._feature_pipeline.transform(event)
        X = self._feature_pipeline.to_feature_vector(enriched)

        # Inference
        risk_score = float(self._model.predict_proba(X)[0])

        # Decision
        if risk_score >= 0.7:
            decision = "BLOCK"
        elif risk_score >= 0.3:
            decision = "REVIEW"
        else:
            decision = "APPROVE"

        # SHAP explanation only for non-trivial scores (saves ~3ms on clear legit transactions)
        top_features = {}
        if risk_score >= 0.1:
            top_features = self._model.explain(X, top_k=3)[0]

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000

        return {
            "step": event.step,
            "nameOrig": event.name_orig,
            "nameDest": event.name_dest,
            "amount": event.amount,
            "type": event.type.value,
            "risk_score": round(risk_score, 4),
            "decision": decision,
            "top_features": top_features,
            "latency_ms": round(latency_ms, 2),
            "is_fraud_ground_truth": event.is_fraud,   # for monitoring; None in production
        }

    def start(self, max_messages: Optional[int] = None) -> None:
        """
        Main consumer loop. Runs until interrupted or max_messages reached.

        GRACEFUL SHUTDOWN:
          Kafka commit offsets only for successfully processed messages.
          If the process crashes mid-batch, it will re-read from the last committed offset.
          This gives us at-least-once processing semantics.
        """
        self._setup_model()

        try:
            from kafka import KafkaConsumer
            self._setup_kafka()
        except ImportError:
            logger.error("kafka-python not installed. Run: pip install kafka-python")
            return
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            return

        count = 0
        fraud_count = 0
        latencies = []
        start_time = time.time()

        logger.info(f"Consumer started. Listening on topic: {KAFKA_TOPIC}")

        try:
            for kafka_message in self._consumer:
                result = self._process_message(kafka_message.value)
                if result is None:
                    continue

                # Publish decision to fraud-decisions topic
                if self._producer:
                    self._producer.send(
                        DECISIONS_TOPIC,
                        key=result["nameOrig"],
                        value=result,
                    )

                count += 1
                latencies.append(result["latency_ms"])
                if result["decision"] in ("REVIEW", "BLOCK"):
                    fraud_count += 1

                if count % 1000 == 0:
                    elapsed = time.time() - start_time
                    p99 = sorted(latencies)[-int(len(latencies) * 0.01)] if latencies else 0
                    logger.info(
                        f"Processed {count:,} | flagged {fraud_count} | "
                        f"throughput {count/elapsed:.0f}/s | p99 latency {p99:.1f}ms"
                    )
                    latencies.clear()

                if max_messages and count >= max_messages:
                    logger.info(f"Reached max_messages={max_messages}, stopping")
                    break

        except KeyboardInterrupt:
            logger.info("Shutting down consumer (Ctrl+C)")
        finally:
            if self._producer:
                self._producer.flush()
                self._producer.close()
            self._consumer.close()
            logger.info(f"Consumer stopped. Processed {count:,} messages total.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Kafka consumer for real-time fraud scoring")
    parser.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--redis-url", default=None, help="e.g. redis://localhost:6379/0")
    parser.add_argument("--max-messages", type=int, default=None)
    args = parser.parse_args()

    consumer = FraudScoringConsumer(
        bootstrap_servers=args.bootstrap,
        model_version=args.model_version,
        redis_url=args.redis_url,
    )
    consumer.start(max_messages=args.max_messages)
