"""
Kafka Producer — Transaction Stream Simulator.

This simulates the "mobile app / payment gateway" that produces transaction events.
In production, your payment gateway would publish events here.
For development, we replay the PaySim dataset to simulate real traffic.

KAFKA CONCEPTS (for learning):
  Producer: publishes messages to a topic
  Topic: a named stream of records (like a log file, but distributed)
  Partition: a topic is split into N partitions for parallelism
  Key: determines which partition a message goes to
       We use nameOrig as the key so all transactions from the same account
       land on the same partition — preserving order per account.

MESSAGE FORMAT:
  We serialize TransactionEvent as JSON bytes.
  In production, use Avro or Protobuf for:
    - Schema enforcement (rejects malformed messages)
    - ~5x smaller payloads
    - Schema evolution (add fields without breaking consumers)

HOW TO RUN:
  # Start Kafka first (see docker-compose.yml):
  docker-compose up -d kafka zookeeper

  # Run the producer:
  python -m src.ingestion.kafka_producer --split test --max-events 1000
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
KAFKA_BOOTSTRAP = "localhost:9092"


def create_producer(bootstrap_servers: str = KAFKA_BOOTSTRAP):
    """
    Create and return a Kafka producer.
    Returns None if kafka-python is not installed (graceful degradation).
    """
    try:
        from kafka import KafkaProducer
        producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            # Batching: wait up to 5ms to accumulate messages before sending
            # This improves throughput at the cost of a tiny latency increase
            linger_ms=5,
            batch_size=16384,   # 16KB batch
            # Compression reduces network bandwidth by ~5x for JSON
            compression_type="gzip",
            # Reliability: wait for all replicas to acknowledge
            acks="all",
            retries=3,
        )
        logger.info(f"Kafka producer connected to {bootstrap_servers}")
        return producer
    except ImportError:
        logger.warning("kafka-python not installed. Run: pip install kafka-python")
        return None
    except Exception as e:
        logger.error(f"Failed to connect to Kafka: {e}")
        return None


def publish_transactions(
    bootstrap_servers: str = KAFKA_BOOTSTRAP,
    split: str = "test",
    speed_multiplier: float = 10.0,
    max_events: Optional[int] = None,
    dry_run: bool = False,
) -> int:
    """
    Replay PaySim transactions onto the Kafka 'transactions' topic.

    Parameters
    ----------
    bootstrap_servers : Kafka broker address
    split             : "train", "test", or "all"
    speed_multiplier  : how fast to replay (1.0 = real-time, 10.0 = 10x faster)
    max_events        : stop after N events
    dry_run           : print events without actually sending to Kafka

    Returns
    -------
    Number of events published
    """
    from src.datasets.adapters.paysim_adapter import PaySimAdapter

    adapter = PaySimAdapter()
    producer = None if dry_run else create_producer(bootstrap_servers)

    count = 0
    fraud_count = 0
    start_time = time.time()

    logger.info(f"Starting stream: split={split}, speed={speed_multiplier}x, dry_run={dry_run}")

    for event in adapter.stream(
        split=split,
        speed_multiplier=speed_multiplier,
        max_events=max_events,
    ):
        # Convert Pydantic model to dict for JSON serialization
        # Use the original field names (aliases) for wire compatibility
        message = {
            "step": event.step,
            "type": event.type.value,
            "amount": event.amount,
            "nameOrig": event.name_orig,
            "oldbalanceOrg": event.old_balance_orig,
            "newbalanceOrig": event.new_balance_orig,
            "nameDest": event.name_dest,
            "oldbalanceDest": event.old_balance_dest,
            "newbalanceDest": event.new_balance_dest,
            "isFraud": event.is_fraud,
        }

        if dry_run:
            if count < 5:   # Print first 5 to verify
                logger.info(f"[DRY RUN] Message: {json.dumps(message, indent=2)}")
        elif producer is not None:
            # Partition by nameOrig: all transactions from one account go to same partition
            # This ensures the consumer sees events in order per account
            producer.send(
                topic=KAFKA_TOPIC,
                key=event.name_orig,
                value=message,
            )

        if event.is_fraud == 1:
            fraud_count += 1

        count += 1
        if count % 10000 == 0:
            elapsed = time.time() - start_time
            throughput = count / elapsed
            logger.info(
                f"Published {count:,} events ({fraud_count} frauds) | "
                f"{throughput:.0f} events/sec"
            )

    # Flush remaining messages in the buffer
    if producer is not None:
        producer.flush()
        producer.close()

    elapsed = time.time() - start_time
    logger.info(
        f"Stream complete: {count:,} events ({fraud_count} frauds) in {elapsed:.1f}s | "
        f"avg {count/elapsed:.0f} events/sec"
    )
    return count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Publish PaySim transactions to Kafka")
    parser.add_argument("--bootstrap", default=KAFKA_BOOTSTRAP)
    parser.add_argument("--split", default="test", choices=["train", "test", "all"])
    parser.add_argument("--speed", type=float, default=10.0)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    publish_transactions(
        bootstrap_servers=args.bootstrap,
        split=args.split,
        speed_multiplier=args.speed,
        max_events=args.max_events,
        dry_run=args.dry_run,
    )
