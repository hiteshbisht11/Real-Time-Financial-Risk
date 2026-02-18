# Real-Time Financial Risk
Production-grade real-time financial risk and behavior intelligence system for fintech-style transactions. The goal is low-latency scoring (<100ms), explainability per transaction, and continuous learning via monitoring and retraining.

**Core Goals**
- Process live transaction events.
- Generate fraud risk scores in real time.
- Learn behavioral patterns and detect anomalies.
- Detect graph-based fraud rings.
- Monitor data/model drift and trigger retraining.
- Provide explainability for each decision.

**HLD Architecture**
```text
                                                ┌──────────────────────┐
                                                │   Mobile App / SDK   │
                                                └─────────┬────────────┘
                                                        │
                                                        ▼
                                                ┌──────────────────────┐
                                                │     API Gateway      │
                                                │ (Auth, Rate Limit)   │
                                                └─────────┬────────────┘
                                                        │
                                                        ▼
                                                ┌──────────────────────┐
                                                │ Transaction Service  │
                                                │ (Validation Layer)   │
                                                └─────────┬────────────┘
                                                        │
                                        ┌──────────────┼──────────────┐
                                        ▼                              ▼
                            ┌──────────────────────┐        ┌──────────────────────┐
                            │  Sync Risk Scoring   │        │   Event Streaming    │
                            │  (Real-Time Path)    │        │   (Kafka Topic)      │
                            └─────────┬────────────┘        └─────────┬────────────┘
                                        │                               │
                                        ▼                               ▼
                            ┌──────────────────────┐        ┌──────────────────────┐
                            │ Online Feature Store │        │  Data Lake (S3)      │
                            │  (Redis / Feast)     │        │  + Offline Store     │
                            └─────────┬────────────┘        └─────────┬────────────┘
                                        │                               │
                                        ▼                               ▼
                            ┌──────────────────────┐        ┌──────────────────────┐
                            │  Model Inference     │        │  Training Pipeline   │
                            │  Service             │        │ (Spark / Airflow)    │
                            └─────────┬────────────┘        └─────────┬────────────┘
                                        │                               │
                                        ▼                               ▼
                            ┌──────────────────────┐        ┌──────────────────────┐
                            │  Decision Engine     │        │  Model Registry      │
                            │  (Rules + Ensemble)  │        │  (MLflow)            │
                            └─────────┬────────────┘        └─────────┬────────────┘
                                        │                               │
                                        ▼                               ▼
                            ┌──────────────────────┐        ┌──────────────────────┐
                            │ Response to Client   │        │  Monitoring & Drift  │
                            └──────────────────────┘        └──────────────────────┘
```

**Planned Modules**
- Transaction simulator and ingestion service.
- Batch feature engineering and supervised fraud model.
- Real-time inference API with explainability.
- Streaming feature updates and online store.
- Anomaly detection layer.
- Graph-based fraud ring detection.
- Monitoring and drift detection.
- Automated retraining pipeline.

**Tech Stack (Planned)**
- Streaming: Kafka or Redpanda.
- Online store: Redis + Feast.
- Model serving: FastAPI.
- Training: LightGBM or XGBoost.
- Monitoring & drift: Evidently + Prometheus.
- Orchestration (later): Airflow.

**Models (Planned)**
- Supervised fraud model: LightGBM or XGBoost.
- Anomaly detection: Isolation Forest.
- Graph risk scoring (later): Node2Vec or GraphSAGE.
- Explainability: SHAP.

**License**
Apache 2.0. See `LICENSE`.
