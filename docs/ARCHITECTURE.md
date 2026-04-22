# Real-Time Financial Risk Scoring Platform — Architecture & Documentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [System Architecture](#3-system-architecture)
4. [Component-by-Component Breakdown](#4-component-by-component-breakdown)
5. [Data Flow Diagrams](#5-data-flow-diagrams)
6. [Feature Engineering Deep Dive](#6-feature-engineering-deep-dive)
7. [Model Design Decisions](#7-model-design-decisions)
8. [API Reference](#8-api-reference)
9. [How to Run — Step by Step](#9-how-to-run--step-by-step)
10. [What's Not Built Yet](#10-whats-not-built-yet)

---

## 1. Project Overview

This platform scores financial transactions for fraud risk **in real time** (<50ms per transaction). It takes a stream of payment events, computes features, runs a machine learning model, applies business decision rules, and returns an `APPROVE / REVIEW / BLOCK` decision along with a human-readable explanation of **why** the model scored it that way.

**Dataset:** PaySim — 6.36 million simulated mobile money transactions, 0.13% fraud rate.

**Core Problem:** Fraud detection is an extreme class imbalance problem (1 fraud per 774 legitimate transactions). Standard accuracy metrics are useless. The system is designed around **Precision-Recall AUC** throughout.

**Baseline vs Final:**

| Model               | PR-AUC | Notes                          |
|---------------------|--------|--------------------------------|
| Logistic Regression | 0.827  | Starting point from notebook   |
| LightGBM            | ~0.930 | 12% improvement, explainable   |

---

## 2. Tech Stack

| Layer              | Technology              | Why                                                     |
|--------------------|-------------------------|---------------------------------------------------------|
| ML Model           | LightGBM                | Fast, handles imbalance, tree SHAP is exact             |
| Explainability     | SHAP (TreeExplainer)    | Exact attribution per prediction in <5ms                |
| API                | FastAPI + Uvicorn       | Async, Pydantic validation, OpenAPI docs out of the box |
| Schema validation  | Pydantic v2             | Runtime type checking at every system boundary          |
| Message broker     | Apache Kafka            | Durable, ordered, replayable event stream               |
| Online feature store | Redis                 | Sub-millisecond velocity counter lookups                |
| Drift detection    | Evidently               | Production-grade data + model drift reports             |
| Containerisation   | Docker + Compose        | Reproducible local and prod environment                 |
| Config             | Environment variables   | 12-Factor App pattern                                   |

---

## 3. System Architecture

### 3.1 Bird's Eye View

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│   PRODUCERS                    PLATFORM                      CONSUMERS      │
│                                                                             │
│  ┌──────────────┐   events   ┌──────────────────────────┐   ┌───────────┐  │
│  │ Payment GW / │──────────► │                          │──►│  Mobile   │  │
│  │ PaySim replay│            │   REAL-TIME RISK ENGINE  │   │   App     │  │
│  └──────────────┘            │                          │   └───────────┘  │
│                              │  Feature Eng → LightGBM  │                  │
│  ┌──────────────┐  HTTP POST │  → SHAP → Decision Rules │   ┌───────────┐  │
│  │  REST Client │──────────► │                          │──►│ Fraud Ops │  │
│  │  (sync path) │            │                          │   │ Dashboard │  │
│  └──────────────┘            └──────────────────────────┘   └───────────┘  │
│                                          │                                  │
│                              ┌───────────▼──────────────┐                  │
│  ┌──────────────┐            │                          │                  │
│  │ Training CSV │──────────► │   OFFLINE PIPELINE       │                  │
│  │ (PaySim)     │  batch     │                          │                  │
│  └──────────────┘            │  Features → Train →      │                  │
│                              │  Evaluate → Register     │                  │
│                              └──────────────────────────┘                  │
│                                          │                                  │
│                              ┌───────────▼──────────────┐                  │
│                              │                          │                  │
│                              │   MONITORING             │                  │
│                              │                          │                  │
│                              │  Drift Detection →       │                  │
│                              │  Auto Retraining         │                  │
│                              └──────────────────────────┘                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Full Component Map

```
Real-Time-Financial-Risk/
│
├── src/
│   ├── datasets/
│   │   ├── schema.py              ← DATA CONTRACTS (Pydantic)
│   │   └── adapters/
│   │       └── paysim_adapter.py  ← BATCH LOAD + STREAM REPLAY
│   │
│   ├── features/
│   │   └── feature_pipeline.py   ← FEATURE ENGINEERING (batch + online)
│   │
│   ├── models/
│   │   └── lgbm_model.py         ← LIGHTGBM + SHAP + VERSIONING
│   │
│   ├── training/
│   │   ├── train.py              ← OFFLINE TRAINING SCRIPT
│   │   └── retrain_pipeline.py   ← AUTOMATED RETRAINING
│   │
│   ├── ingestion/
│   │   ├── kafka_producer.py     ← TRANSACTION STREAM SIMULATOR
│   │   └── kafka_consumer.py     ← REAL-TIME SCORING CONSUMER
│   │
│   └── monitoring/
│       └── drift_detector.py     ← EVIDENTLY DRIFT + SCORE MONITOR
│
├── api/
│   ├── main.py                   ← FASTAPI INFERENCE SERVICE
│   └── schemas.py                ← API REQUEST / RESPONSE MODELS
│
├── configs/
│   └── settings.py               ← ENV-VAR CONFIGURATION
│
├── deployment/
│   ├── Dockerfile                ← MULTI-STAGE PRODUCTION IMAGE
│   ├── docker-compose.yml        ← FULL LOCAL STACK
│   └── requirements.txt          ← ALL DEPENDENCIES
│
└── scripts/
    └── run_pipeline.py           ← END-TO-END DEMO (no Docker needed)
```

---

## 4. Component-by-Component Breakdown

### 4.1 Data Contracts — `src/datasets/schema.py`

**What it does:** Defines three Pydantic models that every component uses. Think of these as the "API contract" between layers.

```
TransactionEvent          EnrichedTransaction        RiskScore
─────────────────         ───────────────────        ─────────
step                      step                       step
type                      type                       type
amount          ──────►   amount          ──────►    amount
nameOrig        features  nameOrig        model      risk_score
oldbalanceOrg   engine    error_balance_orig         decision
newbalanceOrig            error_balance_dest         top_features
nameDest                  type_TRANSFER              latency_ms
...                       orig_tx_count_1h
                          orig_amount_sum_1h
                          dest_tx_count_1h
```

**Key design choice:** Three separate models means training code, API code, and monitoring code can evolve independently. Adding a new feature means adding it to `EnrichedTransaction` — nothing else needs to change.

---

### 4.2 PaySim Adapter — `src/datasets/adapters/paysim_adapter.py`

**What it does:** Bridges the static CSV to the streaming interface the rest of the system expects.

```
PaySim CSV (6.36M rows)
         │
         ├── batch_load() ──────► (train_df, test_df)
         │                        split at step 500
         │                        time-ordered (not random!)
         │
         └── stream() ──────────► Generator[TransactionEvent]
                                  replays rows one-by-one
                                  speed_multiplier controls rate
                                  identical interface to Kafka consumer
```

**Why time-ordered split?** Splitting randomly would leak future fraud patterns into training. Steps 1-500 = training (past), steps 501-743 = test (future). This is how the model will actually be used.

---

### 4.3 Feature Engineering — `src/features/feature_pipeline.py`

**What it does:** Computes the 10 features the model uses. Runs in two modes to avoid training-serving skew.

```
BATCH MODE (training)                ONLINE MODE (inference)
─────────────────────                ───────────────────────
engineer_features_batch(df)          OnlineFeaturePipeline.transform(event)
    │                                    │
    ├─ errorBalanceOrig                  ├─ same balance error calc
    ├─ errorBalanceDest                  ├─ same type dummies
    ├─ type dummies                      └─ Redis GETSET for velocity
    └─ rolling count per step                (exact, not approximate)
         (approximation)
```

**The 10 features:**

| # | Feature | How computed | Why it matters |
|---|---------|-------------|----------------|
| 1 | `amount` | raw | Large transfers are disproportionately fraudulent |
| 2 | `error_balance_orig` | `(oldBal - newBal) - amount` | Fraudsters drain accounts; books don't balance |
| 3 | `error_balance_dest` | `(newBal - oldBal) - amount` | Destination anomaly when layering funds |
| 4 | `type_TRANSFER` | one-hot | Fraud only exists in TRANSFER and CASH_OUT |
| 5 | `type_CASH_OUT` | one-hot | Same |
| 6 | `type_CASH_IN` | one-hot | Baseline comparison |
| 7 | `type_DEBIT` | one-hot | Baseline comparison |
| 8 | `orig_tx_count_1h` | Redis counter | Velocity spike = automation = fraud |
| 9 | `orig_amount_sum_1h` | Redis sum | Total amount sent this hour |
| 10 | `dest_tx_count_1h` | Redis counter | Mule account receiving many transfers |

---

### 4.4 LightGBM Model — `src/models/lgbm_model.py`

**What it does:** Wraps LightGBM with SHAP explanations and versioned persistence.

```
FraudDetectionModel
│
├── train(X_train, y_train)
│     ├─ auto-compute scale_pos_weight = neg/pos ratio
│     ├─ fit LGBMClassifier (500 trees, depth=6)
│     └─ build TreeExplainer immediately after fit
│
├── predict_proba(X)
│     └─ returns P(fraud) ∈ [0.0, 1.0] per row
│
├── explain(X, top_k=5)
│     └─ SHAP values → top K features sorted by |impact|
│        positive = pushed toward fraud
│        negative = pushed toward legitimate
│
├── save(version)
│     └─ joblib dump: model + explainer + metadata
│        → src/models/artifacts/fraud_model_v1.joblib
│
└── load(version)
      └─ load artifact, restore explainer
         → instant rollback: just change MODEL_VERSION env var
```

**Key hyperparameters explained:**

| Parameter | Value | Why |
|-----------|-------|-----|
| `scale_pos_weight` | 774 (auto) | Penalises missing a fraud 774x more than a false positive |
| `min_child_samples` | 50 | Prevents leaf splits on <50 samples — main regulariser for imbalanced data |
| `num_leaves` | 31 | Controls complexity (2^max_depth = 64 max) |
| `subsample` | 0.8 | Each tree trained on 80% of rows — reduces overfitting |
| `learning_rate` | 0.05 | Slower learning = more trees needed but better generalisation |

---

### 4.5 Training Script — `src/training/train.py`

**What it does:** Orchestrates the full offline training run.

```
python -m src.training.train --version v1
         │
         ▼
┌─────────────────────────────────────────┐
│  1. Load PaySim via adapter             │
│     train: steps 1-500 (6.06M rows)     │
│     test:  steps 501-743 (300K rows)    │
│                                         │
│  2. engineer_features_batch()           │
│     creates 10-feature matrix           │
│                                         │
│  3. LightGBM.fit()                      │
│     early stopping on PR-AUC            │
│     (~2-3 minutes on CPU)               │
│                                         │
│  4. Evaluate                            │
│     PR-AUC on held-out test set         │
│     Find best F1 threshold              │
│     Print classification report         │
│     Print feature importances           │
│                                         │
│  5. SHAP demo on 3 fraud cases          │
│     shows which features drove score    │
│                                         │
│  6. model.save("v1")                    │
│     → src/models/artifacts/             │
└─────────────────────────────────────────┘
```

---

### 4.6 Inference API — `api/main.py`

**What it does:** Exposes the scoring engine as an HTTP service.

```
POST /v1/score
      │
      │ ScoreRequest (Pydantic validated)
      ▼
┌─────────────────────────────────────────┐
│  1. Parse → TransactionEvent            │  ~0.5ms
│                                         │
│  2. OnlineFeaturePipeline.transform()   │  ~2ms
│     ├─ balance error features           │
│     └─ Redis velocity lookup            │
│                                         │
│  3. model.predict_proba(X)              │  ~1ms
│     → risk_score ∈ [0.0, 1.0]          │
│                                         │
│  4. model.explain(X, top_k=5)           │  ~3ms
│     → {feature: shap_value, ...}        │
│                                         │
│  5. Decision Engine                     │  <0.1ms
│     score < 0.3  → APPROVE              │
│     0.3–0.7      → REVIEW               │
│     score >= 0.7 → BLOCK                │
│                                         │
│  6. Log to stdout (→ Kafka in prod)     │  async
└─────────────────────────────────────────┘
      │
      │ ScoreResponse (JSON)
      ▼
{
  "risk_score": 0.9312,
  "decision": "BLOCK",
  "top_features": {
    "error_balance_orig": 0.4231,
    "type_TRANSFER": 0.3102,
    "amount": 0.1823
  },
  "latency_ms": 6.4
}
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe (used by Kubernetes/load balancer) |
| POST | `/v1/score` | Score a single transaction |
| POST | `/v1/score/batch` | Score up to 1000 transactions |

---

### 4.7 Kafka Producer — `src/ingestion/kafka_producer.py`

**What it does:** Simulates a payment gateway publishing transactions to Kafka.

```
PaySim CSV
    │
    │ stream() → TransactionEvent (one per row, time-ordered)
    ▼
KafkaProducer
    │
    ├─ topic:     "transactions"
    ├─ key:       nameOrig  (same account → same partition → ordered)
    ├─ value:     JSON bytes
    ├─ linger_ms: 5         (batch up to 5ms for throughput)
    └─ acks:      "all"     (wait for all replicas = no data loss)
```

---

### 4.8 Kafka Consumer — `src/ingestion/kafka_consumer.py`

**What it does:** The async real-time scoring path. Reads from `transactions`, scores each, writes to `fraud-decisions`.

```
Kafka "transactions" topic
    │
    │ KafkaConsumer (group_id="fraud-scoring-service")
    ▼
FraudScoringConsumer._process_message()
    │
    ├─ deserialize JSON → TransactionEvent
    ├─ OnlineFeaturePipeline.transform()
    ├─ model.predict_proba()
    ├─ SHAP explain (only if score > 0.1, saves 3ms on clear legit)
    └─ decision engine
    │
    ▼
Kafka "fraud-decisions" topic
    │
    ├─ consumed by: Monitoring service
    ├─ consumed by: Fraud ops dashboard
    └─ consumed by: Ground truth labelling pipeline
```

**Throughput target:** >1,000 transactions/second per consumer instance. Scale by adding more consumers to the same group (each gets a subset of partitions).

---

### 4.9 Drift Detection — `src/monitoring/drift_detector.py`

**What it does:** Detects when the model's input distribution has shifted enough to require retraining.

```
REFERENCE SET                    CURRENT WINDOW
(training distribution)          (last N decisions)
        │                               │
        └──────────────┬────────────────┘
                       │
               DriftDetector.check_drift()
                       │
              Evidently DataDriftPreset
              KS test for each numerical feature
              Chi-squared for categorical features
                       │
              drift_ratio = drifted_features / total_features
                       │
               ┌───────┴────────┐
          ratio ≤ 0.3        ratio > 0.3
               │                  │
          "No drift"         "DRIFT DETECTED"
               │                  │
          log info          trigger retraining
```

**ScoreMonitor** (separate class, same file):

```
Per-decision recording:
  risk_score → rolling window
  decision   → count per type
  latency_ms → rolling window

Periodic check:
  block_rate > 10x baseline? → ALERT (fraud wave or model error)
  latency_p99 > 100ms?       → ALERT (SLA breach)
```

---

### 4.10 Retraining Pipeline — `src/training/retrain_pipeline.py`

**What it does:** Automates the full champion/challenger model lifecycle.

```
TRIGGER (drift / scheduled / manual)
         │
         ▼
┌────────────────────────────────────────────┐
│  1. Determine new version                  │
│     (latest version + 1)                   │
│                                            │
│  2. Load recent data                       │
│     sliding window: last N steps only      │
│     (recent patterns matter more)          │
│                                            │
│  3. Train new challenger model             │
│                                            │
│  4. Evaluate both on SAME test set         │
│     challenger_pr_auc vs champion_pr_auc   │
│                                            │
│  5. Promotion decision                     │
│     improvement ≥ 0.005?                   │
│         YES → save + update champion.json  │
│         NO  → log result, keep champion    │
│                                            │
│  6. Log to registry/                       │
│     {version}.json (all runs)              │
│     champion.json  (current live model)    │
└────────────────────────────────────────────┘
         │
         ▼
  Restart inference service
  → picks up new MODEL_VERSION from champion.json
```

---

## 5. Data Flow Diagrams

### 5.1 Sync Path (HTTP API)

```
Client                  API Server              Redis           Model
  │                         │                    │               │
  │── POST /v1/score ───────►│                   │               │
  │                         │                    │               │
  │                         │── GET vel:count ──►│               │
  │                         │◄─ count=3 ─────────│               │
  │                         │                    │               │
  │                         │── GET vel:amount ──►│              │
  │                         │◄─ amount=500 ───────│              │
  │                         │                    │               │
  │                         │── predict_proba(X) ────────────────►│
  │                         │◄─ 0.9312 ──────────────────────────│
  │                         │                    │               │
  │                         │── explain(X) ─────────────────────►│
  │                         │◄─ {feature: shap} ────────────────│
  │                         │                    │               │
  │                         │── INCR vel:count ─►│               │
  │                         │                    │               │
  │◄─ {score, BLOCK, ...} ──│                    │               │
  │                         │                    │               │
```

### 5.2 Async Path (Kafka Stream)

```
PaySim                Kafka               Consumer            Redis
Adapter               Broker               Group               Store
  │                     │                    │                   │
  │── publish(tx) ─────►│                    │                   │
  │── publish(tx) ─────►│                    │                   │
  │── publish(tx) ─────►│                    │                   │
  │                     │── poll(100 msgs) ──►│                  │
  │                     │                    │── GET velocity ──►│
  │                     │                    │◄─ counts ─────────│
  │                     │                    │                   │
  │                     │                    │── score all 100   │
  │                     │                    │   in loop         │
  │                     │                    │                   │
  │                     │◄── publish results─│                   │
  │                     │   (fraud-decisions)│                   │
  │                     │                    │                   │
  │                     │── ack offset ──────►│                  │
```

### 5.3 Monitoring & Retraining Loop

```
"fraud-decisions" Kafka topic
         │
         ▼
   ScoreMonitor.record()
   (rolling window of N=10,000)
         │
         ├── every 1,000 decisions
         │       check block_rate
         │       check latency_p99
         │       → alert if anomalous
         │
         └── every 50,000 decisions
                 DriftDetector.check_drift()
                 compare feature distributions
                       │
                  drift_ratio > 0.3?
                       │
                      YES
                       │
                       ▼
             retrain_pipeline.run(
               trigger="drift",
               drift_score=0.45
             )
                       │
                  new model better?
                       │
                  ┌────┴────┐
                 YES        NO
                  │          │
             promote       discard
             to champion    challenger
                  │
                  ▼
          API picks up new version
          on next restart / hot-reload
```

### 5.4 Training Pipeline Flow

```
PaySim CSV (6.36M rows)
         │
         │ PaySimAdapter.batch_load()
         │ time-ordered split at step=500
         │
         ├────────────────────────────────┐
         │                               │
    train_df                          test_df
    steps 1-500                       steps 501-743
    6.06M rows                        300K rows
    5,561 frauds (0.09%)              2,652 frauds (0.88%)
         │                               │
         ▼                               ▼
engineer_features_batch()       engineer_features_batch()
    10 features per row             10 features per row
         │                               │
         ▼                               │
  LightGBM.fit(                          │
    X_train, y_train,                    │
    eval_set=(X_test, y_test), ──────────┘
    early_stopping=50,
    eval_metric="average_precision"
  )
         │
         ▼
  Build SHAP TreeExplainer
  (exact, uses model tree structure)
         │
         ▼
  Evaluate on test set:
  ┌─────────────────────────────────┐
  │  PR-AUC: ~0.93                  │
  │  Best threshold: ~0.45          │
  │  Precision @ 0.5: ~0.99         │
  │  Recall    @ 0.5: ~0.75         │
  └─────────────────────────────────┘
         │
         ▼
  model.save("v1")
  → src/models/artifacts/fraud_model_v1.joblib
  → src/models/registry/champion.json
```

---

## 6. Feature Engineering Deep Dive

### Why `error_balance_orig` is the most powerful feature

```
LEGITIMATE transaction (TRANSFER of $100):
  oldbalanceOrg = 500
  newbalanceOrig = 400       ← decreased by exactly $100
  amount = 100

  error_balance_orig = (500 - 400) - 100 = 0   ← books balance perfectly


FRAUDULENT transaction (drain account):
  oldbalanceOrg = 181
  newbalanceOrig = 0         ← drained completely
  amount = 181

  error_balance_orig = (181 - 0) - 181 = 0    ← also 0 when drained cleanly


FRAUDULENT transaction (fabricated amount):
  oldbalanceOrg = 181
  newbalanceOrig = 0
  amount = 500               ← amount > available balance

  error_balance_orig = (181 - 0) - 500 = -319  ← STRONGLY negative
```

From the EDA: ~50% of fraud transactions show `newbalanceDest = 0` (destination balance never increases despite receiving money). This creates a systematic pattern that tree models can learn as a split condition.

### Velocity Features and Why They Need Redis

```
In-memory (training, testing):
  self._velocity_store = {
    "C1231006815": {"step": 501, "count": 3, "amount_sum": 900.0}
  }
  → lookup is O(1) dict access
  → reset when step changes (1-hour window approximation)

Redis (production):
  key: "vel:count:C1231006815:501"   → 3
  key: "vel:amount:C1231006815:501"  → 900.0
  TTL: 7200 seconds (auto-expires)

  Why Redis and not in-memory?
  → Multiple consumer instances would each have separate in-memory state
  → Redis is shared across all instances
  → LRU eviction handles memory pressure automatically
  → Atomic INCR prevents race conditions
```

---

## 7. Model Design Decisions

### Why PR-AUC, not ROC-AUC

```
PaySim: 6.35M legitimate, 8,213 fraud (0.13% positive rate)

ROC-AUC: measures true positive rate vs false positive rate
  A model predicting "all legitimate" gets ROC-AUC ≈ 0.5
  A model that's slightly better looks amazing at 0.98+
  → Misleading for imbalanced datasets

PR-AUC: measures precision vs recall (only the positive class)
  PR-AUC = 0.93 means:
  "Across ALL possible thresholds, the model achieves
   average precision of 93% for the fraud class"
  → Directly measures what matters: catching fraud without
     blocking legitimate customers
```

### Why SHAP for explanations

```
3 approaches to model explanations:

1. Feature Importance (gain)
   global metric, not per-prediction
   "amount is the most important feature overall"
   ✗ Can't explain why THIS transaction was blocked

2. LIME (Local Interpretable Model-agnostic Explanations)
   fits a local linear model around each prediction
   approximate, ~50ms per prediction
   ✗ Too slow, not exact

3. SHAP TreeExplainer  ← what we use
   exact Shapley values from the tree structure
   "For THIS transaction, error_balance_orig contributed +0.42
    to the fraud log-odds"
   ~3ms per prediction
   Satisfies CFPB/FCA explainability requirements
   ✓ Fast, exact, interpretable, per-prediction
```

### Champion/Challenger Pattern

```
Why not just deploy every new model?

Scenario: You retrain, new model has PR-AUC 0.931 vs champion 0.930.
  Improvement: +0.001
  This is within noise — could be random variation, not real improvement.
  Deploying it introduces unnecessary risk.

Our rule: only promote if improvement ≥ 0.005 (0.5% PR-AUC gain).
  Old model is never deleted (instant rollback via MODEL_VERSION env var).
  Every training run is logged with metrics.
```

---

## 8. API Reference

### `POST /v1/score`

**Request:**
```json
{
  "step": 501,
  "type": "TRANSFER",
  "amount": 181.00,
  "nameOrig": "C1231006815",
  "oldbalanceOrg": 181.00,
  "newbalanceOrig": 0.00,
  "nameDest": "C1666544295",
  "oldbalanceDest": 0.00,
  "newbalanceDest": 0.00
}
```

**Response:**
```json
{
  "step": 501,
  "name_orig": "C1231006815",
  "name_dest": "C1666544295",
  "amount": 181.0,
  "type": "TRANSFER",
  "risk_score": 0.9312,
  "decision": "BLOCK",
  "top_features": {
    "error_balance_orig": 0.4231,
    "type_TRANSFER": 0.3102,
    "amount": 0.1823,
    "error_balance_dest": -0.0941,
    "orig_tx_count_1h": 0.0312
  },
  "latency_ms": 6.4
}
```

**Decision thresholds (configurable via env vars):**

```
APPROVE   0.0 ──────────── 0.3 ──────────── 0.7 ──── 1.0   BLOCK
               APPROVE           REVIEW             BLOCK
```

### `GET /health`

```json
{
  "status": "ok",
  "model_version": "v1",
  "model_loaded": true
}
```

---

## 9. How to Run — Step by Step

### Step 1: Install dependencies
```bash
source venv/bin/activate
pip install -r deployment/requirements.txt
```

### Step 2: Train the model (~3 minutes)
```bash
python -m src.training.train --version v1
# Output: src/models/artifacts/fraud_model_v1.joblib
# Expected PR-AUC: ~0.93
```

### Step 3: Verify with end-to-end demo (no Docker needed)
```bash
python scripts/run_pipeline.py
# Runs: load → features → train → score 500 txns → drift check
```

### Step 4: Start the API
```bash
uvicorn api.main:app --reload --port 8000
# Open: http://localhost:8000/docs (Swagger UI)
```

### Step 5: Score a transaction
```bash
curl -X POST http://localhost:8000/v1/score \
  -H "Content-Type: application/json" \
  -d '{
    "step": 501, "type": "TRANSFER", "amount": 181.0,
    "nameOrig": "C1231006815", "oldbalanceOrg": 181.0, "newbalanceOrig": 0.0,
    "nameDest": "C1666544295", "oldbalanceDest": 0.0, "newbalanceDest": 0.0
  }'
```

### Step 6: Start full streaming stack
```bash
# Start Kafka + Redis
docker-compose -f deployment/docker-compose.yml up -d kafka zookeeper redis

# Produce transactions (10x speed replay)
python -m src.ingestion.kafka_producer --split test --max-events 5000

# Consume + score in real-time (separate terminal)
python -m src.ingestion.kafka_consumer --redis-url redis://localhost:6379/0

# View Kafka topics
open http://localhost:8080
```

### Step 7: Trigger retraining manually
```bash
python -m src.training.retrain_pipeline --trigger manual
# Only promotes if new model improves PR-AUC by ≥ 0.5%
```

---

## 10. What's Not Built Yet

| Component | Description | Priority |
|-----------|-------------|----------|
| Graph fraud ring detection | `src/graph/` is empty. Node2Vec/GraphSAGE to detect money mule networks by analysing the transaction graph | High |
| Anomaly detection | Isolation Forest for zero-day fraud patterns not in training data | Medium |
| Prometheus metrics | `/metrics` endpoint with request counters, latency histograms, block rate gauges | Medium |
| Airflow DAGs | Scheduled trigger for retraining_pipeline (currently manual only) | Medium |
| Feast feature store | Feature versioning and point-in-time correct joins (we only use Redis) | Low |
| Unit tests | Zero test coverage — `tests/` directory is empty | High |
| IEEE adapter | `src/datasets/adapters/ieee_adapter.py` is a stub | Low |
| A/B testing | Shadow mode to run two model versions in parallel before promoting | Medium |
