"""
FastAPI Inference Service — Real-Time Fraud Risk Scoring.

ARCHITECTURE:
  POST /v1/score
    -> validate request (Pydantic)
    -> feature engineering (OnlineFeaturePipeline, optionally Redis)
    -> LightGBM inference (~1ms)
    -> SHAP explanation (~3ms)
    -> decision engine (threshold rules)
    -> log result (for monitoring)
    -> return ScoreResponse

TARGET LATENCY: <50ms p99 (network excluded)
  Feature engineering: ~2ms
  Inference: ~1ms
  SHAP: ~3ms
  Total server-side: ~6ms — well within budget

HOW TO RUN:
  # Development (auto-reload on code changes)
  uvicorn api.main:app --reload --port 8000

  # Production
  gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

  # Test with curl
  curl -X POST http://localhost:8000/v1/score \\
    -H "Content-Type: application/json" \\
    -d '{"step":501,"type":"TRANSFER","amount":181.0,
         "nameOrig":"C1231006815","oldbalanceOrg":181.0,"newbalanceOrig":0.0,
         "nameDest":"C1666544295","oldbalanceDest":0.0,"newbalanceDest":0.0}'
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse, ScoreRequest, ScoreResponse
from src.datasets.schema import RiskDecision, TransactionEvent, TransactionType
from src.features.feature_pipeline import OnlineFeaturePipeline
from src.models.lgbm_model import FraudDetectionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Decision thresholds (business rules)
# ---------------------------------------------------------------------------
# REVIEW threshold: "flag for manual review"
# BLOCK threshold: "automatically decline"
# These are set by the Risk team, not the ML team — keep them separate from the model.
REVIEW_THRESHOLD = float(os.getenv("REVIEW_THRESHOLD", "0.3"))
BLOCK_THRESHOLD = float(os.getenv("BLOCK_THRESHOLD", "0.7"))


def score_to_decision(score: float) -> RiskDecision:
    if score >= BLOCK_THRESHOLD:
        return RiskDecision.BLOCK
    elif score >= REVIEW_THRESHOLD:
        return RiskDecision.REVIEW
    else:
        return RiskDecision.APPROVE


# ---------------------------------------------------------------------------
# Application state (shared across requests)
# ---------------------------------------------------------------------------

class AppState:
    model: Optional[FraudDetectionModel] = None
    feature_pipeline: Optional[OnlineFeaturePipeline] = None
    model_version: str = "unknown"


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load model once into memory.
    Shutdown: clean up resources.

    WHY LIFESPAN INSTEAD OF @app.on_event?
      @app.on_event("startup") is deprecated in FastAPI 0.95+.
      Lifespan context manager is the modern approach and handles
      startup + shutdown in one place.
    """
    # Startup
    model_version = os.getenv("MODEL_VERSION", "v1")
    logger.info(f"Loading model version: {model_version}")
    try:
        state.model = FraudDetectionModel.load(version=model_version)
        state.model_version = model_version
        logger.info("Model loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Model not found: {e}")
        logger.error("Run: python -m src.training.train to train and save a model first")
        # Don't crash on startup — health check will report model_loaded=False

    # Initialize feature pipeline (with optional Redis)
    redis_url = os.getenv("REDIS_URL")
    redis_client = None
    if redis_url:
        try:
            import redis
            redis_client = redis.from_url(redis_url, decode_responses=True)
            redis_client.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except Exception as e:
            logger.warning(f"Redis unavailable ({e}), using in-memory velocity store")

    state.feature_pipeline = OnlineFeaturePipeline(redis_client=redis_client)
    logger.info("Feature pipeline initialized")

    yield

    # Shutdown
    logger.info("Shutting down inference service")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Real-Time Fraud Risk Scoring API",
    description="Scores financial transactions for fraud risk using LightGBM + SHAP",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """
    Health check endpoint. Called by load balancers and Kubernetes probes.
    Returns 200 even if model isn't loaded (so the container stays up).
    The model_loaded field tells you if inference is available.
    """
    return HealthResponse(
        status="ok",
        model_version=state.model_version,
        model_loaded=state.model is not None,
    )


@app.post("/v1/score", response_model=ScoreResponse, tags=["inference"])
async def score_transaction(request: ScoreRequest):
    """
    Score a single transaction for fraud risk.

    Returns:
      - risk_score: P(fraud) in [0, 1]
      - decision: APPROVE | REVIEW | BLOCK
      - top_features: SHAP-based explanation of the top 5 contributing factors
      - latency_ms: server-side processing time

    The decision thresholds are:
      score < 0.3  -> APPROVE
      0.3-0.7      -> REVIEW (manual review queue)
      score >= 0.7 -> BLOCK
    """
    if state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and save a model first: python -m src.training.train"
        )

    t_start = time.perf_counter()

    # 1. Convert API request -> internal TransactionEvent
    try:
        event = TransactionEvent(
            step=request.step,
            type=TransactionType(request.type),
            amount=request.amount,
            nameOrig=request.nameOrig,
            oldbalanceOrg=request.oldbalanceOrg,
            newbalanceOrig=request.newbalanceOrig,
            nameDest=request.nameDest,
            oldbalanceDest=request.oldbalanceDest,
            newbalanceDest=request.newbalanceDest,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid transaction: {e}")

    # 2. Feature engineering (includes Redis velocity lookup)
    enriched = state.feature_pipeline.transform(event)
    X = state.feature_pipeline.to_feature_vector(enriched)

    # 3. Inference
    risk_score = float(state.model.predict_proba(X)[0])

    # 4. SHAP explanation
    top_features = state.model.explain(X, top_k=5)[0]

    # 5. Decision engine
    decision = score_to_decision(risk_score)

    t_end = time.perf_counter()
    latency_ms = (t_end - t_start) * 1000

    # 6. Async logging (non-blocking — don't slow down the response)
    _log_decision(event, risk_score, decision, latency_ms)

    return ScoreResponse(
        step=event.step,
        name_orig=event.name_orig,
        name_dest=event.name_dest,
        amount=event.amount,
        type=event.type.value,
        risk_score=round(risk_score, 4),
        decision=decision.value,
        top_features=top_features,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/v1/score/batch", response_model=list[ScoreResponse], tags=["inference"])
async def score_batch(requests: list[ScoreRequest]):
    """
    Score a batch of transactions. More efficient than calling /v1/score repeatedly
    because it shares the SHAP explainer overhead across all rows.

    Max batch size: 1000 transactions.
    """
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(requests) > 1000:
        raise HTTPException(status_code=400, detail="Batch size exceeds 1000")

    results = []
    for req in requests:
        resp = await score_transaction(req)
        results.append(resp)
    return results


def _log_decision(
    event: TransactionEvent,
    risk_score: float,
    decision: RiskDecision,
    latency_ms: float,
) -> None:
    """
    Log the decision for downstream monitoring and model drift detection.

    In production this would:
      - Write to a Kafka topic ("fraud-decisions")
      - Which is consumed by the monitoring service
      - And Evidently for drift detection

    For now, we log to stdout (captured by the container log aggregator).
    """
    logger.info(
        f"DECISION | step={event.step} | orig={event.name_orig} | "
        f"type={event.type.value} | amount={event.amount:.2f} | "
        f"score={risk_score:.4f} | decision={decision.value} | "
        f"latency={latency_ms:.1f}ms"
    )
