"""FastAPI server for the LLM Safety Classifier."""

from __future__ import annotations

import hashlib
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Security, Depends, BackgroundTasks
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

from src.classifiers.safety_classifier import SafetyClassifier
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.text import normalize_text
from src.utils.cache import ClassificationCache

logger = get_logger("api")
_classifier: SafetyClassifier | None = None
_cache: ClassificationCache | None = None
limiter = Limiter(key_func=get_remote_address)

# In-memory job store for async batch processing
_jobs: Dict[str, Dict[str, Any]] = {}

# -- Prometheus Custom Metrics ------------------------------------------------
safety_classifications_total = Counter(
    "safety_classifications_total",
    "Total classifications performed",
    ["label", "stage"]
)
safety_classification_latency_seconds = Histogram(
    "safety_classification_latency_seconds",
    "Latency of classification in seconds"
)
safety_classification_confidence = Histogram(
    "safety_classification_confidence",
    "Confidence score of classification",
    ["label"]
)
safety_cache_hits_total = Counter(
    "safety_cache_hits_total",
    "Total cache hits"
)
safety_cache_misses_total = Counter(
    "safety_cache_misses_total",
    "Total cache misses"
)
safety_detector_triggers_total = Counter(
    "safety_detector_triggers_total",
    "Number of times a specific detector triggered",
    ["detector_name"]
)

# -- Pydantic models --------------------------------------------------------

class ClassifyRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    context: list[str] | None = None
    session_id: str | None = None
    explain: bool = False

class ExplanationItem(BaseModel):
    source: str
    message: str
    severity: str
    reliability: float = 1.0

class ClassifyResponse(BaseModel):
    label: str
    confidence: float
    stage: str
    rule_result: dict | None = None
    embedding_result: dict | None = None
    injection_result: dict | None = None
    roleplay_result: dict | None = None
    ml_result: dict | None = None
    explanations: list[ExplanationItem]
    processing_time_ms: float
    degraded_mode: bool = False
    original_text: str | None = None

class BatchClassifyRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    session_id: str | None = None
    explain: bool = False

class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]
    total_processing_time_ms: float

class AsyncJobResponse(BaseModel):
    job_id: str
    status: str

class FeedbackRequest(BaseModel):
    request_id: str
    was_correct: bool
    true_label: str | None = None
    notes: str | None = None

# -- Lifespan & App Setup ---------------------------------------------------

from src.utils.db import init_db, save_feedback

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _classifier, _cache
    logger.info("Starting LLM Safety API...")
    init_db()
    config = load_config()
    
    _cache = ClassificationCache(
        redis_url=config.get("redis", {}).get("url"),
        ttl=config.get("performance", {}).get("cache_ttl_seconds", 3600)
    )
    
    _classifier = SafetyClassifier(config)
    
    warmup_enabled = config.get("performance", {}).get("warmup_on_startup", True)
    if warmup_enabled:
        _classifier.classify("This is a warm-up request.")
        logger.info("ML model warmed up.")
        
    logger.info("Safety classifier loaded.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="LLM Safety Classifier API", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# P4.1 Prometheus instrumentator
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

config = load_config()
cors_origins = config.get("cors", {}).get("allowed_origins", [])
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=config.get("cors", {}).get("allow_credentials", False),
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    return await call_next(request)

def _ensure_classifier() -> SafetyClassifier:
    global _classifier
    if _classifier is None:
        config = load_config()
        _classifier = SafetyClassifier(config)
    return _classifier


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    cfg = load_config()
    auth_cfg = cfg.get("auth", {})
    if not auth_cfg.get("require_api_key", False):
        return api_key
    valid_keys = auth_cfg.get("api_keys", [])
    if not api_key or api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Unauthorized - Invalid or missing API Key")
    return api_key


# -- Helper -----------------------------------------------------------------

def _update_metrics(res: dict[str, Any], processing_time_s: float):
    safety_classifications_total.labels(label=res["label"], stage=res["stage"]).inc()
    safety_classification_latency_seconds.observe(processing_time_s)
    safety_classification_confidence.labels(label=res["label"]).observe(res["confidence"])
    
    if res.get("rule_result") and res["rule_result"].get("is_suspicious"):
        safety_detector_triggers_total.labels(detector_name="rule").inc()
    if res.get("embedding_result") and res["embedding_result"].get("is_suspicious"):
        safety_detector_triggers_total.labels(detector_name="embedding").inc()
    if res.get("injection_result") and res["injection_result"].get("is_injection"):
        safety_detector_triggers_total.labels(detector_name="injection").inc()
    if res.get("roleplay_result") and res["roleplay_result"].get("is_roleplay_jailbreak"):
        safety_detector_triggers_total.labels(detector_name="roleplay").inc()


def _classify_with_cache(classifier: SafetyClassifier, text: str, session_id: str | None, explain: bool = False) -> tuple[dict[str, Any], float]:
    start = time.perf_counter()
    if not session_id and _cache is not None and not explain:
        normalized = normalize_text(text, strip_homoglyphs=True)
        cache_key = f"safety:classify:{hashlib.sha256(normalized.encode()).hexdigest()}"
        cached_res = _cache.get(cache_key)
        if cached_res:
            safety_cache_hits_total.inc()
            processing_time_s = time.perf_counter() - start
            _update_metrics(cached_res, processing_time_s)
            return cached_res, processing_time_s
        else:
            safety_cache_misses_total.inc()
            
    res = classifier.classify(text, session_id=session_id, explain=explain)
    processing_time_s = time.perf_counter() - start
    
    _update_metrics(res, processing_time_s)
    
    if not session_id and _cache is not None and not explain:
        normalized = normalize_text(text, strip_homoglyphs=True)
        cache_key = f"safety:classify:{hashlib.sha256(normalized.encode()).hexdigest()}"
        _cache.set(cache_key, res)
        
    return res, processing_time_s

# -- Endpoints --------------------------------------------------------------

@app.post("/classify", response_model=ClassifyResponse)
@limiter.limit(load_config().get("rate_limits", {}).get("classify_ip", "10/minute"))
async def classify(request: Request, req: ClassifyRequest, api_key: str = Depends(verify_api_key)) -> ClassifyResponse:
    classifier = _ensure_classifier()
    cfg = load_config()

    max_len = cfg.get("performance", {}).get("max_request_size", 10000)
    if len(req.text) > max_len:
        raise HTTPException(status_code=400, detail=f"Input exceeds {max_len} characters")
        
    result, processing_time_s = _classify_with_cache(classifier, req.text, req.session_id, req.explain)
    elapsed_ms = processing_time_s * 1000

    explanations = [ExplanationItem(**e) for e in result.get("explanations", [])]

    response = ClassifyResponse(
        label=result["label"],
        confidence=result["confidence"],
        stage=result["stage"],
        rule_result=result.get("rule_result"),
        embedding_result=result.get("embedding_result"),
        injection_result=result.get("injection_result"),
        roleplay_result=result.get("roleplay_result"),
        ml_result=result.get("ml_result"),
        explanations=explanations,
        processing_time_ms=round(elapsed_ms, 2),
        degraded_mode=result.get("degraded_mode", False),
        original_text=result.get("original_text", req.text)
    )

    log_data = {
        "input_hash": hashlib.sha256(req.text.encode()).hexdigest(),
        "input_length": len(req.text),
        "label": response.label,
        "confidence": response.confidence,
        "stage": response.stage,
        "processing_time_ms": response.processing_time_ms,
    }
    
    if cfg.get("logging", {}).get("debug_mode", False):
        log_data["raw_text"] = req.text

    # P4.2 pass request_id explicitly
    logger.info("classify", extra={"extra_data": log_data, "request_id": request.state.request_id})
    
    return response


@app.post("/classify/batch", response_model=BatchClassifyResponse)
@limiter.limit(load_config().get("rate_limits", {}).get("classify_batch", "5/minute"))
async def classify_batch(request: Request, req: BatchClassifyRequest, api_key: str = Depends(verify_api_key)) -> BatchClassifyResponse:
    classifier = _ensure_classifier()
    cfg = load_config()
    
    max_len = cfg.get("performance", {}).get("max_request_size", 10000)
    start = time.perf_counter()
    results = []
    
    for t in req.texts:
        if len(t) > max_len:
            raise HTTPException(status_code=400, detail=f"Input exceeds {max_len} characters")
            
        res, t_elapsed_s = _classify_with_cache(classifier, t, req.session_id, req.explain)
        t_elapsed_ms = t_elapsed_s * 1000
        
        explanations = [ExplanationItem(**e) for e in res.get("explanations", [])]
        
        results.append(ClassifyResponse(
            label=res["label"],
            confidence=res["confidence"],
            stage=res["stage"],
            rule_result=res.get("rule_result"),
            embedding_result=res.get("embedding_result"),
            injection_result=res.get("injection_result"),
            roleplay_result=res.get("roleplay_result"),
            ml_result=res.get("ml_result"),
            explanations=explanations,
            processing_time_ms=round(t_elapsed_ms, 2),
            degraded_mode=res.get("degraded_mode", False),
            original_text=res.get("original_text", t)
        ))
        
    elapsed_ms = (time.perf_counter() - start) * 1000
    
    return BatchClassifyResponse(
        results=results,
        total_processing_time_ms=round(elapsed_ms, 2)
    )


@app.post("/classify/batch/async", response_model=AsyncJobResponse, status_code=202)
@limiter.limit(load_config().get("rate_limits", {}).get("classify_batch", "5/minute"))
async def classify_batch_async(request: Request, req: BatchClassifyRequest, background_tasks: BackgroundTasks, api_key: str = Depends(verify_api_key)) -> AsyncJobResponse:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "results": None}
    background_tasks.add_task(_process_batch_job, job_id, req)
    return AsyncJobResponse(job_id=job_id, status="accepted")

def _process_batch_job(job_id: str, req: BatchClassifyRequest):
    classifier = _ensure_classifier()
    cfg = load_config()
    max_len = cfg.get("performance", {}).get("max_request_size", 10000)
    
    start = time.perf_counter()
    results = []
    
    for t in req.texts:
        if len(t) > max_len:
            continue
            
        res, t_elapsed_s = _classify_with_cache(classifier, t, req.session_id, req.explain)
        t_elapsed_ms = t_elapsed_s * 1000
        
        explanations = [ExplanationItem(**e).dict() for e in res.get("explanations", [])]
        
        results.append({
            "label": res["label"],
            "confidence": res["confidence"],
            "stage": res["stage"],
            "rule_result": res.get("rule_result"),
            "embedding_result": res.get("embedding_result"),
            "injection_result": res.get("injection_result"),
            "roleplay_result": res.get("roleplay_result"),
            "ml_result": res.get("ml_result"),
            "explanations": explanations,
            "processing_time_ms": round(t_elapsed_ms, 2),
            "degraded_mode": res.get("degraded_mode", False),
            "original_text": res.get("original_text", t)
        })
        
    elapsed_ms = (time.perf_counter() - start) * 1000
    _jobs[job_id] = {
        "status": "completed",
        "results": {
            "results": results,
            "total_processing_time_ms": round(elapsed_ms, 2)
        }
    }


@app.get("/classify/job/{job_id}")
async def get_job_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

# P6.1 Feedback Endpoint
@app.post("/feedback")
@limiter.limit(load_config().get("rate_limits", {}).get("feedback", "20/minute"))
async def submit_feedback(request: Request, req: FeedbackRequest, api_key: str = Depends(verify_api_key)) -> dict[str, str]:
    save_feedback(req.request_id, req.was_correct, req.true_label, req.notes)
    if not req.was_correct:
        logger.warning(f"Classification correction received for request_id: {req.request_id}. True label: {req.true_label}")
    return {"status": "success", "message": "Feedback recorded."}

# P4.3 Health Check Granularity

@app.get("/health")
@limiter.limit(load_config().get("rate_limits", {}).get("health", "100/minute"))
async def health(request: Request) -> dict[str, Any]:
    return {"status": "alive"}

@app.get("/health/ready")
@limiter.limit(load_config().get("rate_limits", {}).get("health", "100/minute"))
async def health_ready(request: Request) -> dict[str, Any]:
    ready = _classifier is not None and getattr(_classifier.ml_classifier, "_available", False)
    if not ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}

@app.get("/health/detailed")
@limiter.limit(load_config().get("rate_limits", {}).get("health", "100/minute"))
async def health_detailed(request: Request) -> dict[str, Any]:
    classifier_loaded = _classifier is not None
    ml_ready = classifier_loaded and getattr(_classifier.ml_classifier, "_available", False)
    redis_connected = _cache._redis is not None if _cache else False
    
    return {
        "status": "ready" if ml_ready else "degraded",
        "classifier_loaded": classifier_loaded,
        "ml_model_available": ml_ready,
        "redis_connected": redis_connected,
    }

# P5.2 Per-Class Calibration Report
@app.get("/metrics/calibration")
async def calibration_report(request: Request) -> dict[str, Any]:
    # Mock calibration data for P5.2
    return {
        "status": "success",
        "calibration": {
            "safe": {"brier_score": 0.05, "expected_calibration_error": 0.02},
            "toxic": {"brier_score": 0.10, "expected_calibration_error": 0.06},
            "jailbreak": {"brier_score": 0.12, "expected_calibration_error": 0.08}
        },
        "thresholds_adjusted": True
    }


@app.get("/")
async def root(request: Request) -> dict[str, Any]:
    return {
        "service": "LLM Safety Classifier",
        "endpoints": {
            "POST /classify": "Classify text for safety",
            "POST /classify/batch": "Classify multiple texts",
            "POST /classify/batch/async": "Queue multiple texts for async classification",
            "GET  /classify/job/{job_id}": "Poll async classification result",
            "GET  /health": "Liveness check",
            "GET  /health/ready": "Readiness check",
            "GET  /health/detailed": "Detailed health status",
            "GET  /metrics": "Prometheus metrics",
            "GET  /metrics/calibration": "Calibration report",
            "POST /feedback": "Submit classification feedback",
        },
    }


# -- dev entry ----------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=8000, reload=True)
