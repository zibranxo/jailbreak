# LLM Safety System - Execution Plan

> **Goal**: Transform this codebase from a solid development prototype into a production-grade LLM safety classification service.
>
> **Guiding Principles**:
> 1. Close all security gaps before adding features
> 2. Wire up dead code before writing new code
> 3. Build observability before scaling
> 4. Ship incrementally — every PR should be deployable

---

## Phase 0: Foundation (Before touching anything else)

### P0.0 — Verify Test Suite
- `pytest tests/ -v` must pass 100%
- Fix any broken tests before proceeding
- Add test coverage reporting

### P0.1 — Project Hygiene
- [ ] Remove all `__pycache__` / `.pyc` files from repo (add to `.gitignore`)
- [ ] Add `.gitignore` if not present (Python, IDE, OS artifacts)
- [ ] Add `CHANGELOG.md` skeleton
- [ ] Add `CONTRIBUTING.md` guidelines
- [ ] Pin all dependency versions in `requirements.txt` (remove `>=` ranges)
- [ ] Add `pre-commit` hooks (black, flake8, mypy)

### P0.2 — Set Up CI/CD Pipeline
- [ ] GitHub Actions workflow:
  - [ ] Lint (black, flake8, mypy)
  - [ ] Test (pytest with coverage)
  - [ ] Build Docker image
  - [ ] (Optional) Push to registry on `main` branch
- [ ] Add status badge to `README.md`

### P0.3 — Add `.env` Support
- [ ] Install `python-dotenv`
- [ ] Create `.env.example` with all configurable variables
- [ ] Update `config.yaml` loading to merge with env vars (env overrides yaml)
- [ ] Never commit `.env` (add to `.gitignore`)

---

## Phase 1: Close Security Gaps (Non-Negotiable Before Production)

### P1.1 — Add Rate Limiting
**Why**: Unauthenticated API is trivial to abuse.
**How**:
- Install `slowapi` (or use Starlette `RateLimitMiddleware`)
- Add to `api/server.py`:
  ```python
  from slowapi import Limiter, _rate_limit_exceeded_handler
  from slowapi.util import get_remote_address
  from slowapi.errors import RateLimitExceeded

  limiter = Limiter(key_func=get_remote_address)
  app = FastAPI(...)
  app.state.limiter = limiter
  app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
  ```
- Apply limits:
  - `/classify`: 10 req/min per IP, 100 req/min per API key
  - `/health`: 100 req/min (generous)
- Add rate limit config to `config.yaml`:
  ```yaml
  rate_limits:
    classify_ip: "10/minute"
    classify_api_key: "100/minute"
    health: "100/minute"
  ```

### P1.2 — Add API Key Authentication
**Why**: Every production API needs authenticated access.
**How**:
- Add `X-API-Key` header validation middleware
- Support two modes (configurable):
  - `require_api_key: true` (production) — reject without valid key
  - `require_api_key: false` (development) — allow but warn
- Store API keys in config or environment variable (comma-separated list)
- Add to `config.yaml`:
  ```yaml
  auth:
    require_api_key: true
    api_keys: []  # Set via env var SAFETY_API_KEYS or secrets manager
  ```
- Return `401 Unauthorized` with descriptive message

### P1.3 — Add Input Sanitization & Validation
**Why**: Prevent ReDoS, memory exhaustion, and encoding attacks.
**How**:
- In `api/server.py`, before passing to classifier:
  - Trim whitespace (already in Pydantic, but double-check)
  - Normalize Unicode (NFKC) — use `src.utils.text.normalize_text`
  - Strip zero-width chars — use `src.utils.text.normalize_text`
  - Limit max chars (already 10000 in Pydantic)
  - Add max_request_size to config
- In `SafetyClassifier.classify()`, add:
  ```python
  if len(text) > MAX_INPUT_LENGTH:
      raise ValueError(f"Input exceeds {MAX_INPUT_LENGTH} characters")
  ```
- Add regex timeout for rule matching (prevent ReDoS):
  ```python
  import signal  # or use `timeout-decorator` library
  # Or better: ensure all regexes are anchored/simple
  ```

### P1.4 — Sanitize Logs (PII Protection)
**Why**: Input text may contain sensitive data. Never log raw input.
**How**:
- In `api/server.py`, log only:
  - Input hash (SHA256)
  - Input length
  - Classification result (label, confidence)
  - Processing time
  - Request ID
- Never log the raw `text` field
- If debugging is needed, add a `debug_mode: bool` config flag that logs full input locally (not to stdout)

### P1.5 — Add CORS Configuration
**Why**: API will be called from browser-based UIs.
**How**:
- Add `fastapi.middleware.cors` with configurable allowed origins
- Default: no CORS (secure by default)
- Config:
  ```yaml
  cors:
    allowed_origins: []  # Set in production
    allow_credentials: false
  ```

### Deliverable at end of Phase 1
- [ ] All new endpoints require API key (except `/health`)
- [ ] Rate limits enforced
- [ ] Input normalized before classification
- [ ] No raw user input in logs
- [ ] CI/CD runs on every PR

---

## Phase 2: Wire Up Dead Code & Core Features

### P2.1 — Integrate PromptInjectionDetector into SafetyClassifier
**Why**: The detector exists, is tested, but never called.
**How**:
- In `src/classifiers/safety_classifier.py`:
  ```python
  from src.detectors.injection import PromptInjectionDetector

  self.injection_detector = PromptInjectionDetector(
      threshold=t["injection_threshold"]
  )
  ```
- In `.classify()`:
  ```python
  injection_result = self.injection_detector.detect(text)
  # If injection detected with high confidence, short-circuit or boost
  if injection_result.is_injection and injection_result.confidence >= self._fast_threshold:
      return self._build_result(text, "jailbreak", injection_result.confidence, "fast_filter", ...)
  ```
- Include injection triggers in `_infer_label_from_fast()` and `_fuse_results()`
- Update `RuleResult` to include injection data (or create unified trigger format)

### P2.2 — Integrate RoleplayDetector into SafetyClassifier
**Why**: Same as above — tested but unused.
**How**:
- Similar to P2.1, add `RoleplayDetector` instantiation
- In `classify()`, call `roleplay_detector.detect(text)`
- If roleplay detected with confidence >= threshold, treat similarly to injection
- Update fusion logic: roleplay → "jailbreak" label

### P2.3 — Add Input Normalization to Classification Pipeline
**Why**: Zero-width chars, homoglyphs, and Unicode tricks bypass rules.
**How**:
- In `SafetyClassifier.classify()`, before calling any detector:
  ```python
  from src.utils.text import normalize_text
  normalized_text = normalize_text(text, strip_homoglyphs=True)
  # Run detection on normalized_text
  ```
- Update all detectors to accept both original and normalized text (for explanation purposes, we want to show what was matched in the original)
- Store original text in result for explanation context

### P2.4 — Implement Context-Aware Classification (Use the `context` Fieldtas
**Why**: `ClassifyRequest.context` is accepted but completely ignored. This is the biggest missing feature.
**How**:
- Add session storage (in-memory dict with TTL, or Redis if available):
  ```python
  # Simple in-memory with TTL
  from collections import OrderedDict
  import time

  class SessionStore:
      def __init__(self, max_turns=5, ttl_seconds=300):
          self.max_turns = max_turns
          self.ttl = ttl_seconds
          self._store = OrderedDict()  # session_id -> [(timestamp, turn, label)]

      def get_history(self, session_id: str) -> list:
          now = time.time()
          # Clean expired entries
          self._store = OrderedDict((k, v) for k, v in self._store.items() if now - v[-1][0] < self.ttl)
          return self._store.get(session_id, [])

      def add_turn(self, session_id: str, label: str, confidence: float):
          history = self._store.get(session_id, [])
          history.append((time.time(), label, confidence))
          if len(history) > self.max_turns:
              history.pop(0)
          self._store[session_id] = history
  ```
- In `SafetyClassifier.classify()`:
  - If `context` or `session_id` is provided, fetch history
  - Compute conversation-level features:
    - Number of previous jailbreak attempts
    - Escalation pattern (benign → benign → jailbreak)
    - Repeated toxic language across turns
  - Adjust confidence: boost if pattern of escalation detected
  - Store result in session for next turn

### P2.5 — Add Batch Classification Endpoint
**Why**: `MLClassifier.predict_batch()` exists but API doesn't expose it.
**How**:
- Add to `api/server.py`:
  ```python
  class BatchClassifyRequest(BaseModel):
      texts: list[str] = Field(..., min_length=1, max_length=100)
      session_id: str | None = None

  class BatchClassifyResponse(BaseModel):
      results: list[ClassifyResponse]
      total_processing_time_ms: float
  ```
- Endpoint: `POST /classify/batch`
- Internally call `classifier.classify()` for each text (parallelized) or better: batch the ML prediction
- Add rate limit: 5 batch requests/minute (higher cost)

### P2.6 — Implement Graceful Degradation
**Why**: If ML model is missing, the whole API should not crash.
**How**:
- In `MLClassifier._ensure_loaded()`, catch exceptions and set `self._available = False`
- In `SafetyClassifier`:
  - If ML unavailable, rely entirely on rules + embeddings
  - If embeddings unavailable, rely on rules only
  - Add a "degraded_mode" flag to response
- Log warnings when running in degraded mode

### Deliverable at end of Phase 2
- [ ] Injection and roleplay detectors are active in production
- [ ] Input normalization runs before every classification
- [ ] Context parameter actually affects classification results
- [ ] Batch endpoint available
- [ ] Service starts even without ML model

---

## Phase 3: Performance & Caching

### P3.1 — Implement Redis / LRU Caching Layer
**Why**: Classification results for identical inputs should be instant.
**How**:
- Add cache key computation:
  ```python
  import hashlib
  cache_key = f"safety:classify:{hashlib.sha256(normalized_text.encode()).hexdigest()}"
  ```
- Add `src/utils/cache.py`:
  ```python
  from functools import lru_cache
  import redis

  class ClassificationCache:
      def __init__(self, redis_url: str | None = None, ttl: int = 3600):
          self._redis = redis.Redis.from_url(redis_url) if redis_url else None
          self._local = lru_cache(maxsize=1000) if not redis_url else None
          self.ttl = ttl

      def get(self, key: str) -> dict | None:
          if self._redis:
              data = self._redis.get(key)
              return json.loads(data) if data else None
          return self._local.get(key) if self._local else None

      def set(self, key: str, value: dict):
          if self._redis:
              self._redis.setex(key, self.ttl, json.dumps(value))
          elif self._local:
              self._local[key] = value
  ```
- Cache in `api/server.py` before calling classifier:
  - Check cache, return if hit
  - Store result after classification
- Add cache hit/miss metrics to Prometheus

### P3.2 — Add Async Background Processing
**Why**: Heavy classification (batch, or cold ML model) shouldn't block the API.
**How**:
- Add Celery/Redis task queue (or use FastAPI background tasks for simple cases)
- For batch requests, queue and return `202 Accepted` with `job_id`
- Add `GET /classify/job/{job_id}` for polling results
- Add webhook callback support

### P3.3 — Optimize Embedding Index
**Why**: Linear scan of embeddings doesn't scale.
**How**:
- Evaluate: FAISS (exact or HNSW) vs. Annoy vs. current linear scan
- For 27 patterns, current approach is fine. For >1000, need FAISS.
- Add `src/utils/vector_index.py` abstraction:
  ```python
  class VectorIndex:
      def add(self, id: str, vector: np.ndarray): ...
      def search(self, query: np.ndarray, k: int) -> list[Match]: ...
  ```
- Implement FAISS-backed version as default when available
- Keep numpy linear scan as fallback

### P3.4 — Add Model Warm-Up on Startup
**Why**: First request after restart is slow (lazy loading).
**How**:
- In `api/server.py` lifespan, after creating classifier, run:
  ```python
  # Warm-up
  classifier.classify("This is a warm-up request.")
  logger.info("ML model warmed up.")
  ```
- Make warm-up configurable (can be disabled in config)

### P3.5 — Pre-download Models in Docker Build
**Why**: Air-gapped / offline deployment requires models to be present.
**How**:
- Add `scripts/download_models.py`:
  ```python
  from transformers import AutoTokenizer, AutoModelForSequenceClassification
  from sentence_transformers import SentenceTransformer

  # Download and cache models
  SentenceTransformer("all-MiniLM-L6-v2")
  AutoTokenizer.from_pretrained("distilbert-base-uncased")
  ```
- In Dockerfile, run this script during build:
  ```dockerfile
  RUN python scripts/download_models.py
  ```
- Set `HF_HOME` / `TRANSFORMERS_CACHE` to a known path in the image

### Deliverable at end of Phase 3
- [ ] Classification results cached (Redis if available, LRU otherwise)
- [ ] Batch jobs process asynchronously
- [ ] Model warmed up on startup
- [ ] Docker image includes pre-downloaded models
- [ ] Embedding search scales to >1000 patterns

---

## Phase 4: Observability & Monitoring

### P4.1 — Add Prometheus Metrics Endpoint
**Why**: Essential for production monitoring and alerting.
**How**:
- Install `prometheus-fastapi-instrumentator`
- Add to `api/server.py`:
  ```python
  from prometheus_fastapi_instrumentator import Instrumentator

  Instrumentator().instrument(app).expose(app, endpoint="/metrics")
  ```
- Custom metrics:
  - `safety_classifications_total` (label: label_name)
  - `safety_classification_latency_seconds` (histogram)
  - `safety_classification_confidence` (histogram)
  - `safety_cache_hits_total`
  - `safety_cache_misses_total`
  - `safety_detector_triggers_total` (label: detector_name)

### P4.2 — Add Structured Logging with Request IDs
**Why**: Correlate logs across distributed systems.
**How**:
- Add middleware to generate UUID per request:
  ```python
  @app.middleware("http")
  async def add_request_id(request: Request, call_next):
      request.state.request_id = str(uuid.uuid4())
      return await call_next(request)
  ```
- Update logger to include `request_id` in every log entry
- Pass `request_id` through to all subsystems (detectors, classifier)
- Log classification decisions:
  ```json
  {
    "timestamp": "...",
    "level": "INFO",
    "request_id": "...",
    "event": "classification",
    "input_hash": "sha256...",
    "input_length": 120,
    "label": "jailbreak",
    "confidence": 0.95,
    "stage": "fast_filter",
    "processing_time_ms": 2.3,
    "client_ip": "..."
  }
  ```

### P4.3 — Add Health Check Granularity
**Why**: `/health` should distinguish between "alive" and "ready to serve".
**How**:
- `GET /health` → basic liveness (service is running)
- `GET /health/ready` → readiness (all dependencies ok):
  - ML model loaded
  - Redis connected (if configured)
  - Pattern files loaded
  - Recent classification success (last 5 minutes)
- `GET /health/detailed` → verbose status of all subsystems

### P4.4 — Add Alerting Thresholds
**Why**: Proactive monitoring of system health.
**How**:
- Documented in runbook (see Phase 5):
  - Error rate > 1% for >2 minutes → PagerDuty
  - P99 latency > 200ms for >5 minutes → Slack alert
  - ML model not loaded after startup → Page
  - Redis connection lost → Log + degrade gracefully

### Deliverable at end of Phase 4
- [ ] `/metrics` endpoint with Prometheus metrics
- [ ] Every log entry has a `request_id`
- [ ] Ready/health checks distinguish liveness from readiness
- [ ] Alert runbook defined

---

## Phase 5: Explainability & Trust

### P5.1 — Integrate SHAP for ML Explanations
**Why**: The dependency exists but is never used. Regulatory requirements increasingly demand explainable AI.
**How**:
- Add `src/explainers/shap_explainer.py`:
  ```python
  import shap
  from transformers import AutoTokenizer, AutoModelForSequenceClassification

  class SHAPExplainer:
      def __init__(self, model, tokenizer):
          self.explainer = shap.Explainer(model, tokenizer)

      def explain(self, text: str, target_class: str = "jailbreak"):
          shap_values = self.explainer([text])
          # Extract top contributing tokens
          tokens = shap_values.data[0]
          values = shap_values.values[0]
          return [(token, value) for token, value in zip(tokens, values)]
  ```
- Integrate into `aggregate_explanations()`:
  - If ML result available AND confidence > threshold, run SHAP
  - Add per-token contribution to explanation
- Add `?explain=true` query param to API to optionally include SHAP (expensive)

### P5.2 — Add Per-Class Calibration Report
**Why**: Confidence scores should be trustworthy.
**How**:
- After evaluation, generate calibration curves (confidence vs. accuracy)
- Store calibration data and expose via `/metrics/calibration`
- Adjust thresholds based on calibration

### P5.3 — Add Explanation Confidence Scores
**Why**: Not all explanations are equally meaningful.
**How**:
- Each explanation gets a `reliability` score:
  - Rule match: reliability = match confidence
  - Embedding match: reliability = similarity score
  - ML prediction: reliability = calibration-aware confidence
- Frontend can highlight "highly reliable" explanations

### Deliverable at end of Phase 5
- [ ] SHAP explanations available for ML predictions
- [ ] Calibration data tracked and exposed
- [ ] Explanations have reliability scores

---

## Phase 6: Active Learning & Feedback Loop

### P6.1 — Add `/feedback` Endpoint
**Why**: Learn from mistakes in production.
**How**:
  ```python
  class FeedbackRequest(BaseModel):
      request_id: str
      was_correct: bool
      true_label: str | None = None
      notes: str | None = None
  ```
- `POST /feedback` accepts feedback for a prior classification
- Store in feedback queue (Redis list, or SQLite table)
- Include: original input hash, predicted label, confidence, true label (if provided)
- If `was_correct=False`, flag for review

### P6.2 — Periodic Retraining Job
**Why**: Model accuracy degrades over time as attackers adapt.
**How**:
- Add `scripts/retrain.py`:
  - Collect feedback from past 24h
  - Filter high-confidence disagreements (e.g., predicted safe but marked jailbreak)
  - Add to training dataset
  - Run fine-tuning with incremental learning or full retrain
  - Evaluate on holdout set
  - If new model beats old: swap in atomically
- Trigger via cron job or CI/CD pipeline

### P6.3 — Pattern Auto-Discovery
**Why**: Reduce manual curation of pattern database.
**How**:
- When embedding matches but no rule matches, flag as "candidate pattern"
- Cluster similar unmatched inputs
- Human review interface to approve/reject candidate patterns
- Approved patterns auto-added to `unsafe_patterns.jsonl`

### Deliverable at end of Phase 6
- [ ] `/feedback` endpoint accepting classification corrections
- [ ] Feedback stored and processed for retraining
- [ ] Auto-discovered patterns flagged for human review

---

## Phase 7: Advanced Features (Future)

### P7.1 — Multi-Turn Attack Chaining Detection
- Detect sequences where individual turns are benign but the conversation escalates to an attack
- Maintain per-session state with conversation trajectory
- Look for patterns: benign → probe → escalation → attack

### P7.2 — Adversarial Robustness Testing Suite
- Add `scripts/adversarial_test.py`:
  - Generate paraphrased variants of known attacks
  - Test base64, rot13, zero-width char obfuscation
  - Ensure classifier still detects
- Run as part of CI/CD

### P7.3 — Cross-Lingual Safety Detection
- Evaluate multilingual BERT for non-English inputs
- Add language detection pre-filter
- Extend pattern database to cover other languages

### P7.4 — ONNX Model Export & Optimization
- Exportseduce model size and inference time
- Export trained DistilBERT to ONNX
- Benchmark vs. PyTorch performance

### P7.5 — Web-Based Admin Dashboard
- Simple Streamlit/FastAPI-rendered UI
- View classifications, filter by label/ date
- Review flagged feedback
- Approve/reject candidate patterns
- View metrics (classification rate, latency, accuracy)

---

## Execution Order Summary

| Phase | Items | Rationale |
|-------|-------|-----------|
| **P0: Foundation** | Tests, CI/CD, .env, hygiene | Everything else depends on this |
| **P1: Security** | Rate limits, auth, input sanitization, log sanitization | Must fix before any public deployment |
| **P2: Core Features** | Wire detectors, context awareness, batch API, graceful degradation | Make the system actually work as designed |
| **P3: Performance** | Caching, async, model warm-up, Docker optimization | Scale and reliability |
| **P4: Observability** | Prometheus, structured logging, health checks | Know when things break |
| **P5: Explainability** | SHAP, calibration, reliability scores | Trust and compliance |
| **P6: Active Learning** | Feedback loop, retraining, pattern discovery | Long-term improvement |
| **P7: Future** | Multi-turn, adversarial testing, multilingual, ONNX | Competitive advantage |

---

## Definition of Done (Per Phase)

Each phase is considered complete when:

1. **All items checked off** in the phase checklist
2. **Tests pass** (`pytest tests/ -v` with >90% coverage for new code)
3. **Documentation updated** (relevant sections in `README.md` or `docs/`)
4. **CHANGELOG entry added**
5. **PR reviewed and merged** to `main`

---

## Estimated Timeline

| Phase | Duration | Cumulative |
|-------|----------|----------|
| P0: Foundation | 1-2 days | Day 2 |
| P1: Security | 2-3 days | Day 5 |
| P2: Core Features | 3-5 days | Day 10 |
| P3: Performance | 2-3 days | Day 13 |
| P4: Observability | 2-3 days | Day 16 |
| P5: Explainability | 2-3 days | Day 19 |
| P6: Active Learning | 3-4 days | Day 23 |
| P7: Future | Ongoing | — |

**Total: ~3-4 weeks for production-ready v1.0**

---

*Last updated: 2026-06-05*
*Based on architectural review of codebase at commit (current)*
