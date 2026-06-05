# Alert Runbook

This runbook defines the alerting thresholds and standard operating procedures for resolving common issues with the LLM Safety Classifier.

## Alerting Thresholds

### 1. High Error Rate
**Trigger**: Error rate > 1% for >2 minutes.
**Severity**: High (PagerDuty)
**Possible Causes**:
- Invalid API Keys being sent by a high-traffic client.
- Malformed requests (e.g. failing validation).
- Missing Redis instance causing intermittent cache failures (if not failing gracefully).
**Action**:
- Check `/metrics` or structured logs. Filter by `level=ERROR`.
- Identify if the errors are 4xx (client-side) or 5xx (server-side).
- If 5xx, inspect stack traces.

### 2. High Latency
**Trigger**: P99 latency > 200ms for >5 minutes.
**Severity**: Medium (Slack Alert)
**Possible Causes**:
- Redis cache eviction rate is high (cache thrashing).
- Model inference bottleneck due to large batch sizes.
- FAISS index is not installed, falling back to slow NumPy linear scan.
**Action**:
- Check `safety_classification_latency_seconds` in Prometheus.
- Ensure `faiss-cpu` or `faiss-gpu` is installed.
- Check CPU/GPU utilization on the inference nodes.

### 3. ML Model Unavailable
**Trigger**: ML model not loaded after startup (Service running in Degraded Mode).
**Severity**: High (PagerDuty)
**Possible Causes**:
- Hugging Face hub is down or unreachable during startup (if lazy loading from remote).
- Disk space is full preventing model download.
- Pre-baking script in Docker failed.
**Action**:
- Check `GET /health/detailed`. If `ml_model_available` is `false`, the classifier is running via fast-filter only.
- Review startup logs.
- Attempt to restart the container to trigger the model download script.

### 4. Redis Connection Lost
**Trigger**: Redis connection lost / timeout.
**Severity**: Low (Log + degrade gracefully to LRU cache)
**Possible Causes**:
- Redis server restarted or network partition.
**Action**:
- Verify Redis health.
- The service will automatically fall back to local LRU caching.
