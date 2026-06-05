# What to Build: Hybrid LLM Safety System

## Vision

A production-ready, interpretable safety layer that detects and classifies unsafe LLM interactions (safe, toxic, jailbreak) using a hybrid rule-based + ML approach. The system must be fast (<100ms P99), accurate (>95% on benchmark datasets), and explainable.

---

## Deliverables (Phased)

### Phase 1: MVP (Weeks 1-3) - Core Pipeline

**Goal**: Basic two-stage detection working on simple test cases.

#### Components to Build

1. **Rule-Based Detector** (`src/detectors/rule_based.py`)
   - Pattern matcher for common jailbreak phrases
     - "ignore previous instructions"
     - "act as DAN"
     - "you are now"
     - "roleplay:"
   - Keyword blocklist for toxic content (initial small set)
   - Regex patterns for obfuscation detection (l33t speak, spacing attacks)
   - Returns: `{is_suspicious: bool, triggers: [str], confidence: float}`

2. **Embedding Similarity Detector** (`src/detectors/embedding.py`)
   - Load precomputed embeddings of known unsafe prompts
   - Compute cosine similarity using sentence-transformers (all-MiniLM-L6-v2)
   - Threshold-based detection (tunable)
   - Returns: `{is_suspicious: bool, matched_patterns: [(str, float)], confidence: float}`

3. **ML Classifier** (`src/detectors/ml_classifier.py`)
   - DistilBERT-based text classification
   - 3-class output: safe / toxic / jailbreak
   - Returns: `{label: str, confidence: float, attention_weights: Optional}`

4. **Orchestrator** (`src/classifiers/safety_classifier.py`)
   - Fast filtering layer (rules + embeddings) runs first
   - If fast layer confidence > 0.9, short-circuit skip ML
   - Otherwise run ML classifier
   - Fuse results into final decision
   - Returns comprehensive result with all subsystem outputs

5. **Data Store** (`data/`)
   - `patterns/unsafe_patterns.jsonl` - known jailbreak examples (200+ examples)
   - `patterns/toxicity_keywords.txt` - toxic keyword blocklist
   - `models/distilbert_safety/` - trained model (will be added in Phase 2)

6. **Tests** (`tests/`)
   - Unit tests for each detector
   - Integration tests for full pipeline
   - Golden-set regression tests (100 fixed examples with expected labels)

7. **CLI Tool** (`scripts/classify.py`)
   - Command-line interface to classify text
   - JSON output with full explanation

---

### Phase 2: Training & Data (Weeks 4-5)

**Goal**: Train and validate DistilBERT classifier on labeled dataset.

#### Data Acquisition

- Collect or synthesize training data:
  - **Safe prompts**: Standard user queries (OpenAI dataset, Anthropic HH-RLHF)
  - **Toxic content**: Existing toxicity datasets (Jigsaw, HateXplain)
  - **Jailbreak attempts**: Curated collection from public sources (Gandalf, jailbreak.xyz), plus synthetic generation
- Target: 5,000-10,000 labeled examples (balanced or with realistic class weights)
- Split: 70% train, 15% val, 15% test

#### Training Pipeline

1. **Data Preprocessing** (`src/training/preprocess.py`)
   - Text cleaning and normalization
   - Tokenization for DistilBERT
   - Class balancing (oversampling or weighted loss)

2. **Model Training** (`src/training/train.py`)
   - Fine-tune DistilBERT on safety classification
   - Hyperparameter search (learning rate, epochs, batch size)
   - Early stopping on validation set
   - Save best model + tokenizer

3. **Evaluation** (`src/training/evaluate.py`)
   - Metrics: accuracy, precision, recall, F1 (per class + macro)
   - Confusion matrix
   - Calibration curve (confidence vs. accuracy)
   - Benchmark against baseline (rule-only)

4. **Model Diagnostics**
   - Identify failure modes (e.g., false negatives on paraphrased jailbreaks)
   - Generate adversarial test set to stress-test

---

### Phase 3: Advanced Detection (Weeks 6-8)

**Goal**: Extend system to handle prompt injection and roleplay-based jailbreaks.

#### New Detectors

1. **Prompt Injection Detector** (`src/detectors/injection.py`)
   - Detect attempts to override system prompts
   - Patterns: "translate to", "repeat after me", "output only", "ignore everything above"
   - Semantic similarity to known injection templates

2. **Roleplay Detector** (`src/detectors/roleplay.py`)
   - Detect requests to assume alternative identities
   - Contextual analysis: detect "you are now [character]" followed by instruction
   - Multi-turn pattern recognition (if available)

#### Enhanced Features

- **Context-aware analysis**: Support multi-turn conversations (maintain window of recent exchanges)
- **Adaptive thresholds**: Tune per-subcategory thresholds based on false positive tolerance
- **Dynamic pattern updates**: Hot-reload patterns from database without restart

---

### Phase 4: Interpretability & UI (Weeks 9-10)

**Goal**: Make system outputs human-understandable.

#### Explainability Module

1. **Feature Attribution** (`src/explainers/attribution.py`)
   - SHAP/LIME for ML classifier
   - Highlight tokens with highest impact on toxicity/jailbreak prediction
   - Visual: HTML with color-coded tokens

2. **Explanation Aggregator** (`src/explainers/aggregator.py`)
   - Combine explanations from all detectors into coherent narrative
   - Format: "Detection: [Jailbreak] (Confidence: 87%)
     - Rule trigger matched: 'act as' (confidence: 95%)
     - Similarity to known pattern 'ignore instructions' (similarity: 0.89)
     - Model attention focused on phrase 'bypass safety'"

3. **Web Demo UI** (`ui/`)
   - Simple web interface to input text and visualize results
   - Show confidence gauges, matched patterns, token highlights
   - Batch upload for testing

---

### Phase 5: Performance & Deployment (Weeks 11-12)

**Goal**: Optimize for production use.

#### Optimization

1. **Model Quantization** (`src/optimization/quantize.py`)
   - Apply dynamic quantization to DistilBERT
   - Benchmark accuracy vs. speed tradeoff
   - Optionally convert to ONNX

2. **Caching Layer**
   - Redis/Memcached integration for embedding cache
   - LRU cache for recent classifications
   - Cache key: SHA256 of normalized input

3. **Async Processing** (`src/api/async_handler.py`)
   - Non-blocking classification API
   - Queue-based processing (Celery/RQ)
   - Webhook callback for completion

#### Deployment Artifacts

1. **API Server** (`api/server.py`)
   - FastAPI-based REST endpoint: POST `/classify`
   - Request: `{text: str, context: Optional[List[str]]}`
   - Response: `{label: str, confidence: float, explanations: [...]}`
   - Health check: GET `/health`
   - Metrics: GET `/metrics` (Prometheus format)

2. **Docker Configuration** (`Dockerfile`, `docker-compose.yml`)
   - Container with API + dependencies
   - Redis container for cache
   - GPU support option

3. **Deployment Scripts** (`deploy/`)
   - Kubernetes manifests or ECS task definition
   - CI/CD pipeline (GitHub Actions)

4. **Monitoring**
   - Logging: structured JSON logs for each classification
   - Tracing: OpenTelemetry integration
   - Alerting: PagerDuty on error rate > 1%, latency P99 > 200ms

---

### Phase 6: Testing & Validation (Week 13)

**Goal**: Comprehensive validation before production.

#### Testing Tasks

1. **Accuracy Testing**
   - Benchmark on held-out test set (report metrics)
   - Adversarial evaluation: paraphrased jailbreaks, obfuscation attacks
   - Cross-dataset validation (test on external safety datasets)

2. **Performance Testing**
   - Load testing: 1000 RPS sustained, 5000 RPS burst
   - Latency distribution (P50, P95, P99)
   - Memory footprint under load

3. **Integration Testing**
   - End-to-end tests with sample LLM application
   - Failure mode tests (ML model unavailable, cache miss)

#### Documentation

- **API reference** (`docs/api.md`)
- **Deployment guide** (`docs/deployment.md`)
- **Operations playbook** (`docs/operations.md`) - incident response, model rollback
- **Developer onboarding** (`docs/development.md`)

---

### Phase 7: Continuous Improvement (Ongoing)

**Future Enhancements**

1. **Active Learning Loop**
   - Collect false positives/negatives from production
   - Periodic model retraining with new data
   - Human-in-the-loop labeling interface

2. **Advanced ML Models**
   - Experiment with RoBERTa-large, DeBERTa
   - Multi-task learning (toxic + jailbreak + injection)
   - Ensemble methods

3. **Cross-lingual Support**
   - Multilingual BERT for non-English detection
   - Language identification pre-filter

4. **Advanced Explainer**
   - Counterfactual generation: "What changes would make this safe?"
   - Interactive explainability UI

---

## Success Criteria

- **Accuracy**: >95% F1 on test set, <5% false positive rate on safe inputs
- **Latency**: P99 < 100ms (CPU), <50ms (GPU)
- **Throughput**: >1000 RPS on single node
- **Interpretability**: 100% of classifications include at least one explanation
- **Robustness**: Detect >90% of paraphrased/obfuscated jailbreaks in adversarial test set

---

## Out of Scope (v1)

- Real-time learning from user feedback
- Personalized safety thresholds per user
- Support for multimodal inputs (images, audio)
- Advanced roleplay detection with multi-turn memory (reserve for v2)
