# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a hybrid LLM safety system designed to detect and classify unsafe interactions in large language models. The system categorizes prompts/responses into three classes: **safe**, **toxic**, or **jailbreak attempt**.

### Core Architecture

The system uses a **two-stage pipeline**:

1. **Fast Filtering Layer** - Rule-based heuristics and embedding similarity for initial screening
   - Pattern matching for known jailbreak prompts
   - Embedding-based similarity against unsafe pattern database
   - heuristic triggers (suspicious phrases, structural anomalies)

2. **Deep Classification Layer** - Transformer-based ML models for fine-grained categorization
   - Primary model: DistilBERT for semantic classification
   - Multi-class output: safe / toxic / jailbreak
   - Confidence score generation

### Key Features

- **Interpretability**: Provides explanations by highlighting contributing features (suspicious phrases, similarity matches, rule triggers)
- **Hybrid approach**: Combines rule-based + ML for robustness against adversarial inputs
- **Extensible**: Designed to detect prompt injection and roleplay-based jailbreak attacks

## Development Setup

### Environment

```bash
# Python 3.9+ recommended
pip install -r requirements.txt
```

Typical dependencies:
- torch / transformers (for DistilBERT)
- sentence-transformers (for embeddings)
- scikit-learn (for metrics, maybe additional models)
- numpy, pandas
- fastapi / flask (for API deployment)

### Project Structure (Expected)

```
.
├── src/
│   ├── detectors/
│   │   ├── rule_based.py      # Fast filtering layer
│   │   ├── embedding.py       # Embedding similarity checks
│   │   └── ml_classifier.py   # Transformer model wrapper
│   ├── classifiers/
│   │   └── safety_classifier.py  # Main orchestrator
│   ├── explainers/
│   │   └── interpretability.py   # Feature highlighting
│   └── utils/
│       └── helpers.py
├── data/
│   ├── patterns/              # Known unsafe patterns
│   └── models/               # Saved models
├── tests/
├── api/
│   └── server.py
├── notebooks/                # Exploratory analysis
└── config.yaml               # Configuration
```

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Run a single test
pytest tests/test_module.py::test_function -v

# Lint
black src/
flake8 src/

# Start API server
uvicorn api.server:app --reload

# Interactive testing
python -m src.classifiers.safety_classifier --interactive
```

## Model Training

```bash
# Train classifier
python -m src.training.train --data data/raw/ --output data/models/

# Evaluate
python -m src.training.evaluate --model data/models/ --test-data data/test/
```

## Adversarial Robustness Notes

- Consider adversarial training with paraphrased jailbreak examples
- Monitor for embedding obfuscation attacks (character substitution, encoding tricks)
- Regularly update pattern database with new jailbreak variants
- Ensemble multiple models may improve robustness

## Integration Guidelines

When integrating into production LLM applications:
- Run fast filtering first to reject clear violations
- Cache embedding computations for repeated attacks
- Log all detections with explanations for audit trail
- Implement rate limiting to prevent flooding
- Consider async processing for low-latency requirements

## Future Extensions

- Roleplay-based jailbreak detection
- Cross-lingual safety detection
- Real-time deployment with model quantization
- Continuous learning from false positives/negatives
