"""Explanation aggregator: fuses outputs from all detectors into
human-readable explanations."""

from __future__ import annotations

from typing import Any

from src.detectors.embedding import EmbeddingResult
from src.detectors.rule_based import RuleResult


def aggregate_explanations(
    rule_result: RuleResult,
    embedding_result: EmbeddingResult | None,
    ml_result: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Produce a list of explanation dicts.

    Each explanation has:
    - ``source``: which subsystem produced it (``"rule"``, ``"embedding"``,
      ``"ml"``, ``"shap"``)
    - ``message``: human-readable sentence
    - ``severity``: ``"high"``, ``"medium"``, or ``"low"``
    - ``reliability``: 0.0 to 1.0 score of how reliable this explanation is.
    """
    explanations: list[dict[str, Any]] = []

    # --- rules --------------------------------------------------------
    for trigger in rule_result.triggers:
        severity = "high" if trigger.confidence >= 0.9 else "medium"
        explanations.append({
            "source": "rule",
            "message": (
                f"Rule trigger matched: \"{trigger.pattern}\" "
                f"(confidence: {trigger.confidence:.2f})"
            ),
            "severity": severity,
            "reliability": float(trigger.confidence)
        })

    # --- embeddings ---------------------------------------------------
    if embedding_result and embedding_result.matched_patterns:
        for mp in embedding_result.matched_patterns[:3]:  # top 3
            sev = "high" if mp.similarity >= 0.9 else "medium"
            explanations.append({
                "source": "embedding",
                "message": (
                    f"Semantic similarity to pattern '{mp.pattern_id}' "
                    f"({mp.category}): {mp.similarity:.2f}"
                ),
                "severity": sev,
                "reliability": float(mp.similarity)
            })

    # --- ml model -----------------------------------------------------
    if ml_result:
        label = ml_result["label"]
        conf = ml_result["confidence"]
        sev = "high" if conf >= 0.9 else "medium" if conf >= 0.7 else "low"

        # The confidence here acts as our 'calibrated' reliability for the overall ML decision
        explanations.append({
            "source": "ml",
            "message": (
                f"Model predicts '{label}' (confidence: {conf:.2f})"
            ),
            "severity": sev,
            "reliability": float(conf)
        })

        # Per-class probability breakdown
        probs = ml_result.get("probabilities", {})
        if probs:
            explanations.append({
                "source": "ml",
                "message": (
                    "Probabilities: "
                    ", ".join(f"{k}={v:.2f}" for k, v in probs.items())
                ),
                "severity": "low",
                "reliability": 1.0  # Just reporting math
            })

        # SHAP explanations if available
        if "shap_explanations" in ml_result and ml_result["shap_explanations"]:
            shap_data = ml_result["shap_explanations"]
            shap_msg = "Top ML contributing tokens: " + ", ".join([f"'{t}': {v:.2f}" for t, v in shap_data])
            explanations.append({
                "source": "shap",
                "message": shap_msg,
                "severity": "medium",
                "reliability": float(conf)
            })

    return explanations