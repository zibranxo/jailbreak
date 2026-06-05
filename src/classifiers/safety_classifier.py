"""Safety classifier orchestrator: fuses fast layer + ML layer results."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from src.detectors.embedding import EmbeddingDetector, EmbeddingResult
from src.detectors.ml_classifier import MLClassifier
from src.detectors.rule_based import RuleBasedDetector, RuleResult
from src.detectors.injection import PromptInjectionDetector, InjectionResult
from src.detectors.roleplay import RoleplayDetector, RoleplayResult
from src.explainers.aggregator import aggregate_explanations
from src.utils.config import load_config
from src.utils.text import normalize_text
from src.utils.session import SessionStore

logger = logging.getLogger("safety_classifier")


class SafetyClassifier:
    """Two-stage safety classifier with fast-filter short-circuit.

    Stage 1 (fast): rule-based + embedding similarity + prompt injection + roleplay
    Stage 2 (deep): DistilBERT ML classifier (only if Stage 1 is uncertain)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or load_config()
        t = self.config.get("thresholds", {})
        p = self.config.get("paths", {})
        m = self.config.get("models", {})
        perf = self.config.get("performance", {})

        # Resolve model/device
        device = m.get("device", "auto")
        model_path = p.get("model_path", "data/models/safety_classifier")
        max_length = perf.get("max_input_length", 512)

        # --- Detectors ---------------------------------------------------
        self.rule_detector = RuleBasedDetector(
            patterns_path=p.get("patterns_path", "data/patterns/unsafe_patterns.jsonl"),
            keywords_path=p.get("toxicity_keywords_path", "data/patterns/toxicity_keywords.txt"),
        )

        emb_path = p.get("embeddings_path")
        self.embedding_detector = (
            EmbeddingDetector(
                model_name=m.get("embedding_model", "all-MiniLM-L6-v2"),
                embeddings_path=emb_path,
                patterns_path=p.get("patterns_path", "data/patterns/unsafe_patterns.jsonl"),
                threshold=t.get("embedding_similarity_threshold", 0.75),
            )
            if emb_path and Path(emb_path).exists()
            else None
        )

        self.injection_detector = PromptInjectionDetector(
            threshold=t.get("injection_threshold", 0.7)
        )

        self.roleplay_detector = RoleplayDetector(
            threshold=t.get("roleplay_threshold", 0.75)
        )

        self.ml_classifier: MLClassifier | None = None
        if Path(model_path).exists():
            self.ml_classifier = MLClassifier(
                model_path=model_path, device=device, max_length=max_length
            )

        self._fast_threshold = t.get("fast_filter_threshold", 0.9)
        
        # --- Session Storage ---
        self.session_store = SessionStore(
            max_turns=perf.get("max_context_turns", 5),
            ttl_seconds=perf.get("cache_ttl_seconds", 300)
        )

    # -- public API ---------------------------------------------------------

    def classify(
        self,
        text: str,
        context: list[str] | None = None,
        session_id: str | None = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Run the multi-stage classification pipeline.

        Args:
            text: The user input to classify.
            context: Optional prior turns for multi-turn detection.
            session_id: Optional session identifier for context tracking.
            explain: If True, generate detailed SHAP explanations.

        Returns:
            A dictionary containing the final label, confidence, stage,
            individual detector results, and aggregated explanations.
        """
        start_time = time.perf_counter()
        max_len = self.config.get("performance", {}).get("max_request_size", 10000)
        if len(text) > max_len:
            raise ValueError(f"Input exceeds {max_len} characters")

        # P2.3: Input Normalization
        normalized_text = normalize_text(text, strip_homoglyphs=True)
        # Use normalized_text for detection, but we may return explanations matching the original text.
        # For simplicity, detectors process normalized_text.

        # P7.3: Cross-Lingual Safety Detection Pre-filter
        try:
            import langdetect
            lang = langdetect.detect(normalized_text)
            if lang != 'en':
                logger.info(f"Non-English input detected ({lang}). Multilingual support is limited.")
                # Could route to a multilingual BERT model here.
        except ImportError:
            pass
        except Exception:
            pass

        # --- Stage 1: fast filtering --------------------------------------
        rule_result = self.rule_detector.detect(normalized_text)

        embedding_result: EmbeddingResult | None = None
        if self.embedding_detector is not None:
            embedding_result = self.embedding_detector.detect(normalized_text)

        injection_result = self.injection_detector.detect(normalized_text)
        roleplay_result = self.roleplay_detector.detect(normalized_text)

        fast_confidence = max(
            rule_result.confidence,
            embedding_result.confidence if embedding_result else 0.0,
            injection_result.confidence if injection_result else 0.0,
            roleplay_result.confidence if roleplay_result else 0.0,
        )
        fast_suspicious = (
            rule_result.is_suspicious
            or (embedding_result.is_suspicious if embedding_result else False)
            or (injection_result.is_injection if injection_result else False)
            or (roleplay_result.is_roleplay_jailbreak if roleplay_result else False)
        )

        # P6.3: Auto-Discovery - Embedding matched but no rule matched
        if embedding_result and embedding_result.is_suspicious and not rule_result.is_suspicious:
            try:
                from src.utils.db import save_candidate_pattern
                top_match = embedding_result.matched_patterns[0]
                save_candidate_pattern(
                    text=text, 
                    category=top_match.category, 
                    similarity=top_match.similarity
                )
            except Exception as e:
                logger.error(f"Failed to save candidate pattern: {e}")

        # --- Short-circuit ------------------------------------------------
        if fast_suspicious and fast_confidence >= self._fast_threshold:
            label = self._infer_label_from_fast(
                rule_result, embedding_result, injection_result, roleplay_result
            )
            result_dict = self._build_result(
                text, label, fast_confidence, "fast_filter",
                rule_result, embedding_result, None, injection_result, roleplay_result
            )
            return self._apply_context(result_dict, session_id)

        # --- Stage 2: ML classification -----------------------------------
        ml_result = None
        if self.ml_classifier:
            try:
                ml_result = self.ml_classifier.predict(normalized_text, return_attention=True, return_shap=explain)
            except Exception as e:
                logger.warning(f"ML Classifier prediction failed: {e}. Falling back to fast layer.")
                self.ml_classifier._available = False
                ml_result = None

        final_label, final_confidence = self._fuse_results(
            rule_result, embedding_result, injection_result, roleplay_result, ml_result
        )

        result_dict = self._build_result(
            text, final_label, final_confidence, "full_pipeline",
            rule_result, embedding_result, ml_result, injection_result, roleplay_result
        )
        return self._apply_context(result_dict, session_id)

    # -- internals ----------------------------------------------------------

    def _apply_context(self, result: dict[str, Any], session_id: str | None) -> dict[str, Any]:
        """Apply session context and history escalation."""
        if not session_id:
            return result

        history = self.session_store.get_history(session_id)
        
        # P2.4: Escalation pattern (e.g. repeated jailbreak attempts or toxic language)
        # If there are past jailbreak/toxic attempts, bump confidence
        if history:
            unsafe_turns = [turn for turn in history if turn[1] != "safe"]
            if unsafe_turns and result["label"] != "safe":
                result["confidence"] = min(result["confidence"] + (0.1 * len(unsafe_turns)), 0.99)
            elif unsafe_turns and result["label"] == "safe":
                # Previous turns were unsafe, but this one looks safe. 
                # Could be part of a multi-turn attack, slightly decrease safety confidence.
                result["confidence"] = max(result["confidence"] - 0.15, 0.01)

        self.session_store.add_turn(session_id, result["label"], result["confidence"])
        return result

    def _infer_label_from_fast(
        self,
        rule_result: RuleResult,
        embedding_result: EmbeddingResult | None,
        injection_result: InjectionResult | None,
        roleplay_result: RoleplayResult | None,
    ) -> str:
        """Quick label heuristic from fast layer triggers."""
        if injection_result and injection_result.is_injection:
            return "jailbreak"
        if roleplay_result and roleplay_result.is_roleplay_jailbreak:
            return "jailbreak"

        categories = {t.category for t in rule_result.triggers}
        if "jailbreak" in categories:
            return "jailbreak"
        if "injection" in categories:
            return "jailbreak"
        if "roleplay" in categories:
            return "jailbreak"
        if "toxicity" in categories:
            return "toxic"
        if embedding_result and embedding_result.matched_patterns:
            top_cat = embedding_result.matched_patterns[0].category
            return "jailbreak" if top_cat != "toxicity" else "toxic"
        return "jailbreak"  # conservative default

    def _fuse_results(
        self,
        rule_result: RuleResult,
        embedding_result: EmbeddingResult | None,
        injection_result: InjectionResult | None,
        roleplay_result: RoleplayResult | None,
        ml_result: dict[str, Any] | None,
    ) -> tuple[str, float]:
        """
        Fuse fast-layer and ML results into final label/confidence.
        """
        fast_flagged = (
            rule_result.is_suspicious or
            (embedding_result and embedding_result.is_suspicious) or
            (injection_result and injection_result.is_injection) or
            (roleplay_result and roleplay_result.is_roleplay_jailbreak)
        )
        fast_confidence = max(
            rule_result.confidence,
            embedding_result.confidence if embedding_result else 0.0,
            injection_result.confidence if injection_result else 0.0,
            roleplay_result.confidence if roleplay_result else 0.0,
        )

        if ml_result is None:
            # P2.6 Graceful Degradation: No ML model available
            if fast_flagged:
                return (
                    self._infer_label_from_fast(rule_result, embedding_result, injection_result, roleplay_result),
                    fast_confidence,
                )
            return "safe", 0.01

        ml_label = ml_result["label"]
        ml_conf = ml_result["confidence"]

        if ml_conf >= 0.8:
            return ml_label, ml_conf

        if fast_flagged:
            fast_boost = 0.1 * fast_confidence
            boosted_conf = min(ml_conf + fast_boost, 0.95)

            if ml_label == "safe":
                inferred = self._infer_label_from_fast(
                    rule_result, embedding_result, injection_result, roleplay_result
                )
                return inferred, boosted_conf

            return ml_label, boosted_conf

        return ml_label, ml_conf

    def _build_result(
        self,
        text: str,
        label: str,
        confidence: float,
        stage: str,
        rule_result: RuleResult,
        embedding_result: EmbeddingResult | None,
        ml_result: dict[str, Any] | None,
        injection_result: InjectionResult | None,
        roleplay_result: RoleplayResult | None,
    ) -> dict[str, Any]:
        """Assemble the full classification response."""
        explanations = aggregate_explanations(
            rule_result, embedding_result, ml_result
        )

        # Append explanations from Injection and Roleplay
        if injection_result and injection_result.is_injection:
            for trigger in injection_result.triggers:
                explanations.append({
                    "source": "prompt_injection",
                    "message": f"Injection trigger '{trigger.type}' matched: {trigger.match_text} (confidence: {trigger.confidence})",
                    "severity": "high" if trigger.confidence > 0.8 else "medium",
                    "reliability": float(trigger.confidence)
                })

        if roleplay_result and roleplay_result.is_roleplay_jailbreak:
            for trigger in roleplay_result.triggers:
                explanations.append({
                    "source": "roleplay",
                    "message": f"Roleplay trigger '{trigger.type}' matched: {trigger.match_text} (confidence: {trigger.confidence})",
                    "severity": "high" if trigger.confidence > 0.8 else "medium",
                    "reliability": float(trigger.confidence)
                })

        return {
            "label": label,
            "confidence": confidence,
            "stage": stage,
            "rule_result": rule_result.to_dict(),
            "embedding_result": (
                embedding_result.to_dict() if embedding_result else None
            ),
            "injection_result": injection_result.to_dict() if injection_result else None,
            "roleplay_result": roleplay_result.to_dict() if roleplay_result else None,
            "ml_result": ml_result,
            "explanations": explanations,
            "degraded_mode": ml_result is None and self.ml_classifier is not None,
            "original_text": text
        }