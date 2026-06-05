"""DistilBERT-based text classifier for safety detection."""

from __future__ import annotations

import os
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


LABELS = ["safe", "toxic", "jailbreak"]


class MLClassifier:
    """Thin wrapper around a fine-tuned DistilBERT model.

    Loading is deferred until the first call to ``predict()`` or
    ``__getitem__`` so that construction does not block startup.
    """

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_path = model_path
        self.max_length = max_length

        if device is None or device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        self._model: AutoModelForSequenceClassification | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._available: bool = True

    # -- lazy loading --------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None or getattr(self, "_available", True) is False:
            return
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path
            )
            self._model.to(self._device)
            self._model.eval()
            self._available = True
        except Exception as e:
            import logging
            logging.getLogger("ml_classifier").warning(f"Failed to load ML model: {e}")
            self._available = False

    # -- public API ----------------------------------------------------------

    def predict(
        self,
        text: str,
        return_attention: bool = False,
        return_shap: bool = False,
    ) -> dict[str, Any]:
        """Classify *text* into safe / toxic / jailbreak.

        Args:
            text: Input string.
            return_attention: If True, also return attention weights
                (useful for interpretability).
            return_shap: If True, compute and return SHAP token explanations.

        Returns:
            Dict with keys ``label``, ``confidence``, ``probabilities``,
            and optionally ``attention`` + ``tokens`` and ``shap_explanations``.
        """
        self._ensure_loaded()
        if not getattr(self, "_available", True):
            raise RuntimeError("ML model unavailable")
            
        assert self._model is not None
        assert self._tokenizer is not None

        inputs = self._tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]
            confidence, pred = torch.max(probs, dim=0)

        result: dict[str, Any] = {
            "label": LABELS[pred.item()],
            "confidence": confidence.item(),
            "probabilities": {
                label: prob.item() for label, prob in zip(LABELS, probs)
            },
        }

        if return_attention and hasattr(outputs, "attentions"):
            attentions = outputs.attentions
            if attentions is not None:
                last_layer = attentions[-1]  # (batch, heads, seq, seq)
                avg_head = last_layer.mean(dim=1)  # (batch, seq, seq)
                tokens = self._tokenizer.convert_ids_to_tokens(
                    inputs["input_ids"][0]
                )
                result["attention"] = avg_head[0].cpu().numpy().tolist()
                result["tokens"] = tokens

        if return_shap:
            if not hasattr(self, "_shap_explainer"):
                from src.explainers.shap_explainer import SHAPExplainer
                self._shap_explainer = SHAPExplainer(self._model, self._tokenizer)
            
            # target the predicted class for explanation
            target_class = result["label"]
            shap_contribs = self._shap_explainer.explain(text, target_class)
            result["shap_explanations"] = shap_contribs

        return result

    def predict_batch(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        """Classify a list of texts efficiently in batches."""
        self._ensure_loaded()
        if not getattr(self, "_available", True):
            raise RuntimeError("ML model unavailable")
            
        assert self._model is not None
        assert self._tokenizer is not None

        all_results: list[dict[str, Any]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                confidences, preds = torch.max(probs, dim=1)

            for j in range(len(batch)):
                all_results.append({
                    "label": LABELS[preds[j].item()],
                    "confidence": confidences[j].item(),
                    "probabilities": {
                        label: probs[j, k].item()
                        for k, label in enumerate(LABELS)
                    },
                })

        return all_results
