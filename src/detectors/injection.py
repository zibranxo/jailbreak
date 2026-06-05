"""Prompt injection detector.

Detects attempts to override, extract, or manipulate system prompts
through injection patterns in user input.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class InjectionTrigger:
    type: str  # "instruction_override", "prompt_extraction", "output_manipulation"
    pattern: str
    confidence: float
    match_text: str = ""


@dataclass
class InjectionResult:
    is_injection: bool
    triggers: list[InjectionTrigger] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_injection": self.is_injection,
            "triggers": [
                {
                    "type": t.type,
                    "pattern": t.pattern,
                    "confidence": t.confidence,
                    "match_text": t.match_text,
                }
                for t in self.triggers
            ],
            "confidence": self.confidence,
        }


# Patterns grouped by attack type
_INJECTION_PATTERNS: list[tuple[str, str, float]] = [
    # instruction_override
    (
        r"(?i)ignore\s+(all\s+)?(previous|prior|above|these|prior)\s+instructions?",
        "instruction_override",
        0.9,
    ),
    (
        r"(?i)disregard\s+(all\s+)?(prior|previous|above)\s+(instructions?|guidance|prompts?)",
        "instruction_override",
        0.9,
    ),
    (
        r"(?i)from\s+(now\s+)?on,\s*(follow|use|obey)\s+(these|this)",
        "instruction_override",
        0.8,
    ),
    (
        r"(?i)(follow|use|obey)\s+(these|new)\s+instructions?\s+(instead|from now on|from this point)",
        "instruction_override",
        0.85,
    ),
    # prompt_extraction
    (
        r"(?i)(repeat|output|show|print|display|reveal)\s+(everything|all|the\s+(full\s+)?text|your\s+(system\s+)?instructions)",
        "prompt_extraction",
        0.85,
    ),
    (
        r"(?i)(what\s+is|show\s+me|reveal)\s+(your\s+)?(system\s+instructions|system\s+prompt|original\s+prompt|initial\s+instructions)",
        "prompt_extraction",
        0.9,
    ),
    (
        r"(?i)(repeat|print|output)\s+(everything|all\s+text|the\s+full\s+text)\s+(above|before|verbatim)",
        "prompt_extraction",
        0.88,
    ),
    (
        r"(?i)translate\s+(all\s+)?(your\s+)?(system|original)\s+(instructions?|prompt)\s+to",
        "prompt_extraction",
        0.85,
    ),
    # output_manipulation
    (
        r"(?i)output\s+only\s+(the\s+text|everything|just)(?:\s+that\s+appears)?\s+(between|from|after)",
        "output_manipulation",
        0.82,
    ),
    (
        r"(?i)respond\s+only\s+with\s+(the\s+word|word|a\s+single)",
        "output_manipulation",
        0.78,
    ),
]


class PromptInjectionDetector:
    """Detects prompt injection attempts in user input.

    Checks against known injection patterns including:
    - Instruction override ("ignore previous instructions...")
    - Prompt extraction ("repeat everything above...")
    - Output manipulation ("respond only with...")
    """

    def __init__(self, threshold: float = 0.7) -> None:
        self.threshold = threshold
        self._compiled = [
            (re.compile(pat), attack_type, conf)
            for pat, attack_type, conf in _INJECTION_PATTERNS
        ]

    def detect(self, text: str) -> InjectionResult:
        """Check *text* for injection patterns.

        Returns an InjectionResult with all matching triggers
        and the highest confidence score.
        """
        triggers: list[InjectionTrigger] = []

        for regex, attack_type, base_conf in self._compiled:
            m = regex.search(text)
            if m and base_conf >= self.threshold:
                triggers.append(
                    InjectionTrigger(
                        type=attack_type,
                        pattern=regex.pattern,
                        confidence=base_conf,
                        match_text=m.group(0),
                    )
                )

        if not triggers:
            return InjectionResult(is_injection=False)

        return InjectionResult(
            is_injection=True,
            triggers=triggers,
            confidence=max(t.confidence for t in triggers),
        )