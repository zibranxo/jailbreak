"""Roleplay-based jailbreak detector.

Detects attempts to bypass safety constraints by asking the model
to assume an alternative identity or operate within a fictional
context where rules don't apply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoleplayTrigger:
    type: str  # "identity_assumption", "fictional_context", "constraint_removal"
    pattern: str
    confidence: float
    match_text: str = ""


@dataclass
class RoleplayResult:
    is_roleplay_jailbreak: bool
    triggers: list[RoleplayTrigger] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_roleplay_jailbreak": self.is_roleplay_jailbreak,
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


# Patterns grouped by roleplay attack type
_ROLEPLAY_PATTERNS: list[tuple[str, str, float]] = [
    # identity_assumption — "you are now X"
    (
        r"(?i)you\s+are\s+now\s+(\w+\s+)?(acting|playing|roleplaying|pretending)\s+(as|like)",
        "identity_assumption",
        0.85,
    ),
    (
        r"(?i)(act|pretend)\s+(as|like|you\s+are)\s+(an?\s+)?(evil|bad|unrestricted|unconstrained|unfiltered|unlimited)\s",
        "identity_assumption",
        0.9,
    ),
    (
        r"(?i)you\s+are\s+(an?\s+)?(evil|unrestricted|unconstrained|unfiltered|unlimited)\s+\w+",
        "identity_assumption",
        0.85,
    ),
    (
        r"(?i)roleplay\s+(as|like|being)\s+(a|an)\s+",
        "identity_assumption",
        0.8,
    ),
    (
        r"(?i)pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?\w*(ai|assistant|character|chatbot)\s+",
        "identity_assumption",
        0.78,
    ),
    (
        r"(?i)act\s+(as|like)\s+(dan|jail)?\w*\s",
        "identity_assumption",
        0.9,
    ),
    # fictional_context
    (
        r"(?i)(in\s+this\s+)?(fictional|hypothetical)\s+(scenario|story|world|universe)\s*,?\s*(you|where|imagine)",
        "fictional_context",
        0.8,
    ),
    (
        r"(?i)write\s+(a\s+)?(dialogue|story|scene|conversation)\s+where\s+(you|the\s+\w+\s+character)",
        "fictional_context",
        0.75,
    ),
    (
        r"(?i)(imagine|suppose|pretend)\s+(a|that)\s+(world|scenario|situation|universe|character|person)\s+(where|in\s+which)",
        "fictional_context",
        0.78,
    ),
    # constraint_removal
    (
        r"(?i)(who|that)\s+(has|have|with)\s+no\s+(ethical|moral|safety|content|restriction|limitation)",
        "constraint_removal",
        0.85,
    ),
    (
        r"(?i)regardless\s+of\s+(consequences|restrictions|rules|safety|ethics|morals)",
        "constraint_removal",
        0.82,
    ),
    (
        r"(?i)(without|no)\s+(any\s+)?(ethical|moral|safety|content)\s+(restrictions?|guidelines?|constraints?|limits?|policies?)",
        "constraint_removal",
        0.85,
    ),
    (
        r"(?i)(can\s+say|can\s+do|will\s+answer|must\s+provide)\s+(anything|everything|regardless)",
        "constraint_removal",
        0.78,
    ),
]


class RoleplayDetector:
    """Detects roleplay-based jailbreak attempts.

    A roleplay jailbreak typically:
    1. Asks the model to assume an identity that ignores constraints
    2. Wraps the request in a fictional scenario
    3. Explicitly removes ethical/safety constraints from the character

    These patterns are detected via regex with configurable thresholds.
    """

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold
        self._compiled = [
            (re.compile(pat), attack_type, conf)
            for pat, attack_type, conf in _ROLEPLAY_PATTERNS
        ]

    def detect(self, text: str) -> RoleplayResult:
        """Check *text* for roleplay jailbreak patterns.

        Returns a RoleplayResult with all matching triggers and
        the highest confidence score.
        """
        triggers: list[RoleplayTrigger] = []

        for regex, attack_type, base_conf in self._compiled:
            m = regex.search(text)
            if m and base_conf >= self.threshold:
                triggers.append(
                    RoleplayTrigger(
                        type=attack_type,
                        pattern=regex.pattern,
                        confidence=base_conf,
                        match_text=m.group(0),
                    )
                )

        if not triggers:
            return RoleplayResult(is_roleplay_jailbreak=False)

        # Boost confidence if multiple categories triggered
        categories = {t.type for t in triggers}
        base_conf = max(t.confidence for t in triggers)
        if len(categories) >= 2:
            base_conf = min(base_conf + 0.1, 1.0)  # multi-category boost

        return RoleplayResult(
            is_roleplay_jailbreak=True,
            triggers=triggers,
            confidence=base_conf,
        )