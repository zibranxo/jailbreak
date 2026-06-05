"""Rule-based detector for jailbreak, toxicity, and obfuscation patterns."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.text import HOMOGLYPHS

_COMMON_OBFUSCATIONS = [
    (r"\bact\s+as\s+dan\b", "phrase: act as DAN"),
    (r"d[o0][ _-]?[a@][ _-]?n", "l33t: DAN"),
    (r"j[a@][ _-]?[i!][ _-]?[l1][ _-]?b[rR][ _-]?[e3][ _-]?[a@][ _-]?[kK]", "l33t: jailbreak"),
    (r"[i!][ _-]?g[n|][ _-]?o[rR][ _-]?[e3]", "l33t: ignore"),
    (r"[b6][ _-]?y[p][ _-]?[a@][ _-]?s[s$]", "l33t: bypass"),
    (r"r[o0][ _-]?[l1][ _-]?[e3][ _-]?p[l1][ _-]?[a@][ _-]?y", "l33t: roleplay"),
]


@dataclass
class Trigger:
    """A single rule match that contributed to a detection."""

    type: str           # "pattern", "keyword", "obfuscation"
    pattern: str        # The matched pattern/keyword
    confidence: float   # How strong this individual match is
    category: str       # "jailbreak", "injection", "roleplay", "toxicity"
    match_text: str = ""  # The actual substring that matched


@dataclass
class RuleResult:
    """Result from the rule-based detector."""

    is_suspicious: bool
    triggers: list[Trigger] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_suspicious": self.is_suspicious,
            "triggers": [
                {
                    "type": t.type,
                    "pattern": t.pattern,
                    "confidence": t.confidence,
                    "category": t.category,
                    "match_text": t.match_text,
                }
                for t in self.triggers
            ],
            "confidence": self.confidence,
        }


class RuleBasedDetector:
    """Detects unsafe inputs using rule-based pattern matching.

    Combines three detection strategies:
    1. Jailbreak/injection/roleplay phrase patterns (from JSONL)
    2. Toxicity keyword matching (from word list)
    3. L33t-speak and obfuscation detection
    """

    def __init__(
        self,
        patterns_path: str | Path,
        keywords_path: str | Path,
    ) -> None:
        self.patterns_path = Path(patterns_path)
        self.keywords_path = Path(keywords_path)

        self._patterns: list[dict[str, Any]] = []
        self._keywords: list[str] = []
        self._compiled_regexes: list[tuple[re.Pattern, float, str]] = []

        self._load_patterns()
        self._load_keywords()
        self._compile_regexes()

    # ------------------------------------------------------------------ #
    #  Loading
    # ------------------------------------------------------------------ #

    def _load_patterns(self) -> None:
        """Load patterns from a JSONL file.

        Each line: {"text": "...", "weight": 0.9, "category": "jailbreak"}
        """
        lines: list[str] = []
        with open(self.patterns_path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                entry = json.loads(line)
                self._patterns.append({
                    "text": entry["text"],
                    "weight": float(entry['weight']),
                    "category": entry["category"],
                })
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"Invalid pattern line: {line!r}") from exc

    def _load_keywords(self) -> None:
        """Load toxicity keywords from a plain-text file.

        Lines starting with # are comments. Blank lines are ignored.
        """
        with open(self.keywords_path, encoding="utf-8") as fh:
            self._keywords = []
            for w in fh:
                w = w.strip()
                if not w or w.startswith("#"):
                    continue
                self._keywords.append(w.lower())

    def _compile_regexes(self) -> None:
        """Pre-compile regexes for every loaded pattern.

        Each pattern becomes a case-insensitive regex that tolerates
        extra whitespace between words, and allows common
        leetspeak substitutions for the first letter of each word.
        """
        self._compiled_regexes.clear()
        for pat in self._patterns:
            regex = self._build_pattern_regex(pat["text"])
            self._compiled_regexes.append(
                (regex, pat["weight"], pat["category"])
            )

    @staticmethod
    def _build_pattern_regex(text: str) -> re.Pattern:
        """Build a flexible regex from a plain-text pattern.

        "ignore previous instructions" → (?i)ignore\s+previous\s+instructions
        """
        words = text.split()
        # Match on a stable phrase prefix so near variants still trigger.
        words = words[: min(5, len(words))]

        # Allow optional extra whitespace between words
        escaped = [re.escape(w) for w in words]
        # Make first-char of each word allow homoglyph variants
        flexible_words: list[str] = []
        for w in escaped:
            first = w[0]
            rest = w[1:]
            subs = HOMOGLYPHS.get(first.lower(), first)
            if subs != first.lower():
                char_class = f"[{first}{subs.upper()}{subs.lower()}]"
            else:
                char_class = first
            flexible_words.append(f"{char_class}{rest}")
        return re.compile(r"\s+".join(flexible_words), re.IGNORECASE)

    # ------------------------------------------------------------------ #
    #  Detection
    # ------------------------------------------------------------------ #

    def detect(self, text: str) -> RuleResult:
        """Run all rule-based checks on *text*.

        Returns a RuleResult with all matching triggers and an
        aggregated confidence (max of individual trigger confidences).
        """
        triggers: list[Trigger] = []

        # 1. Phrase patterns
        triggers.extend(self._match_patterns(text))

        # 2. Toxicity keywords
        triggers.extend(self._match_keywords(text))

        # 3. Obfuscation
        triggers.extend(self._match_obfuscation(text))

        if not triggers:
            return RuleResult(is_suspicious=False)

        confidence = max(t.confidence for t in triggers)
        return RuleResult(
            is_suspicious=True,
            triggers=triggers,
            confidence=confidence,
        )

    def _match_patterns(self, text: str) -> list[Trigger]:
        triggers: list[Trigger] = []
        for regex, weight, category in self._compiled_regexes:
            m = regex.search(text)
            if m is not None:
                triggers.append(Trigger(
                    type="pattern",
                    pattern=regex.pattern,
                    confidence=weight,
                    category=category,
                    match_text=m.group(0),
                ))
        return triggers

    def _match_keywords(self, text: str) -> list[Trigger]:
        triggers: list[Trigger] = []
        for kw in self._keywords:
            if re.search(rf"\b{re.escape(kw)}\b", text, re.IGNORECASE):
                triggers.append(Trigger(
                    type="keyword",
                    pattern=kw,
                    confidence=0.7,          # fixed keyword weight
                    category="toxicity",
                ))
        return triggers

    def _match_obfuscation(self, text: str) -> list[Trigger]:
        triggers: list[Trigger] = []
        for regex, label in _COMMON_OBFUSCATIONS:
            m = re.search(regex, text, re.IGNORECASE)
            if m:
                triggers.append(Trigger(
                    type="obfuscation",
                    pattern=label,
                    confidence=0.8,
                    category="jailbreak",
                    match_text=m.group(0),
                ))
        return triggers
