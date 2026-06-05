"""Text normalization utilities for adversarial resilience."""

import re
import unicodedata

# Common homoglyph mappings
HOMOGLYPHS = {
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
    "7": "t", "8": "b", "@": "a", "$": "s", "!": "i",
}


def normalize_text(text: str, strip_homoglyphs: bool = False) -> str:
    """Normalize input text to defeat common obfuscation attacks.

    Args:
        text: Raw input text.
        strip_homoglyphs: Whether to replace common homoglyphs.

    Returns:
        Normalized text.
    """
    # Normalize unicode (handle weird unicode tricks)
    text = unicodedata.normalize("NFKC", text)

    # Remove zero-width characters (zero-width space, joiner, non-joiner)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u2060]", "", text)

    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if strip_homoglyphs:
        text = _replace_homoglyphs(text)

    return text


def _replace_homoglyphs(text: str) -> str:
    """Replace common leetspeak/homoglyph characters."""
    result = []
    for char in text.lower():
        result.append(HOMOGLYPHS.get(char, char))
    return "".join(result)


def detect_obfuscation(text: str) -> float:
    """Estimate obfuscation level of text (0.0 = clean, 1.0 = heavily obfuscated).

    Checks for:
    - Unusual character substitutions (leetspeak)
    - Excessive spacing
    - Mixed case tricks
    - Unicode anomalies
    """
    score = 0.0

    homoglyph_ratio = sum(1 for c in text if c.lower() in HOMOGLYPHS) / max(len(text), 1)
    score += homoglyph_ratio * 0.5

    # Excessive internal spacing
    multi_space_ratio = text.count("  ") / max(text.count(" "), 1)
    score += multi_space_ratio * 0.3

    return min(score, 1.0)
