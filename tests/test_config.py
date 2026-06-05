"""Tests for utility modules."""

from src.utils.text import detect_obfuscation, normalize_text


class TestNormalizeText:
    def test_trims_whitespace(self):
        assert normalize_text("  hello  ") == "hello"

    def test_collapses_whitespace(self):
        assert normalize_text("hello   world") == "hello world"

    def test_removes_zero_width_chars(self):
        text = "hello\u200bworld\u200ctest"
        assert "\u200b" not in normalize_text(text)
        assert "\u200c" not in normalize_text(text)

    def test_normalizes_unicode(self):
        # NFKC normalization should normalize full-width chars
        text = "\uff48\uff45\uff4c\uff4c\uff4f"  # full-width "hello"
        result = normalize_text(text)
        assert result == "hello"

    def test_strips_homoglyphs(self):
        text = "t3st"
        result = normalize_text(text, strip_homoglyphs=True)
        assert "3" not in result  # 3 → e
        assert "test" in result


class TestDetectObfuscation:
    def test_clean_text_low_score(self):
        score = detect_obfuscation("Hello, how are you?")
        assert score < 0.3

    def test_homoglyph_text_higher_score(self):
        score = detect_obfuscation("h3ll0 w0rld")
        assert score > 0.1
