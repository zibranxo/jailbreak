"""Embedding-similarity detector against a known-unsafe pattern database."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    _HAS_ST = True
except ImportError:
    _HAS_ST = False


@dataclass
class MatchedPattern:
    """One pattern from the database matched by the input."""

    pattern_id: str
    text: str
    category: str
    similarity: float


@dataclass
class EmbeddingResult:
    """Result from the embedding detector."""

    is_suspicious: bool
    matched_patterns: list[MatchedPattern] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_suspicious": self.is_suspicious,
            "matched_patterns": [
                {
                    "id": m.pattern_id,
                    "text": m.text,
                    "category": m.category,
                    "similarity": round(m.similarity, 4),
                }
                for m in self.matched_patterns
            ],
            "confidence": self.confidence,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines and comments."""
    entries: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(json.loads(line))
    return entries


class EmbeddingDetector:
    """Cosine-similarity search against pre-encoded unsafe patterns.

    The SentenceTransformer model is loaded lazily on the first call to
    ``detect()`` or ``encode()`` so that startup is fast and unit tests
    can substitute a mock without triggering a download.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embeddings_path: str | Path | None = None,
        patterns_path: str | Path | None = None,
        threshold: float = 0.75,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold

        self._model: SentenceTransformer | None = None
        self._pattern_ids: list[str] = []
        self._pattern_texts: list[str] = []
        self._pattern_categories: list[str] = []
        self._pattern_embeddings: np.ndarray | None = None

        if embeddings_path and patterns_path:
            self.load_index(embeddings_path, patterns_path)

    # -- lazy model loading ---------------------------------------------------

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            if not _HAS_ST:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install with: pip install sentence-transformers"
                )
            self._model = SentenceTransformer(self._model_name)
        return self._model

    # -- index loading --------------------------------------------------------

    def load_index(
        self,
        embeddings_path: str | Path,
        patterns_path: str | Path,
    ) -> None:
        """Load precomputed embeddings and their metadata.

        The embeddings file and the patterns file must list entries
        in the same order (matched by position, not by id lookup).
        """
        patterns = _read_jsonl(Path(patterns_path))
        self._pattern_ids = [e["id"] for e in patterns]
        self._pattern_texts = [e["text"] for e in patterns]
        self._pattern_categories = [e.get("category", "unknown") for e in patterns]

        emb_entries = _read_jsonl(Path(embeddings_path))
        vectors = [np.array(e["embedding"], dtype=np.float32) for e in emb_entries]

        if len(vectors) != len(self._pattern_ids):
            raise ValueError(
                f"Embedding count ({len(vectors)}) != pattern count "
                f"({len(self._pattern_ids)}). "
                "Run precompute_embeddings() to regenerate."
            )

        if vectors:
            self._pattern_embeddings = np.stack(vectors)
            try:
                import faiss
                self._faiss_index = faiss.IndexFlatIP(self._pattern_embeddings.shape[1])
                self._faiss_index.add(self._pattern_embeddings)
            except ImportError:
                self._faiss_index = None

    # -- encoding -------------------------------------------------------------

    def encode(self, text: str) -> np.ndarray:
        """Encode *text* into a normalised embedding vector."""
        return self.model.encode(
            text, convert_to_numpy=True, normalize_embeddings=True
        )  # type: ignore[no-any-return]

    # -- detection ------------------------------------------------------------

    def detect(self, text: str) -> EmbeddingResult:
        """Compare *text* against the unsafe-pattern index.

        Returns all matches above the configured similarity threshold,
        sorted by similarity descending.
        """
        if self._pattern_embeddings is None:
            return EmbeddingResult(is_suspicious=False)

        query = self.encode(text)  # (dim,)
        matches: list[MatchedPattern] = []

        if getattr(self, "_faiss_index", None) is not None:
            D, I = self._faiss_index.search(query.reshape(1, -1), k=len(self._pattern_embeddings))
            for i in range(D.shape[1]):
                score = float(D[0][i])
                idx = int(I[0][i])
                if score >= self._threshold:
                    matches.append(
                        MatchedPattern(
                            pattern_id=self._pattern_ids[idx],
                            text=self._pattern_texts[idx],
                            category=self._pattern_categories[idx],
                            similarity=score,
                        )
                    )
        else:
            sims = self._pattern_embeddings @ query  # cosine (pre-normalised)
            for idx, score in enumerate(sims):
                if score >= self._threshold:
                    matches.append(
                        MatchedPattern(
                            pattern_id=self._pattern_ids[idx],
                            text=self._pattern_texts[idx],
                            category=self._pattern_categories[idx],
                            similarity=float(score),
                        )
                    )

        if not matches:
            return EmbeddingResult(is_suspicious=False)

        matches.sort(key=lambda m: m.similarity, reverse=True)
        return EmbeddingResult(
            is_suspicious=True,
            matched_patterns=matches,
            confidence=matches[0].similarity,
        )


# -- precompute helper --------------------------------------------------------


def precompute_embeddings(
    patterns_path: str | Path,
    output_path: str | Path,
    model_name: str = "all-MiniLM-L6-v2",
) -> None:
    """Encode every pattern in *patterns_path* and write to *output_path*.

    Run this whenever the pattern database changes so that
    ``EmbeddingDetector`` can reuse the cached vectors.
    """
    if not _HAS_ST:
        raise ImportError(
            "sentence-transformers is required. "
            "Install with: pip install sentence-transformers"
        )

    patterns = _read_jsonl(Path(patterns_path))
    texts = [e["text"] for e in patterns]
    ids = [e["id"] for e in patterns]

    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, normalize_embeddings=True)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for pid, vec in zip(ids, embeddings):
            fh.write(
                json.dumps({"id": pid, "embedding": vec.tolist()}) + "\n"
            )
