"""FAISS exact cosine similarity index with guaranteed self-match exclusion."""

from __future__ import annotations

from collections import Counter
from typing import Any, List, Optional, Tuple

import numpy as np

from src.logger import get_logger
from src.models import TextChunk

try:
    import faiss
except Exception:  # pragma: no cover - defensive fallback for exotic DLL setups
    faiss = None  # type: ignore[assignment]

logger = get_logger("indexing.vector_index")


def _prepare_vectors(embeddings: np.ndarray) -> np.ndarray:
    """Cast to contiguous float32 and L2-normalize rows safely (no div-by-zero)."""
    arr = np.ascontiguousarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    return arr / safe_norms


class VectorIndex:
    """FAISS IndexFlatIP over L2-normalized chunk embeddings (= exact Cosine Similarity).

    DSA: FAISS flat dense matrix + parallel metadata list + Counter(doc_id -> count).
    TC: add() = O(N * D); search() = O(N * D + k log k) per query.
    SC: O(N * D) floats (~1.5 KB per 384-dim chunk).
    """

    def __init__(self, dimension: Optional[int] = None) -> None:
        self._dim: Optional[int] = dimension
        self._use_faiss: bool = faiss is not None
        self._index: Optional[Any] = (
            faiss.IndexFlatIP(dimension) if (dimension is not None and self._use_faiss) else None
        )
        self._matrix: Optional[np.ndarray] = None
        self._chunks: List[TextChunk] = []
        self._doc_chunk_counts: Counter[str] = Counter()

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        """Add chunks and their embeddings to the FAISS index (or NumPy fallback)."""
        if not chunks or embeddings.size == 0:
            logger.debug("VectorIndex.add() called with empty input; skipping")
            return

        vecs = _prepare_vectors(embeddings)
        if len(chunks) != vecs.shape[0]:
            raise ValueError(
                f"Chunk count ({len(chunks)}) does not match embedding rows ({vecs.shape[0]})"
            )

        if self._dim is None:
            self._dim = int(vecs.shape[1])
            if self._use_faiss:
                self._index = faiss.IndexFlatIP(self._dim)
        elif vecs.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dimension ({vecs.shape[1]}) does not match index ({self._dim})"
            )

        if self._use_faiss and self._index is not None:
            self._index.add(vecs)
        else:
            self._matrix = vecs if self._matrix is None else np.vstack([self._matrix, vecs])

        self._chunks.extend(chunks)
        for ch in chunks:
            self._doc_chunk_counts[ch.document_id] += 1

        backend = "FAISS IndexFlatIP" if self._use_faiss else "NumPy exact IP fallback"
        logger.info(
            "Indexed %d chunk(s) into %s (total=%d, dim=%d)",
            len(chunks),
            backend,
            self.size,
            self._dim,
        )

    def _raw_search(self, q_vec: np.ndarray, k_fetch: int) -> Tuple[np.ndarray, np.ndarray]:
        if self._use_faiss and self._index is not None:
            return self._index.search(q_vec, k_fetch)
        assert self._matrix is not None
        sims = (self._matrix @ q_vec.T).ravel()
        order = np.argsort(-sims)[:k_fetch]
        return sims[order].reshape(1, -1), order.reshape(1, -1)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[TextChunk, float]]:
        """Return up to top_k closest chunks across the entire index."""
        if self.size == 0 or top_k <= 0:
            return []

        q_vec = _prepare_vectors(query_embedding)
        k_fetch = min(self.size, top_k)
        scores, indices = self._raw_search(q_vec, k_fetch)

        results: List[Tuple[TextChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= self.size:
                continue
            results.append((self._chunks[int(idx)], float(np.clip(score, -1.0, 1.0))))
        return results

    def search_excluding_document(
        self,
        query_embedding: np.ndarray,
        exclude_document_id: str,
        top_k: int = 5,
    ) -> List[Tuple[TextChunk, float]]:
        """Return up to top_k closest peer chunks, strictly excluding exclude_document_id.

        Fetches top_k + own_chunk_count so a document with many self-chunks
        can never starve out genuine peer matches.
        """
        if self.size == 0 or top_k <= 0:
            return []

        own_count = self._doc_chunk_counts.get(exclude_document_id, 0)
        if own_count >= self.size:
            logger.debug(
                "All %d indexed chunk(s) belong to '%s'; no peer candidates exist",
                self.size,
                exclude_document_id,
            )
            return []

        q_vec = _prepare_vectors(query_embedding)
        k_fetch = min(self.size, top_k + own_count)
        scores, indices = self._raw_search(q_vec, k_fetch)

        results: List[Tuple[TextChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= self.size:
                continue
            candidate = self._chunks[int(idx)]
            if candidate.document_id == exclude_document_id:
                continue
            results.append((candidate, float(np.clip(score, -1.0, 1.0))))
            if len(results) >= top_k:
                break
        return results
