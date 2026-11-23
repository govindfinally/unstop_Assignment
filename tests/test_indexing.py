"""Unit tests for src/indexing/embedder.py and src/indexing/vector_index.py."""

import numpy as np
import pytest

from src.config import Settings
from src.indexing.embedder import TextEmbedder
from src.indexing.vector_index import VectorIndex
from src.models import TextChunk


class _FakeSentenceModel:
    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True):
        # Deterministic 4-dim vectors for fast unit testing
        vecs = np.array([[float(i + 1), 1.0, 0.5, 0.25] for i in range(len(texts))], dtype=np.float32)
        if normalize_embeddings:
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs


def _chunk(doc_id: str, idx: int) -> TextChunk:
    return TextChunk(
        chunk_id=f"{doc_id}::chunk_{idx}",
        document_id=doc_id,
        modality="pdf",
        text=f"Sample text for {doc_id} chunk {idx}",
        chunk_index=idx,
        page_number=1,
    )


def test_embedder_handles_empty_and_normalizes_float32():
    embedder = TextEmbedder(Settings(), model=_FakeSentenceModel())

    empty_out = embedder.encode([])
    assert empty_out.shape == (0, 384)
    assert empty_out.dtype == np.float32

    out = embedder.encode(["hello world", "second chunk"])
    assert out.shape == (2, 4)
    assert out.dtype == np.float32
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_empty_vector_index_returns_empty_without_crashing():
    index = VectorIndex()
    index.add([], np.empty((0, 4), dtype=np.float32))
    assert index.size == 0
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert index.search(q, top_k=3) == []
    assert index.search_excluding_document(q, exclude_document_id="doc_a.pdf", top_k=3) == []


def test_strict_self_match_exclusion_even_with_many_identical_self_chunks():
    index = VectorIndex()
    # Doc A has 5 identical chunks; Doc B has 1 slightly different chunk
    chunks = [_chunk("doc_a.pdf", i) for i in range(5)] + [_chunk("doc_b.pdf", 0)]
    vecs = np.array(
        [[1.0, 0.0, 0.0, 0.0]] * 5 + [[0.9, 0.43589, 0.0, 0.0]],
        dtype=np.float32,
    )
    index.add(chunks, vecs)

    results = index.search_excluding_document(vecs[0], exclude_document_id="doc_a.pdf", top_k=3)
    assert len(results) == 1
    matched_chunk, score = results[0]
    assert matched_chunk.document_id == "doc_b.pdf"
    assert score == pytest.approx(0.9, abs=1e-4)


def test_single_document_corpus_returns_no_peer_matches():
    index = VectorIndex()
    chunks = [_chunk("only_doc.pdf", 0), _chunk("only_doc.pdf", 1)]
    vecs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    index.add(chunks, vecs)

    assert index.search_excluding_document(vecs[0], exclude_document_id="only_doc.pdf", top_k=5) == []


def test_exact_cosine_similarity_and_float64_unnormalized_defensive_guard():
    index = VectorIndex()
    chunks = [_chunk("d1.pdf", 0), _chunk("d2.pdf", 0)]
    # Deliberately unnormalized float64 vectors (identical vs orthogonal)
    vecs = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]], dtype=np.float64)
    index.add(chunks, vecs)

    query = np.array([6.0, 8.0, 0.0], dtype=np.float64)
    results = index.search(query, top_k=2)
    assert results[0][0].document_id == "d1.pdf"
    assert results[0][1] == pytest.approx(1.0, abs=1e-5)
    assert results[1][0].document_id == "d2.pdf"
    assert results[1][1] == pytest.approx(0.0, abs=1e-5)
