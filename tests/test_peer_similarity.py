"""Unit tests for src/signals/peer_similarity.py."""

import numpy as np
import pytest

from src.config import Settings
from src.indexing.vector_index import VectorIndex
from src.models import TextChunk
from src.signals.peer_similarity import PeerSimilarityDetector


def _doc_chunk(doc_id: str, idx: int, modality: str = "pdf", page: int | None = 1, start: float | None = None, end: float | None = None) -> TextChunk:
    return TextChunk(
        chunk_id=f"{doc_id}::chunk_{idx}",
        document_id=doc_id,
        modality=modality,
        text=f"Passage {idx} from {doc_id}",
        chunk_index=idx,
        page_number=page,
        start_time=start,
        end_time=end,
    )


def test_zero_chunks_returns_zero_without_division_error():
    detector = PeerSimilarityDetector(Settings())
    res = detector.evaluate_document("empty.pdf", [], np.empty((0, 4), dtype=np.float32), VectorIndex())
    assert res.peer_similarity_percentage == 0.0
    assert res.peer_matches == []
    assert res.multimodal_matches == []


def test_weighted_coverage_distinguishes_exact_copy_from_paraphrase():
    index = VectorIndex()
    c_src = _doc_chunk("source.pdf", 0, modality="pdf", page=1)
    c_exact = _doc_chunk("exact.docx", 0, modality="docx", page=2)
    c_para = _doc_chunk("paraphrase.docx", 0, modality="docx", page=1)

    v_src = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_exact = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # Cosine similarity = 0.82 with v_src (above 0.78 text threshold)
    v_para = np.array([0.82, float(np.sqrt(1.0 - 0.82**2)), 0.0], dtype=np.float32)

    index.add([c_src, c_exact, c_para], np.vstack([v_src, v_exact, v_para]))
    detector = PeerSimilarityDetector(Settings())

    res_exact = detector.evaluate_document("exact.docx", [c_exact], v_exact.reshape(1, -1), index)
    res_para = detector.evaluate_document("paraphrase.docx", [c_para], v_para.reshape(1, -1), index)

    assert res_exact.peer_similarity_percentage == pytest.approx(100.0, abs=0.1)
    assert res_para.peer_similarity_percentage == pytest.approx(82.0, abs=0.1)
    assert res_exact.peer_similarity_percentage > res_para.peer_similarity_percentage
    assert res_exact.peer_matches[0].location == {"query_page": 2, "matched_page": 1}


def test_cross_modal_mp4_pdf_uses_lower_threshold_and_preserves_timestamps():
    index = VectorIndex()
    c_pdf = _doc_chunk("report.pdf", 0, modality="pdf", page=3)
    c_mp4 = _doc_chunk("pitch.mp4", 0, modality="mp4", page=None, start=24.0, end=36.0)

    v_pdf = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # Cosine similarity = 0.74 (above 0.72 cross-modal threshold, below 0.78 text threshold)
    v_mp4 = np.array([0.74, float(np.sqrt(1.0 - 0.74**2)), 0.0], dtype=np.float32)

    index.add([c_pdf, c_mp4], np.vstack([v_pdf, v_mp4]))
    detector = PeerSimilarityDetector(Settings())

    res = detector.evaluate_document("report.pdf", [c_pdf], v_pdf.reshape(1, -1), index)
    assert res.peer_similarity_percentage == pytest.approx(74.0, abs=0.1)
    assert res.peer_matches == []
    assert len(res.multimodal_matches) == 1
    assert res.multimodal_matches[0].matched_file == "pitch.mp4"
    assert res.multimodal_matches[0].location == {
        "query_page": 3,
        "matched_timestamp": "00:00:24 - 00:00:36",
    }


def test_unrelated_documents_score_zero_and_matches_sort_descending():
    index = VectorIndex()
    c_q = _doc_chunk("query.pdf", 0, modality="pdf", page=1)
    c_low = _doc_chunk("unrelated.pdf", 0, modality="pdf", page=1)
    c_med = _doc_chunk("peer_med.pdf", 0, modality="pdf", page=2)
    c_high = _doc_chunk("peer_high.pdf", 0, modality="pdf", page=4)

    v_q = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_low = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v_med = np.array([0.85, float(np.sqrt(1.0 - 0.85**2)), 0.0], dtype=np.float32)
    v_high = np.array([0.96, float(np.sqrt(1.0 - 0.96**2)), 0.0], dtype=np.float32)

    # First test strictly unrelated
    idx_unrel = VectorIndex()
    idx_unrel.add([c_q, c_low], np.vstack([v_q, v_low]))
    detector = PeerSimilarityDetector(Settings())
    res_unrel = detector.evaluate_document("query.pdf", [c_q], v_q.reshape(1, -1), idx_unrel)
    assert res_unrel.peer_similarity_percentage == 0.0
    assert res_unrel.peer_matches == []

    # Now test descending sort order across multiple matches
    index.add([c_q, c_low, c_med, c_high], np.vstack([v_q, v_low, v_med, v_high]))
    res_multi = detector.evaluate_document("query.pdf", [c_q], v_q.reshape(1, -1), index)
    assert [m.matched_file for m in res_multi.peer_matches] == ["peer_high.pdf", "peer_med.pdf"]
