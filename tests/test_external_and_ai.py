"""Unit tests for external_similarity.py and ai_detector.py."""

from pathlib import Path

import numpy as np
import pytest

from src.config import Settings
from src.indexing.embedder import TextEmbedder
from src.models import TextChunk
from src.signals.ai_detector import AIDetector
from src.signals.external_similarity import ExternalSimilarityDetector


class _KeywordEmbedderModel:
    """Deterministic toy embedder mapping topic keywords to orthogonal unit vectors."""

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True):
        rows = []
        for t in texts:
            low = t.lower()
            v = np.array(
                [
                    1.0 if "supply chain" in low else 0.05,
                    1.0 if "solar" in low else 0.05,
                    1.0 if "blockchain" in low else 0.05,
                    0.1,
                ],
                dtype=np.float32,
            )
            v /= np.linalg.norm(v)
            rows.append(v)
        return np.vstack(rows)


def test_missing_and_empty_external_kb_never_crash(tmp_path: Path):
    settings = Settings(external_kb_dir=tmp_path / "does_not_exist")
    embedder = TextEmbedder(settings, model=_KeywordEmbedderModel())
    detector = ExternalSimilarityDetector(settings, embedder)

    chunk = TextChunk("sub.pdf::0", "sub.pdf", "pdf", "Supply chain margins fell in Q3.", 0, page_number=1)
    emb = embedder.encode([chunk.text])

    res = detector.evaluate_document("sub.pdf", [chunk], emb)
    assert res.external_similarity_percentage == 0.0
    assert res.external_matches == []

    # Empty folder case
    empty_dir = tmp_path / "empty_kb"
    empty_dir.mkdir()
    assert detector.build_index(empty_dir) == 0


def test_external_kb_flags_copied_reference_article(tmp_path: Path):
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "wiki_supply_chain.txt").write_text(
        "Global supply chain disruptions increased freight costs sharply across maritime routes. "
        "Retailers responded by regionalizing safety stock inventories and dual-sourcing key components.",
        encoding="utf-8",
    )

    settings = Settings(external_kb_dir=kb_dir)
    embedder = TextEmbedder(settings, model=_KeywordEmbedderModel())
    detector = ExternalSimilarityDetector(settings, embedder)
    assert detector.build_index() >= 1

    copied_chunk = TextChunk(
        "cand.pdf::0",
        "cand.pdf",
        "pdf",
        "Global supply chain disruptions raised freight costs sharply across maritime routes.",
        0,
        page_number=2,
    )
    emb = embedder.encode([copied_chunk.text])
    res = detector.evaluate_document("cand.pdf", [copied_chunk], emb)

    assert res.external_similarity_percentage >= 90.0
    assert len(res.external_matches) == 1
    assert res.external_matches[0].matched_file == "wiki_supply_chain.txt"
    assert res.external_matches[0].location["query_page"] == 2


def test_ai_detector_handles_empty_and_short_text_conservatively():
    detector = AIDetector(Settings())
    assert detector.estimate("empty.pdf", "").ai_generated_probability == 0.0
    short_res = detector.estimate("short.mp4", "Hello judges, we cut costs by ten percent.")
    assert short_res.method == "insufficient_text"
    assert short_res.ai_generated_probability <= 20.0


def test_ai_detector_separates_llm_styled_prose_from_human_writing():
    llm_text = (
        "Furthermore, adopting a holistic approach to supply chain resilience is paramount for modern enterprises. "
        "Moreover, it is crucial to note that vertical integration plays a crucial role in stabilizing quarterly margins. "
        "Additionally, organizations must leverage synergies across regional warehouses to build a robust framework. "
        "Consequently, navigating the complexities of global logistics underscores the need for comprehensive strategy. "
        "In conclusion, seamless integration of predictive analytics will ultimately foster innovation across the landscape."
    )
    human_text = (
        "Look, Q3 was rough. We didn't expect port delays in Chennai to drag on for three weeks, "
        "and honestly our safety stock model broke almost immediately! "
        "Why? Because we'd assumed 4-day lead times. "
        "So we shifted 30% of orders to a local Pune supplier, cut SKU variants from 42 down to 19, "
        "and it's already saving us Rs 14 lakh a month — not glamorous, just practical."
    )

    detector = AIDetector(Settings(use_hf_ai_classifier=False))
    llm_res = detector.estimate("llm.docx", llm_text)
    human_res = detector.estimate("human.docx", human_text)

    assert llm_res.ai_generated_probability >= 65.0
    assert human_res.ai_generated_probability <= 35.0
    assert llm_res.ai_generated_probability > human_res.ai_generated_probability + 30.0


def test_ai_detector_hf_failure_falls_back_gracefully_to_heuristic():
    def _broken_factory():
        raise RuntimeError("Offline: cannot download model")

    settings = Settings(use_hf_ai_classifier=True)
    detector = AIDetector(settings, hf_pipeline_factory=_broken_factory)

    text = " ".join(["This is a moderately long sentence with enough words for detection."] * 8)
    res = detector.estimate("doc.pdf", text)
    assert res.method == "heuristic"
    assert 0.0 <= res.ai_generated_probability <= 100.0
