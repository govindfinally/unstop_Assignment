"""External/web reference plagiarism detection against a local knowledge base."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from src.config import Settings
from src.extractors.normalizer import (
    normalize_text,
    sentence_aware_sliding_window,
    split_sentences,
)
from src.indexing.embedder import TextEmbedder
from src.indexing.vector_index import VectorIndex
from src.logger import get_logger
from src.models import MatchEvidence, TextChunk

logger = get_logger("signals.external_similarity")

_KB_EXTENSIONS = (".txt", ".md")


@dataclass
class ExternalSimilarityResult:
    """Output of external reference plagiarism scoring for one document."""

    external_similarity_percentage: float = 0.0
    external_matches: List[MatchEvidence] = field(default_factory=list)


class ExternalSimilarityDetector:
    """Indexes local reference articles in external_knowledge/ into FAISS and queries chunks.

    DSA: Reuses VectorIndex (faiss.IndexFlatIP) + Hash Map deduplication + Timsort.
    TC: Build index O(E * D); Query per doc O(C * (E * D + k log k)).
    SC: O(E * D) floats where E is external KB chunk count.
    """

    def __init__(self, settings: Settings, embedder: TextEmbedder) -> None:
        self.settings = settings
        self.embedder = embedder
        self._index = VectorIndex()
        self._kb_loaded = False

    def build_index(self, kb_dir: Path | None = None) -> int:
        """Load and index all .txt/.md reference files from kb_dir into FAISS."""
        target_dir = kb_dir if kb_dir is not None else self.settings.external_kb_dir
        self._index = VectorIndex()
        self._kb_loaded = True

        if not target_dir.exists() or not target_dir.is_dir():
            logger.warning(
                "External knowledge directory not found: %s (external similarity will be 0.0%%)",
                target_dir,
            )
            return 0

        ref_files = sorted(
            p for p in target_dir.iterdir() if p.is_file() and p.suffix.lower() in _KB_EXTENSIONS
        )
        if not ref_files:
            logger.warning(
                "External knowledge directory is empty: %s (external similarity will be 0.0%%)",
                target_dir,
            )
            return 0

        kb_chunks: List[TextChunk] = []
        for ref_path in ref_files:
            try:
                raw = ref_path.read_text(encoding="utf-8", errors="ignore")
            except Exception as exc:
                logger.warning("Failed to read external KB file '%s': %s", ref_path.name, exc)
                continue

            cleaned = normalize_text(raw)
            sentences = split_sentences(cleaned)
            chunk_texts = sentence_aware_sliding_window(
                sentences,
                chunk_size_words=self.settings.chunk_size_words,
                overlap_words=self.settings.chunk_overlap_words,
                min_chunk_words=self.settings.min_chunk_words,
            )
            for idx, c_text in enumerate(chunk_texts):
                kb_chunks.append(
                    TextChunk(
                        chunk_id=f"{ref_path.name}::kb_{idx}",
                        document_id=ref_path.name,
                        modality="external",
                        text=c_text,
                        chunk_index=idx,
                        page_number=1,
                    )
                )

        if not kb_chunks:
            logger.warning("No extractable chunks found in external KB: %s", target_dir)
            return 0

        embeddings = self.embedder.encode([c.text for c in kb_chunks])
        self._index.add(kb_chunks, embeddings)
        logger.info(
            "Built external KB index from %d file(s): %d chunk(s)",
            len(ref_files),
            len(kb_chunks),
        )
        return len(kb_chunks)

    def evaluate_document(
        self,
        document_id: str,
        doc_chunks: List[TextChunk],
        doc_embeddings: np.ndarray,
    ) -> ExternalSimilarityResult:
        """Score a document's chunks against the indexed external knowledge base."""
        if not self._kb_loaded:
            self.build_index()

        total_chunks = len(doc_chunks)
        if total_chunks == 0 or doc_embeddings.size == 0 or self._index.size == 0:
            logger.debug("[%s] No chunks or empty external KB; external similarity = 0.0%%", document_id)
            return ExternalSimilarityResult()

        weighted_sim_sum = 0.0
        dedup_map: Dict[Tuple[str, str], MatchEvidence] = {}

        for idx, q_chunk in enumerate(doc_chunks):
            q_vec = doc_embeddings[idx]
            candidates = self._index.search(q_vec, top_k=self.settings.top_k_matches)

            best_sim = 0.0
            for cand_chunk, sim in candidates:
                if sim < self.settings.external_similarity_threshold:
                    continue
                if sim > best_sim:
                    best_sim = sim

                location: Dict[str, object] = {}
                if q_chunk.modality == "mp4":
                    ts = q_chunk.format_timestamp()
                    if ts is not None:
                        location["query_timestamp"] = ts
                elif q_chunk.page_number is not None:
                    location["query_page"] = q_chunk.page_number
                location["matched_source"] = cand_chunk.document_id

                evidence = MatchEvidence(
                    matched_file=cand_chunk.document_id,
                    similarity_score=sim,
                    query_passage=q_chunk.text,
                    matched_passage=cand_chunk.text,
                    location=location,
                    match_type="external",
                )
                key = (cand_chunk.document_id, cand_chunk.chunk_id)
                existing = dedup_map.get(key)
                if existing is None or sim > existing.similarity_score:
                    dedup_map[key] = evidence

            weighted_sim_sum += best_sim

        ext_pct = min(100.0, (weighted_sim_sum / total_chunks) * 100.0)
        matches = sorted(dedup_map.values(), key=lambda m: m.similarity_score, reverse=True)

        logger.info(
            "[%s] External similarity = %.2f%% (%d match(es))",
            document_id,
            ext_pct,
            len(matches),
        )
        return ExternalSimilarityResult(
            external_similarity_percentage=ext_pct,
            external_matches=matches,
        )
