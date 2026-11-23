"""Internal peer-to-peer and cross-modal similarity scoring (Similarity-Weighted Coverage)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from src.config import Settings
from src.indexing.vector_index import VectorIndex
from src.logger import get_logger
from src.models import MatchEvidence, TextChunk

logger = get_logger("signals.peer_similarity")


@dataclass
class PeerSimilarityResult:
    """Output of internal peer & cross-modal similarity scoring for one document."""

    peer_similarity_percentage: float = 0.0
    peer_matches: List[MatchEvidence] = field(default_factory=list)
    multimodal_matches: List[MatchEvidence] = field(default_factory=list)


def _build_location(query_chunk: TextChunk, matched_chunk: TextChunk) -> Dict[str, object]:
    """Construct schema-compliant location metadata (page numbers and/or timestamps)."""
    location: Dict[str, object] = {}
    if query_chunk.modality == "mp4":
        ts = query_chunk.format_timestamp()
        if ts is not None:
            location["query_timestamp"] = ts
    elif query_chunk.page_number is not None:
        location["query_page"] = query_chunk.page_number

    if matched_chunk.modality == "mp4":
        ts = matched_chunk.format_timestamp()
        if ts is not None:
            location["matched_timestamp"] = ts
    elif matched_chunk.page_number is not None:
        location["matched_page"] = matched_chunk.page_number

    return location


def _is_cross_modal(q_chunk: TextChunk, m_chunk: TextChunk) -> bool:
    """True when one side is MP4 audio/video and the other is a text document."""
    return (q_chunk.modality == "mp4") != (m_chunk.modality == "mp4")


class PeerSimilarityDetector:
    """Computes Similarity-Weighted Coverage % and explainable peer/multimodal matches.

    DSA: FAISS nearest-neighbor search + Hash Map deduplication + Timsort.
    TC: O(C * (N * D + k log k) + M log M).
    SC: O(M) for stored MatchEvidence records.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def evaluate_document(
        self,
        document_id: str,
        doc_chunks: List[TextChunk],
        doc_embeddings: np.ndarray,
        vector_index: VectorIndex,
    ) -> PeerSimilarityResult:
        """Score a single document's chunks against all peer chunks in vector_index."""
        total_chunks = len(doc_chunks)
        if total_chunks == 0 or doc_embeddings.size == 0 or vector_index.size == 0:
            logger.debug("[%s] Zero chunks or empty index; peer similarity = 0.0%%", document_id)
            return PeerSimilarityResult()

        weighted_sim_sum = 0.0
        peer_dedup: Dict[Tuple[str, str], MatchEvidence] = {}
        multimodal_dedup: Dict[Tuple[str, str], MatchEvidence] = {}

        for idx, q_chunk in enumerate(doc_chunks):
            q_vec = doc_embeddings[idx]
            candidates = vector_index.search_excluding_document(
                q_vec,
                exclude_document_id=document_id,
                top_k=self.settings.top_k_matches,
            )

            best_chunk_sim = 0.0
            for cand_chunk, sim in candidates:
                cross_modal = _is_cross_modal(q_chunk, cand_chunk)
                threshold = (
                    self.settings.cross_modal_similarity_threshold
                    if cross_modal
                    else self.settings.peer_similarity_threshold
                )
                if sim < threshold:
                    continue

                if sim > best_chunk_sim:
                    best_chunk_sim = sim

                evidence = MatchEvidence(
                    matched_file=cand_chunk.document_id,
                    similarity_score=sim,
                    query_passage=q_chunk.text,
                    matched_passage=cand_chunk.text,
                    location=_build_location(q_chunk, cand_chunk),
                    match_type="multimodal" if cross_modal else "peer",
                )
                target_map = multimodal_dedup if cross_modal else peer_dedup
                dedup_key = (cand_chunk.document_id, cand_chunk.chunk_id)
                existing = target_map.get(dedup_key)
                if existing is None or sim > existing.similarity_score:
                    target_map[dedup_key] = evidence

            weighted_sim_sum += best_chunk_sim

        peer_pct = min(100.0, (weighted_sim_sum / total_chunks) * 100.0)
        peer_matches = sorted(
            peer_dedup.values(),
            key=lambda m: m.similarity_score,
            reverse=True,
        )
        multimodal_matches = sorted(
            multimodal_dedup.values(),
            key=lambda m: m.similarity_score,
            reverse=True,
        )

        logger.info(
            "[%s] Peer similarity = %.2f%% (%d peer match(es), %d multimodal match(es))",
            document_id,
            peer_pct,
            len(peer_matches),
            len(multimodal_matches),
        )
        return PeerSimilarityResult(
            peer_similarity_percentage=peer_pct,
            peer_matches=peer_matches,
            multimodal_matches=multimodal_matches,
        )
