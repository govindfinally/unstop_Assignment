"""End-to-end orchestration pipeline for multimodal submission integrity checks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import Settings
from src.extractors.document_extractor import DocumentExtractor
from src.extractors.normalizer import sentence_aware_sliding_window_with_meta
from src.extractors.video_extractor import VideoExtractor, build_video_chunks
from src.indexing.boilerplate import flag_boilerplate_sentences
from src.indexing.embedder import TextEmbedder
from src.indexing.vector_index import VectorIndex
from src.logger import get_logger
from src.models import DocumentReport, ExtractedDocument, TextChunk
from src.reporting.report_builder import ReportBuilder
from src.signals.ai_detector import AIDetector
from src.signals.external_similarity import ExternalSimilarityDetector
from src.signals.peer_similarity import PeerSimilarityDetector

logger = get_logger("pipeline")


class IntegrityPipeline:
    """Coordinates Stages 1-4: extraction, boilerplate, indexing, scoring, and reporting.

    DSA: Single-batch contiguous embedding matrix + doc_id -> (start, end) slice map.
    TC: O(T_audio + N * L_embed + N^2 * D_faiss).
    SC: O(N * D + M) where N = total chunks, D = 384, M = total matches.
    """

    def __init__(
        self,
        settings: Settings,
        document_extractor: Optional[DocumentExtractor] = None,
        video_extractor: Optional[VideoExtractor] = None,
        embedder: Optional[TextEmbedder] = None,
        ai_detector: Optional[AIDetector] = None,
        report_builder: Optional[ReportBuilder] = None,
    ) -> None:
        self.settings = settings
        self.document_extractor = document_extractor or DocumentExtractor(settings)
        self.video_extractor = video_extractor or VideoExtractor(settings)
        self.embedder = embedder or TextEmbedder(settings)
        self.peer_detector = PeerSimilarityDetector(settings)
        self.external_detector = ExternalSimilarityDetector(settings, self.embedder)
        self.ai_detector = ai_detector or AIDetector(settings)
        self.report_builder = report_builder or ReportBuilder()

    def _discover_files(self, input_dir: Path) -> List[Path]:
        """Return sorted supported submission files, ignoring manifests/junk files."""
        if not input_dir.exists() or not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

        supported = set(ext.lower() for ext in self.settings.supported_extensions)
        files = sorted(
            p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in supported
        )
        logger.info("Discovered %d supported submission file(s) in %s", len(files), input_dir)
        return files

    def _extract_all(self, files: List[Path]) -> List[ExtractedDocument]:
        docs: List[ExtractedDocument] = []
        for path in files:
            ext = path.suffix.lower()
            if ext in (".pdf", ".docx"):
                docs.append(self.document_extractor.extract(path))
            elif ext == ".mp4":
                docs.append(self.video_extractor.extract(path))
        return docs

    def _build_chunks_for_document(self, doc: ExtractedDocument) -> List[TextChunk]:
        """Chunk only non-boilerplate SentenceItems while preserving page/timestamp metadata."""
        if doc.status != "ok":
            return []

        if doc.modality == "mp4":
            return build_video_chunks(
                document_id=doc.document_id,
                segments=doc.sentences,
                max_words=self.settings.chunk_size_words,
                pause_threshold_sec=2.0,
                min_chunk_words=min(6, self.settings.min_chunk_words),
            )

        active_items = [
            (item.text, item.page_number)
            for item in doc.sentences
            if not item.is_boilerplate and item.text.strip()
        ]
        if not active_items:
            return []

        chunk_pairs: List[Tuple[str, Optional[int]]] = []
        if doc.modality == "pdf":
            # Keep PDF chunks strictly page-bounded so page citations are 100% pinpoint accurate
            pages_ordered: Dict[Optional[int], List[Tuple[str, Optional[int]]]] = {}
            for item_pair in active_items:
                pages_ordered.setdefault(item_pair[1], []).append(item_pair)
            for page_items in pages_ordered.values():
                chunk_pairs.extend(
                    sentence_aware_sliding_window_with_meta(
                        page_items,
                        chunk_size_words=self.settings.chunk_size_words,
                        overlap_words=self.settings.chunk_overlap_words,
                        min_chunk_words=self.settings.min_chunk_words,
                    )
                )
        else:
            chunk_pairs = sentence_aware_sliding_window_with_meta(
                active_items,
                chunk_size_words=self.settings.chunk_size_words,
                overlap_words=self.settings.chunk_overlap_words,
                min_chunk_words=self.settings.min_chunk_words,
            )

        return [
            TextChunk(
                chunk_id=f"{doc.document_id}::chunk_{idx}",
                document_id=doc.document_id,
                modality=doc.modality,
                text=c_text,
                chunk_index=idx,
                page_number=page_num,
            )
            for idx, (c_text, page_num) in enumerate(chunk_pairs)
        ]

    def _classify_risk(
        self,
        status: str,
        peer_pct: float,
        ext_pct: float,
        ai_pct: float,
    ) -> str:
        if status != "ok":
            return "SKIPPED"
        if (
            peer_pct >= self.settings.high_risk_peer_pct
            or ext_pct >= self.settings.high_risk_external_pct
            or ai_pct >= self.settings.high_risk_ai_pct
        ):
            return "HIGH"
        if (
            peer_pct >= self.settings.medium_risk_peer_pct
            or ext_pct >= self.settings.medium_risk_external_pct
            or ai_pct >= self.settings.medium_risk_ai_pct
        ):
            return "MEDIUM"
        return "LOW"

    def _build_explanation(
        self,
        doc: ExtractedDocument,
        risk: str,
        peer_pct: float,
        ext_pct: float,
        ai_pct: float,
        peer_match_count: int,
        mm_match_count: int,
        ext_match_count: int,
    ) -> str:
        if doc.status != "ok":
            reason = doc.warnings[0] if doc.warnings else f"Extraction status: {doc.status}"
            return f"Submission skipped safely without crashing pipeline. Reason: {reason}"

        parts: List[str] = [f"Overall integrity risk classified as {risk}."]
        if peer_match_count > 0:
            parts.append(
                f"Detected {peer_match_count} internal peer overlap(s) ({peer_pct:.1f}% weighted peer similarity)."
            )
        if mm_match_count > 0:
            parts.append(
                f"Detected {mm_match_count} cross-modal (audio/video <-> text) overlap(s)."
            )
        if ext_match_count > 0:
            parts.append(
                f"Detected {ext_match_count} match(es) against external reference sources ({ext_pct:.1f}%)."
            )
        if ai_pct >= self.settings.medium_risk_ai_pct:
            parts.append(
                f"Elevated AI-generated writing indicators ({ai_pct:.1f}% likelihood)."
            )
        if doc.boilerplate_sentences_ignored > 0:
            parts.append(
                f"Discounted {doc.boilerplate_sentences_ignored} shared competition prompt/boilerplate sentence(s) to prevent false positives."
            )
        if risk == "LOW" and peer_match_count == 0 and mm_match_count == 0 and ext_match_count == 0:
            parts.append("No significant peer, cross-modal, or external overlaps detected.")
        return " ".join(parts)

    def run(self, input_dir: Path, output_dir: Path) -> List[DocumentReport]:
        """Execute the full 4-stage pipeline and write all JSON/Markdown reports."""
        logger.info("Running IntegrityPipeline: input=%s -> output=%s", input_dir, output_dir)
        files = self._discover_files(input_dir)
        if not files:
            logger.warning("No supported files found in %s", input_dir)
            self.report_builder.write_all([], output_dir)
            return []

        # Stage 1: Multimodal Extraction
        documents = self._extract_all(files)

        # Stage 2A: Sentence-Level Boilerplate Filtering (BEFORE chunking!)
        flag_boilerplate_sentences(documents, self.settings)

        # Stage 2B: Chunk non-boilerplate text + Single-batch Embedding + FAISS Index
        all_chunks: List[TextChunk] = []
        doc_slices: Dict[str, Tuple[int, int]] = {}

        for doc in documents:
            chunks = self._build_chunks_for_document(doc)
            doc.chunks = chunks
            start = len(all_chunks)
            all_chunks.extend(chunks)
            doc_slices[doc.document_id] = (start, len(all_chunks))

        if all_chunks:
            all_embeddings = self.embedder.encode([c.text for c in all_chunks])
        else:
            all_embeddings = np.empty((0, self.embedder.dimension), dtype=np.float32)

        submission_index = VectorIndex()
        submission_index.add(all_chunks, all_embeddings)

        # Pre-build External Knowledge Base index once for the batch
        self.external_detector.build_index()

        # Stage 3 & 4: Compute Signals, Risk Bands & Explanations
        reports: List[DocumentReport] = []
        for doc in documents:
            if doc.status != "ok":
                reports.append(
                    DocumentReport(
                        document_id=doc.document_id,
                        status=doc.status,
                        total_integrity_risk="SKIPPED",
                        warnings=list(doc.warnings),
                        explanations=self._build_explanation(
                            doc, "SKIPPED", 0.0, 0.0, 0.0, 0, 0, 0
                        ),
                    )
                )
                continue

            start, end = doc_slices[doc.document_id]
            doc_emb = all_embeddings[start:end]

            peer_res = self.peer_detector.evaluate_document(
                doc.document_id, doc.chunks, doc_emb, submission_index
            )
            ext_res = self.external_detector.evaluate_document(
                doc.document_id, doc.chunks, doc_emb
            )

            # Run AI detector strictly on post-boilerplate original text
            original_text = " ".join(
                s.text for s in doc.sentences if not s.is_boilerplate
            ).strip()
            ai_res = self.ai_detector.estimate(doc.document_id, original_text)

            risk = self._classify_risk(
                doc.status,
                peer_res.peer_similarity_percentage,
                ext_res.external_similarity_percentage,
                ai_res.ai_generated_probability,
            )
            explanation = self._build_explanation(
                doc=doc,
                risk=risk,
                peer_pct=peer_res.peer_similarity_percentage,
                ext_pct=ext_res.external_similarity_percentage,
                ai_pct=ai_res.ai_generated_probability,
                peer_match_count=len(peer_res.peer_matches),
                mm_match_count=len(peer_res.multimodal_matches),
                ext_match_count=len(ext_res.external_matches),
            )

            reports.append(
                DocumentReport(
                    document_id=doc.document_id,
                    status=doc.status,
                    total_integrity_risk=risk,
                    peer_similarity_percentage=peer_res.peer_similarity_percentage,
                    external_similarity_percentage=ext_res.external_similarity_percentage,
                    ai_generated_probability=ai_res.ai_generated_probability,
                    peer_matches=peer_res.peer_matches,
                    multimodal_matches=peer_res.multimodal_matches,
                    external_matches=ext_res.external_matches,
                    boilerplate_sentences_ignored=doc.boilerplate_sentences_ignored,
                    warnings=list(doc.warnings),
                    explanations=explanation,
                )
            )

        self.report_builder.write_all(reports, output_dir)
        logger.info("IntegrityPipeline completed cleanly for %d document(s)", len(reports))
        return reports
