"""Unified PDF and DOCX extractor sharing sentence-aware normalization.

Handles .pdf via PyMuPDF (fitz) and .docx via python-docx, producing unified
SentenceItem lists with exact (PDF) or estimated (DOCX) page numbers so
sentence-level boilerplate filtering runs cleanly before sliding-window chunking.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import docx
import pymupdf

from src.config import Settings
from src.extractors.base import BaseExtractor
from src.extractors.normalizer import (
    normalize_text,
    split_sentences,
    strip_repeated_headers_footers,
)
from src.logger import get_logger
from src.models import ExtractedDocument, SentenceItem

logger = get_logger("extractors.document")

_DOCX_WORDS_PER_ESTIMATED_PAGE = 250


class DocumentExtractor(BaseExtractor):
    """Unified fault-tolerant extractor for .pdf and .docx submissions."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract normalized text and page-tagged SentenceItems from PDF or DOCX.

        TC: O(P + L) where P = pages/paragraphs/table cells, L = total characters.
        SC: O(L) for extracted text and SentenceItem list.
        """
        doc_id = file_path.name
        ext = file_path.suffix.lower()
        modality = ext.lstrip(".")

        if not file_path.exists() or not file_path.is_file():
            msg = f"File not found or not a regular file: {file_path}"
            logger.warning("[%s] %s", doc_id, msg)
            return ExtractedDocument(
                document_id=doc_id,
                file_path=str(file_path),
                modality=modality,
                status="skipped_corrupted",
                warnings=[msg],
            )

        if file_path.stat().st_size == 0:
            msg = f"File is empty (0 bytes): {doc_id}"
            logger.warning("[%s] %s", doc_id, msg)
            return ExtractedDocument(
                document_id=doc_id,
                file_path=str(file_path),
                modality=modality,
                status="skipped_empty",
                warnings=[msg],
            )

        if ext == ".pdf":
            return self._extract_pdf(file_path, doc_id)
        if ext == ".docx":
            return self._extract_docx(file_path, doc_id)

        msg = f"Unsupported document extension for DocumentExtractor: {ext}"
        logger.warning("[%s] %s", doc_id, msg)
        return ExtractedDocument(
            document_id=doc_id,
            file_path=str(file_path),
            modality=modality,
            status="skipped_corrupted",
            warnings=[msg],
        )

    def _extract_pdf(self, file_path: Path, doc_id: str) -> ExtractedDocument:
        warnings: List[str] = []
        try:
            with pymupdf.open(file_path) as pdf:
                if pdf.is_encrypted and not pdf.authenticate(""):
                    msg = f"PDF is password-protected/encrypted and cannot be read: {doc_id}"
                    logger.warning("[%s] %s", doc_id, msg)
                    return ExtractedDocument(
                        document_id=doc_id,
                        file_path=str(file_path),
                        modality="pdf",
                        status="skipped_corrupted",
                        warnings=[msg],
                    )

                raw_pages = [page.get_text("text") or "" for page in pdf]
        except Exception as exc:
            msg = f"Failed to parse PDF ({type(exc).__name__}: {exc})"
            logger.warning("[%s] %s", doc_id, msg)
            return ExtractedDocument(
                document_id=doc_id,
                file_path=str(file_path),
                modality="pdf",
                status="skipped_corrupted",
                warnings=[msg],
            )

        cleaned_pages = strip_repeated_headers_footers(raw_pages)
        sentences: List[SentenceItem] = []
        for page_idx, page_text in enumerate(cleaned_pages, start=1):
            for sent in split_sentences(page_text):
                sentences.append(SentenceItem(text=sent, page_number=page_idx))

        raw_text = "\n\n".join(p for p in cleaned_pages if p.strip()).strip()
        if not sentences:
            msg = f"No extractable text found in PDF (blank or scanned image-only): {doc_id}"
            logger.warning("[%s] %s", doc_id, msg)
            return ExtractedDocument(
                document_id=doc_id,
                file_path=str(file_path),
                modality="pdf",
                status="skipped_empty",
                warnings=[msg],
            )

        logger.info("[%s] Extracted PDF: %d pages, %d sentences", doc_id, len(raw_pages), len(sentences))
        return ExtractedDocument(
            document_id=doc_id,
            file_path=str(file_path),
            modality="pdf",
            status="ok",
            raw_text=raw_text,
            sentences=sentences,
            warnings=warnings,
        )

    def _extract_docx(self, file_path: Path, doc_id: str) -> ExtractedDocument:
        warnings: List[str] = []
        try:
            document = docx.Document(str(file_path))
            blocks: List[str] = []

            for para in document.paragraphs:
                cleaned = normalize_text(para.text)
                if cleaned:
                    blocks.append(cleaned)

            for table in document.tables:
                for row in table.rows:
                    row_cells = [normalize_text(cell.text) for cell in row.cells]
                    row_text = " ".join(c for c in row_cells if c)
                    if row_text:
                        blocks.append(row_text)
        except Exception as exc:
            msg = f"Failed to parse DOCX ({type(exc).__name__}: {exc})"
            logger.warning("[%s] %s", doc_id, msg)
            return ExtractedDocument(
                document_id=doc_id,
                file_path=str(file_path),
                modality="docx",
                status="skipped_corrupted",
                warnings=[msg],
            )

        # Merge all blocks into one clean stream to defeat Enter-spam gaming,
        # then split sentences and assign deterministic estimated page numbers.
        raw_text = " ".join(blocks).strip()
        raw_sentences = split_sentences(raw_text)

        sentences: List[SentenceItem] = []
        cumulative_words = 0
        for sent in raw_sentences:
            est_page = 1 + (cumulative_words // _DOCX_WORDS_PER_ESTIMATED_PAGE)
            sentences.append(SentenceItem(text=sent, page_number=est_page))
            cumulative_words += len(sent.split())

        if not sentences:
            msg = f"No extractable text found in DOCX: {doc_id}"
            logger.warning("[%s] %s", doc_id, msg)
            return ExtractedDocument(
                document_id=doc_id,
                file_path=str(file_path),
                modality="docx",
                status="skipped_empty",
                warnings=[msg],
            )

        logger.info("[%s] Extracted DOCX: %d sentences (~%d est. pages)", doc_id, len(sentences), sentences[-1].page_number)
        return ExtractedDocument(
            document_id=doc_id,
            file_path=str(file_path),
            modality="docx",
            status="ok",
            raw_text=raw_text,
            sentences=sentences,
            warnings=warnings,
        )
