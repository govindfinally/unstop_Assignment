"""Shared data models for documents, chunks, matches, and audit reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TextChunk:
    """A normalized text segment with source location metadata."""

    chunk_id: str
    document_id: str
    modality: str  # "pdf", "docx", "mp4", "external"
    text: str
    chunk_index: int
    page_number: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    is_boilerplate: bool = False

    def format_timestamp(self) -> Optional[str]:
        """Format start/end seconds as HH:MM:SS - HH:MM:SS."""
        if self.start_time is None or self.end_time is None:
            return None

        def _fmt(sec: float) -> str:
            total = max(0, int(round(sec)))
            hrs, rem = divmod(total, 3600)
            mins, secs = divmod(rem, 60)
            return f"{hrs:02d}:{mins:02d}:{secs:02d}"

        return f"{_fmt(self.start_time)} - {_fmt(self.end_time)}"


@dataclass
class SentenceItem:
    """A single extracted sentence/segment with source location metadata."""

    text: str
    page_number: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    is_boilerplate: bool = False


@dataclass
class ExtractedDocument:
    """Result of ingesting a single submission file."""

    document_id: str
    file_path: str
    modality: str
    status: str  # "ok", "skipped_empty", "skipped_corrupted", "skipped_no_audio", "error"
    raw_text: str = ""
    sentences: List[SentenceItem] = field(default_factory=list)
    chunks: List[TextChunk] = field(default_factory=list)
    boilerplate_sentences_ignored: int = 0
    warnings: List[str] = field(default_factory=list)


@dataclass
class MatchEvidence:
    """Explainable evidence for a single overlapping passage."""

    matched_file: str
    similarity_score: float
    query_passage: str
    matched_passage: str
    location: Dict[str, Any] = field(default_factory=dict)
    match_type: str = "peer"  # "peer", "multimodal", "external"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "matched_file": self.matched_file,
            "similarity_score": round(float(self.similarity_score), 4),
            "query_passage": self.query_passage,
            "matched_passage": self.matched_passage,
            "location": self.location,
        }


@dataclass
class DocumentReport:
    """Structured audit report matching the assessment JSON schema."""

    document_id: str
    status: str
    total_integrity_risk: str  # "HIGH", "MEDIUM", "LOW", "ERROR"
    peer_similarity_percentage: float = 0.0
    external_similarity_percentage: float = 0.0
    ai_generated_probability: float = 0.0
    peer_matches: List[MatchEvidence] = field(default_factory=list)
    multimodal_matches: List[MatchEvidence] = field(default_factory=list)
    external_matches: List[MatchEvidence] = field(default_factory=list)
    boilerplate_sentences_ignored: int = 0
    warnings: List[str] = field(default_factory=list)
    explanations: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "status": self.status,
            "total_integrity_risk": self.total_integrity_risk,
            "scores": {
                "peer_similarity_percentage": round(float(self.peer_similarity_percentage), 2),
                "external_similarity_percentage": round(float(self.external_similarity_percentage), 2),
                "ai_generated_probability": round(float(self.ai_generated_probability), 2),
            },
            "peer_matches": [m.to_dict() for m in self.peer_matches],
            "multimodal_matches": [m.to_dict() for m in self.multimodal_matches],
            "external_matches": [m.to_dict() for m in self.external_matches],
            "boilerplate_sentences_ignored": self.boilerplate_sentences_ignored,
            "warnings": self.warnings,
            "explanations": self.explanations,
        }
