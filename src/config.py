"""Centralized configuration and tunable thresholds for the integrity engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass
class Settings:
    """Runtime settings for ingestion, indexing, detection, and reporting."""

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    external_kb_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "external_knowledge"
    )
    log_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "logs"
    )
    log_file_name: str = "integrity_engine.log"
    log_level: str = "INFO"

    # Supported file formats
    supported_extensions: Tuple[str, ...] = (".pdf", ".docx", ".mp4")

    # Task 1: Chunking & Normalization (target 200-400 tokens -> ~150-250 words)
    chunk_size_words: int = 120
    chunk_overlap_words: int = 25
    min_chunk_words: int = 12

    # Audio / Video Transcription (Whisper) - base.en chosen for accuracy over speed (D-006)
    whisper_model_size: str = "base.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # Task 2: Embeddings & Peer Similarity
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    peer_similarity_threshold: float = 0.78
    cross_modal_similarity_threshold: float = 0.72
    top_k_matches: int = 5

    # Scenario 4: Sentence-Level Boilerplate / Shared Prompt Discounting (D-007)
    boilerplate_min_doc_count: int = 3
    boilerplate_min_doc_ratio: float = 0.25
    boilerplate_min_sentence_words: int = 6

    # Task 3: External & AI Signals
    external_similarity_threshold: float = 0.75
    ai_classifier_model: str = "roberta-base-openai-detector"
    use_hf_ai_classifier: bool = False  # Set True to download HF RoBERTa; False uses fast heuristic + burstiness

    # Risk Band Thresholds (0-100 scale)
    high_risk_peer_pct: float = 40.0
    medium_risk_peer_pct: float = 18.0
    high_risk_external_pct: float = 40.0
    medium_risk_external_pct: float = 18.0
    high_risk_ai_pct: float = 75.0
    medium_risk_ai_pct: float = 50.0


DEFAULT_SETTINGS = Settings()
