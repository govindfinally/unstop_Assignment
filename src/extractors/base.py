"""Base interface for modality-specific extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.models import ExtractedDocument


class BaseExtractor(ABC):
    """Abstract base class for all file format extractors."""

    @abstractmethod
    def extract(self, file_path: Path) -> ExtractedDocument:
        """Extract raw text and metadata-preserving chunks from a file."""
        raise NotImplementedError
