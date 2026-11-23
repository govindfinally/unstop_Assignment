"""MP4 audio extraction and ASR transcription with timestamp & filler handling.

Uses PyAV to validate container/audio tracks before lazy-loading faster-whisper
(base.en, int8 CPU), cleans spoken fillers (uh, um, ah, er, hmm), and provides
pause/word-aware timestamp chunking with 1-segment overlap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import av

from src.config import Settings
from src.extractors.base import BaseExtractor
from src.extractors.normalizer import clean_spoken_fillers, normalize_text
from src.logger import get_logger
from src.models import ExtractedDocument, SentenceItem, TextChunk

logger = get_logger("extractors.video")


class VideoExtractor(BaseExtractor):
    """Fault-tolerant MP4 extractor using PyAV + faster-whisper."""

    def __init__(
        self,
        settings: Settings,
        whisper_model: Optional[Any] = None,
    ) -> None:
        self.settings = settings
        self._model = whisper_model

    def _get_model(self) -> Any:
        """Lazy-load faster-whisper once so multiple videos share the same model."""
        if self._model is None:
            from faster_whisper import WhisperModel

            logger.info(
                "Loading faster-whisper model '%s' (device=%s, compute_type=%s)",
                self.settings.whisper_model_size,
                self.settings.whisper_device,
                self.settings.whisper_compute_type,
            )
            self._model = WhisperModel(
                self.settings.whisper_model_size,
                device=self.settings.whisper_device,
                compute_type=self.settings.whisper_compute_type,
            )
        return self._model

    def _has_audio_stream(self, file_path: Path) -> tuple[bool, Optional[str]]:
        """Inspect MP4 container headers via PyAV in O(1) without decoding video."""
        try:
            with av.open(str(file_path)) as container:
                if len(container.streams.audio) == 0:
                    return False, None
                return True, None
        except Exception as exc:
            return False, f"Failed to open MP4 container ({type(exc).__name__}: {exc})"

    def extract(self, file_path: Path) -> ExtractedDocument:
        """Validate MP4 audio, transcribe speech, clean fillers, and emit SentenceItems.

        TC: O(1) header inspection + O(T) ASR transcription + O(S + W) cleaning.
        SC: O(M + W) where M is quantized Whisper model RAM and W is transcript words.
        """
        doc_id = file_path.name

        if not file_path.exists() or not file_path.is_file():
            msg = f"Video file not found: {file_path}"
            logger.warning("[%s] %s", doc_id, msg)
            return ExtractedDocument(
                document_id=doc_id,
                file_path=str(file_path),
                modality="mp4",
                status="skipped_corrupted",
                warnings=[msg],
            )

        if file_path.stat().st_size == 0:
            msg = f"Video file is empty (0 bytes): {doc_id}"
            logger.warning("[%s] %s", doc_id, msg)
            return ExtractedDocument(
                document_id=doc_id,
                file_path=str(file_path),
                modality="mp4",
                status="skipped_empty",
                warnings=[msg],
            )

        has_audio, container_err = self._has_audio_stream(file_path)
        if container_err is not None:
            logger.warning("[%s] %s", doc_id, container_err)
            return ExtractedDocument(
                document_id=doc_id,
                file_path=str(file_path),
                modality="mp4",
                status="skipped_corrupted",
                warnings=[container_err],
            )

        if not has_audio:
            msg = f"MP4 file contains no audio track: {doc_id}"
            logger.warning("[%s] %s", doc_id, msg)
            return ExtractedDocument(
                document_id=doc_id,
                file_path=str(file_path),
                modality="mp4",
                status="skipped_no_audio",
                warnings=[msg],
            )

        try:
            model = self._get_model()
            segments, _info = model.transcribe(str(file_path), vad_filter=True)
            sentences: List[SentenceItem] = []

            for seg in segments:
                cleaned = clean_spoken_fillers(normalize_text(getattr(seg, "text", "") or ""))
                if not cleaned:
                    continue
                sentences.append(
                    SentenceItem(
                        text=cleaned,
                        page_number=None,
                        start_time=float(getattr(seg, "start", 0.0)),
                        end_time=float(getattr(seg, "end", 0.0)),
                    )
                )
        except Exception as exc:
            msg = f"ASR transcription failed ({type(exc).__name__}: {exc})"
            logger.warning("[%s] %s", doc_id, msg)
            return ExtractedDocument(
                document_id=doc_id,
                file_path=str(file_path),
                modality="mp4",
                status="skipped_corrupted",
                warnings=[msg],
            )

        if not sentences:
            msg = f"MP4 audio track is silent or contains no recognizable speech: {doc_id}"
            logger.warning("[%s] %s", doc_id, msg)
            return ExtractedDocument(
                document_id=doc_id,
                file_path=str(file_path),
                modality="mp4",
                status="skipped_no_audio",
                warnings=[msg],
            )

        raw_text = " ".join(s.text for s in sentences).strip()
        logger.info("[%s] Transcribed MP4: %d speech segments", doc_id, len(sentences))
        return ExtractedDocument(
            document_id=doc_id,
            file_path=str(file_path),
            modality="mp4",
            status="ok",
            raw_text=raw_text,
            sentences=sentences,
        )


def build_video_chunks(
    document_id: str,
    segments: List[SentenceItem],
    max_words: int = 100,
    pause_threshold_sec: float = 2.0,
    min_chunk_words: int = 6,
) -> List[TextChunk]:
    """Group non-boilerplate timestamped segments into overlapping TextChunks.

    Splits when window reaches max_words OR pause between consecutive segments
    exceeds pause_threshold_sec, retaining 1-segment overlap across word-limit splits.

    DSA: Sliding Segment Window.
    TC: O(S + W), SC: O(W).
    """
    active = [s for s in segments if not s.is_boilerplate and s.text.strip()]
    if not active:
        return []

    chunks: List[TextChunk] = []
    n = len(active)
    start_idx = 0

    while start_idx < n:
        end_idx = start_idx
        word_count = 0
        split_on_pause = False

        while end_idx < n:
            if end_idx > start_idx:
                prev_end = active[end_idx - 1].end_time
                curr_start = active[end_idx].start_time
                if (
                    prev_end is not None
                    and curr_start is not None
                    and (curr_start - prev_end) > pause_threshold_sec
                ):
                    split_on_pause = True
                    break

            seg_words = len(active[end_idx].text.split())
            if word_count > 0 and word_count + seg_words > max_words:
                break

            word_count += seg_words
            end_idx += 1

        if end_idx == start_idx:
            end_idx += 1
            word_count = len(active[start_idx].text.split())

        chunk_text = " ".join(active[i].text for i in range(start_idx, end_idx)).strip()
        if word_count >= min_chunk_words or not chunks:
            chunks.append(
                TextChunk(
                    chunk_id=f"{document_id}::chunk_{len(chunks)}",
                    document_id=document_id,
                    modality="mp4",
                    text=chunk_text,
                    chunk_index=len(chunks),
                    page_number=None,
                    start_time=active[start_idx].start_time,
                    end_time=active[end_idx - 1].end_time,
                )
            )

        if end_idx >= n:
            break

        # On natural long pauses, start fresh; on word-limit splits, keep 1-segment overlap
        if split_on_pause or (end_idx - start_idx) <= 1:
            start_idx = end_idx
        else:
            start_idx = end_idx - 1

    return chunks
