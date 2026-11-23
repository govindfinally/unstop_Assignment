"""Unit tests for src/extractors/video_extractor.py covering edge cases & timestamps."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import av
import numpy as np

from src.config import Settings
from src.extractors.video_extractor import VideoExtractor, build_video_chunks
from src.models import SentenceItem


@dataclass
class _FakeSegment:
    text: str
    start: float
    end: float


class _FakeWhisperModel:
    def __init__(self, segments: List[_FakeSegment]) -> None:
        self._segments = segments

    def transcribe(self, _path: str, vad_filter: bool = True):
        return iter(self._segments), {"language": "en"}


def _create_video_only_mp4(path: Path) -> None:
    """Create a tiny valid MP4 with a video stream and NO audio stream."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = 64
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        for _ in range(5):
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _create_silent_audio_mp4(path: Path) -> None:
    """Create a tiny valid MP4 with a silent AAC audio stream."""
    with av.open(str(path), mode="w") as container:
        vstream = container.add_stream("mpeg4", rate=10)
        vstream.width = 64
        vstream.height = 64
        vstream.pix_fmt = "yuv420p"
        astream = container.add_stream("aac", rate=16000)

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        for _ in range(5):
            vframe = av.VideoFrame.from_ndarray(img, format="rgb24")
            for packet in vstream.encode(vframe):
                container.mux(packet)
        for packet in vstream.encode():
            container.mux(packet)

        samples = np.zeros((1, 1024), dtype=np.float32)
        for _ in range(8):
            aframe = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
            aframe.sample_rate = 16000
            for packet in astream.encode(aframe):
                container.mux(packet)
        for packet in astream.encode():
            container.mux(packet)


def test_zero_byte_and_corrupted_mp4_are_handled_gracefully(tmp_path: Path):
    extractor = VideoExtractor(Settings(), whisper_model=_FakeWhisperModel([]))

    empty_mp4 = tmp_path / "empty.mp4"
    empty_mp4.write_bytes(b"")
    res_empty = extractor.extract(empty_mp4)
    assert res_empty.status == "skipped_empty"
    assert len(res_empty.warnings) == 1

    bad_mp4 = tmp_path / "corrupt.mp4"
    bad_mp4.write_bytes(b"not a valid mp4 container")
    res_bad = extractor.extract(bad_mp4)
    assert res_bad.status == "skipped_corrupted"
    assert len(res_bad.warnings) == 1


def test_video_only_mp4_without_audio_stream_skips_without_loading_whisper(tmp_path: Path):
    video_only = tmp_path / "no_audio.mp4"
    _create_video_only_mp4(video_only)

    # Passing whisper_model=None proves PyAV catches missing audio before Whisper loads
    extractor = VideoExtractor(Settings(), whisper_model=None)
    res = extractor.extract(video_only)

    assert res.status == "skipped_no_audio"
    assert "no audio track" in res.warnings[0].lower()


def test_silent_audio_mp4_returns_skipped_no_audio(tmp_path: Path):
    silent_mp4 = tmp_path / "silent.mp4"
    _create_silent_audio_mp4(silent_mp4)

    extractor = VideoExtractor(Settings(), whisper_model=_FakeWhisperModel([]))
    res = extractor.extract(silent_mp4)

    assert res.status == "skipped_no_audio"
    assert "silent" in res.warnings[0].lower()


def test_spoken_mp4_cleans_fillers_and_preserves_timestamps(tmp_path: Path):
    mp4_path = tmp_path / "pitch.mp4"
    _create_silent_audio_mp4(mp4_path)

    fake_segments = [
        _FakeSegment("Um, operating margins, uh, contracted by 4.2 percent in Q3.", 0.0, 5.5),
        _FakeSegment("Ah, this was driven by regional supply chain bottlenecks.", 5.8, 11.2),
        _FakeSegment("Hmm, after a long pause, we recommend vertical integration.", 15.0, 21.0),
    ]
    extractor = VideoExtractor(Settings(), whisper_model=_FakeWhisperModel(fake_segments))
    res = extractor.extract(mp4_path)

    assert res.status == "ok"
    assert len(res.sentences) == 3
    assert "um" not in res.sentences[0].text.lower().split()
    assert "uh" not in res.sentences[0].text.lower().split()
    assert "4.2 percent" in res.sentences[0].text

    chunks = build_video_chunks("pitch.mp4", res.sentences, max_words=100, pause_threshold_sec=2.0)
    # Pause between 11.2s and 15.0s (> 2.0s) must split into 2 clean timestamped chunks
    assert len(chunks) == 2
    assert chunks[0].format_timestamp() == "00:00:00 - 00:00:11"
    assert chunks[1].format_timestamp() == "00:00:15 - 00:00:21"


def test_build_video_chunks_respects_word_limit_overlap_and_boilerplate():
    segs = [
        SentenceItem("Shared competition prompt text ignored.", start_time=0.0, end_time=3.0, is_boilerplate=True),
        SentenceItem("Segment one contains several important words about supply chains.", start_time=3.5, end_time=7.0),
        SentenceItem("Segment two adds further analysis of regional warehouse costs.", start_time=7.2, end_time=11.0),
        SentenceItem("Segment three concludes with projected margin recovery next quarter.", start_time=11.5, end_time=15.0),
    ]
    chunks = build_video_chunks("clip.mp4", segs, max_words=18, pause_threshold_sec=2.0, min_chunk_words=5)
    assert len(chunks) == 2
    assert "Shared competition prompt" not in chunks[0].text
    # 1-segment overlap: segment two appears in both chunk 0 and chunk 1
    assert "Segment two" in chunks[0].text
    assert "Segment two" in chunks[1].text
