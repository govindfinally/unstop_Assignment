"""Text cleaning, header/footer stripping, filler removal, and sentence-aware chunking."""

from __future__ import annotations

from collections import Counter
import math
import re
from typing import List, Optional, Tuple

from src.logger import get_logger

logger = get_logger("extractors.normalizer")


# Precompiled regexes for O(L) single-pass text cleaning
_CONTROL_CHARS_RE = re.compile(r"[\x00\ufeff\u200b\u200c\u200d]")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_DIGIT_PLACEHOLDER_RE = re.compile(r"\d+")

# Standalone spoken disfluencies (won't match inside words like "summary" or "earth")
_FILLER_WORDS_RE = re.compile(
    r"(?i)\b(?:uh+|um+|ah+|er+|hmm+|erm+)\b[\s,]*"
)
_ORPHAN_COMMA_RE = re.compile(r"\s*,\s*,+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")

# Sentence boundary splitter protecting common abbreviations and decimals
_ABBREV_RE = re.compile(
    r"\b(e\.g|i\.e|Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|vs|etc|Fig|No|St|U\.S)\.",
    re.IGNORECASE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_BOILERPLATE_NORM_RE = re.compile(r"[^a-z0-9\s]")


def normalize_text(text: str) -> str:
    """Clean control characters, line endings, and redundant whitespace.

    TC: O(L), SC: O(L) where L is character length.
    """
    if not text:
        return ""
    cleaned = _CONTROL_CHARS_RE.sub("", text)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_MULTI_SPACE_RE.sub(" ", line).strip() for line in cleaned.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = _MULTI_NEWLINE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def _header_footer_signature(line: str) -> str:
    """Normalize digits so 'Page 1 of 5' and 'Page 2 of 5' share one signature."""
    return _DIGIT_PLACEHOLDER_RE.sub("#", line.strip().lower())


def strip_repeated_headers_footers(
    pages: List[str],
    min_ratio: float = 0.60,
    max_line_words: int = 12,
) -> List[str]:
    """Remove short top/bottom lines repeated across >= min_ratio of PDF pages.

    Uses a Hash Map (Counter) + Hash Set for O(1) lookup.
    Only activates when len(pages) >= 3 to avoid stripping real content on short docs.

    TC: O(N_lines), SC: O(U) where U is unique candidate top/bottom lines.
    """
    normalized_pages = [normalize_text(p) for p in pages]
    num_pages = len(normalized_pages)
    if num_pages < 3:
        return normalized_pages

    line_doc_counts: Counter[str] = Counter()
    page_lines_list: List[List[str]] = []

    for page in normalized_pages:
        lines = [ln for ln in page.split("\n") if ln.strip()]
        page_lines_list.append(lines)
        if not lines:
            continue

        candidates = set()
        for edge_line in (lines[0], lines[-1]):
            if len(edge_line.split()) <= max_line_words:
                candidates.add(_header_footer_signature(edge_line))
        for sig in candidates:
            line_doc_counts[sig] += 1

    threshold = math.ceil(num_pages * min_ratio)
    repeated_sigs = {sig for sig, count in line_doc_counts.items() if count >= threshold}
    if not repeated_sigs:
        return normalized_pages

    logger.debug(
        "Stripping %d repeated header/footer pattern(s) across %d pages",
        len(repeated_sigs),
        num_pages,
    )

    cleaned_pages: List[str] = []
    for lines in page_lines_list:
        if not lines:
            cleaned_pages.append("")
            continue
        start = 0
        end = len(lines)
        if len(lines[0].split()) <= max_line_words and _header_footer_signature(lines[0]) in repeated_sigs:
            start = 1
        if end > start and len(lines[-1].split()) <= max_line_words and _header_footer_signature(lines[-1]) in repeated_sigs:
            end -= 1
        cleaned_pages.append("\n".join(lines[start:end]).strip())

    return cleaned_pages


def clean_spoken_fillers(text: str) -> str:
    """Remove standalone spoken disfluencies (uh, um, ah, er, hmm, erm) from ASR text.

    TC: O(L), SC: O(L).
    """
    if not text:
        return ""
    cleaned = _FILLER_WORDS_RE.sub(" ", text)
    cleaned = _ORPHAN_COMMA_RE.sub(",", cleaned)
    cleaned = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", cleaned)
    cleaned = _MULTI_SPACE_RE.sub(" ", cleaned)
    return cleaned.strip(" ,")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences deterministically, ignoring arbitrary line breaks.

    Protects decimals (4.2%) and common abbreviations (e.g., Dr., Q3).
    TC: O(L), SC: O(L).
    """
    cleaned = normalize_text(text)
    if not cleaned:
        return []

    # Collapse line breaks so student Enter-spam cannot game chunk boundaries
    flat = _MULTI_SPACE_RE.sub(" ", cleaned.replace("\n", " ")).strip()
    if not flat:
        return []

    placeholder = "<DOT>"
    protected = _ABBREV_RE.sub(lambda m: m.group(1) + placeholder, flat)
    raw_sentences = _SENTENCE_SPLIT_RE.split(protected)

    sentences: List[str] = []
    for seg in raw_sentences:
        restored = seg.replace(placeholder, ".").strip()
        if restored:
            sentences.append(restored)
    return sentences


def normalize_sentence_for_boilerplate(sentence: str) -> str:
    """Canonicalize a sentence for O(1) hash-based boilerplate frequency counting.

    TC: O(L), SC: O(L).
    """
    lowered = sentence.lower()
    alnum_only = _BOILERPLATE_NORM_RE.sub(" ", lowered)
    return _MULTI_SPACE_RE.sub(" ", alnum_only).strip()


def _split_long_sentence(
    sentence: str,
    max_words: int,
    overlap_words: int,
) -> List[str]:
    """Fallback splitter for giant run-on sentences with no punctuation."""
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]

    step = max(1, max_words - min(overlap_words, max_words - 1))
    pieces: List[str] = []
    for start in range(0, len(words), step):
        end = min(len(words), start + max_words)
        pieces.append(" ".join(words[start:end]))
        if end == len(words):
            break
    return pieces


def sentence_aware_sliding_window_with_meta(
    sentence_items: List[Tuple[str, Optional[int]]],
    chunk_size_words: int = 120,
    overlap_words: int = 25,
    min_chunk_words: int = 12,
) -> List[Tuple[str, Optional[int]]]:
    """Group (sentence, page_number) items into overlapping sentence-aware chunks.

    DSA: Two-Pointer Sliding Window + precomputed word_counts array.
    TC: O(S + W) amortized linear (each sentence enters and leaves window <= once).
    SC: O(W) for output chunks.
    """
    if not sentence_items:
        return []

    chunk_size_words = max(1, chunk_size_words)
    overlap_words = max(0, min(overlap_words, chunk_size_words - 1))

    # Expand any giant punctuation-free run-on sentences safely
    expanded: List[Tuple[str, Optional[int]]] = []
    for sent, page_num in sentence_items:
        s = sent.strip()
        if not s:
            continue
        for piece in _split_long_sentence(s, chunk_size_words, overlap_words):
            expanded.append((piece, page_num))

    if not expanded:
        return []

    word_counts = [len(item[0].split()) for item in expanded]
    n = len(expanded)
    chunks: List[Tuple[str, Optional[int]]] = []

    left = 0
    right = 0
    window_words = 0

    while left < n:
        while right < n and (window_words < chunk_size_words or right == left):
            window_words += word_counts[right]
            right += 1

        chunk_text = " ".join(expanded[i][0] for i in range(left, right))
        chunk_page = expanded[left][1]

        if window_words >= min_chunk_words or not chunks:
            chunks.append((chunk_text, chunk_page))

        if right >= n:
            break

        # Advance left pointer while keeping up to overlap_words from the tail
        prev_left = left
        while left < right and window_words > overlap_words:
            window_words -= word_counts[left]
            left += 1

        # Guarantee forward progress even if a single sentence >= overlap_words
        if left == prev_left:
            window_words -= word_counts[left]
            left += 1

    logger.debug("Created %d sentence-aware chunk(s) from %d sentence(s)", len(chunks), n)
    return chunks


def sentence_aware_sliding_window(
    sentences: List[str],
    chunk_size_words: int = 120,
    overlap_words: int = 25,
    min_chunk_words: int = 12,
) -> List[str]:
    """Convenience wrapper returning only chunk strings without page metadata."""
    items = [(s, None) for s in sentences]
    return [
        text
        for text, _ in sentence_aware_sliding_window_with_meta(
            items,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
            min_chunk_words=min_chunk_words,
        )
    ]
