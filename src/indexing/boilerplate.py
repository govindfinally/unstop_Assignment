"""Sentence-level normalized boilerplate & shared exam prompt detector (Scenario 4).

Runs BEFORE sliding-window chunking so shared competition prompts are stripped
cleanly without causing chunk boundary shifts or distorting AI-detection scores.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import List, Set

from src.config import Settings
from src.extractors.normalizer import normalize_sentence_for_boilerplate
from src.logger import get_logger
from src.models import ExtractedDocument

logger = get_logger("indexing.boilerplate")


def flag_boilerplate_sentences(
    documents: List[ExtractedDocument],
    settings: Settings,
) -> int:
    """Identify sentences repeated verbatim across >= max(min_count, ratio * N) valid docs.

    DSA: Per-document Hash Set + corpus-wide Hash Map (Counter) + O(1) lookup Hash Set.
    TC: O(N_chars) strictly linear across all sentences in the batch.
    SC: O(U) where U is the number of unique canonical sentences.

    Returns:
        Total number of SentenceItem instances tagged as boilerplate across all documents.
    """
    valid_docs = [doc for doc in documents if doc.status == "ok" and doc.sentences]
    num_valid = len(valid_docs)

    if num_valid < settings.boilerplate_min_doc_count:
        logger.info(
            "Skipping boilerplate filter: only %d valid doc(s) (< min %d)",
            num_valid,
            settings.boilerplate_min_doc_count,
        )
        return 0

    min_doc_threshold = max(
        settings.boilerplate_min_doc_count,
        math.ceil(num_valid * settings.boilerplate_min_doc_ratio),
    )
    min_words = settings.boilerplate_min_sentence_words

    # Pass 1: Count in how many DISTINCT documents each normalized sentence appears
    doc_freq: Counter[str] = Counter()
    for doc in valid_docs:
        seen_in_doc: Set[str] = set()
        for item in doc.sentences:
            canon = normalize_sentence_for_boilerplate(item.text)
            if len(canon.split()) < min_words:
                continue
            seen_in_doc.add(canon)
        for canon in seen_in_doc:
            doc_freq[canon] += 1

    boilerplate_set: Set[str] = {
        canon for canon, count in doc_freq.items() if count >= min_doc_threshold
    }

    if not boilerplate_set:
        logger.info("No shared boilerplate sentences detected across %d valid docs", num_valid)
        return 0

    # Pass 2: Tag matching SentenceItems across all valid documents
    total_flagged = 0
    for doc in valid_docs:
        doc_flagged = 0
        for item in doc.sentences:
            canon = normalize_sentence_for_boilerplate(item.text)
            if canon in boilerplate_set:
                item.is_boilerplate = True
                doc_flagged += 1
        doc.boilerplate_sentences_ignored = doc_flagged
        total_flagged += doc_flagged
        if doc_flagged > 0:
            logger.debug(
                "[%s] Marked %d sentence(s) as shared boilerplate",
                doc.document_id,
                doc_flagged,
            )

    logger.info(
        "Boilerplate filter flagged %d unique prompt sentence(s) (%d total occurrences) "
        "using threshold >= %d/%d docs",
        len(boilerplate_set),
        total_flagged,
        min_doc_threshold,
        num_valid,
    )
    return total_flagged
