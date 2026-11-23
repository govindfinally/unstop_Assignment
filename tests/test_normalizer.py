"""Unit tests for src/extractors/normalizer.py covering core behavior & edge cases."""

from src.extractors.normalizer import (
    clean_spoken_fillers,
    normalize_sentence_for_boilerplate,
    normalize_text,
    sentence_aware_sliding_window,
    sentence_aware_sliding_window_with_meta,
    split_sentences,
    strip_repeated_headers_footers,
)


def test_normalize_text_cleans_whitespace_and_control_chars():
    raw = "\ufeffHello\x00   world!\r\n\r\n\r\nNext\t\tline.  "
    assert normalize_text(raw) == "Hello world!\n\nNext line."
    assert normalize_text("") == ""
    assert normalize_text("   \n\t  ") == ""


def test_strip_repeated_headers_footers_handles_page_numbers_and_short_docs():
    # 4-page PDF with repeating header and incrementing page footer
    pages = [
        "Unstop Case Challenge 2026\nBody page one content.\nPage 1 of 4",
        "Unstop Case Challenge 2026\nBody page two content.\nPage 2 of 4",
        "Unstop Case Challenge 2026\nBody page three content.\nPage 3 of 4",
        "Unstop Case Challenge 2026\nBody page four content.\nPage 4 of 4",
    ]
    cleaned = strip_repeated_headers_footers(pages, min_ratio=0.60)
    assert cleaned == [
        "Body page one content.",
        "Body page two content.",
        "Body page three content.",
        "Body page four content.",
    ]

    # 2-page PDF should NOT strip lines (below 3-page minimum guard)
    short_pages = ["Title Line\nContent A.", "Title Line\nContent B."]
    assert strip_repeated_headers_footers(short_pages) == [
        "Title Line\nContent A.",
        "Title Line\nContent B.",
    ]


def test_clean_spoken_fillers_preserves_real_words():
    transcript = "Um, operating margins, uh, dropped 4.2% in our summary of earth resources, ah, hmm."
    cleaned = clean_spoken_fillers(transcript)
    assert "summary" in cleaned
    assert "earth" in cleaned
    assert "4.2%" in cleaned
    assert "um" not in cleaned.lower().split()
    assert "uh" not in cleaned.lower().split()
    assert "hmm" not in cleaned.lower()


def test_split_sentences_protects_abbreviations_and_defeats_enter_spam():
    text = (
        "Dr. Smith noted e.g. margins fell 4.2%.\n\n\n"
        "Supply\nchains\nwere\ndelayed\nin\nQ3.\n"
        "Revenue recovered strongly!"
    )
    sents = split_sentences(text)
    assert len(sents) == 3
    assert "Dr. Smith" in sents[0]
    assert "e.g." in sents[0]
    assert "4.2%." in sents[0]
    assert sents[1] == "Supply chains were delayed in Q3."
    assert sents[2] == "Revenue recovered strongly!"


def test_normalize_sentence_for_boilerplate():
    s1 = "Analyze the Q3 Supply-Chain Delays & Margins!"
    s2 = "analyze the q3 supply chain delays   margins."
    assert normalize_sentence_for_boilerplate(s1) == normalize_sentence_for_boilerplate(s2)


def test_sliding_window_overlap_and_run_on_sentence_edge_case():
    # Empty input
    assert sentence_aware_sliding_window([]) == []

    # Normal multi-sentence sliding window with overlap
    sents = [f"Sentence number {i} has several meaningful words for testing." for i in range(12)]
    chunks = sentence_aware_sliding_window(sents, chunk_size_words=30, overlap_words=10, min_chunk_words=5)
    assert len(chunks) >= 3
    # Verify consecutive chunks share at least one overlapping sentence
    assert any(word in chunks[1] for word in chunks[0].split()[-8:])

    # Giant 250-word run-on sentence with NO periods must sub-split safely without infinite loop
    giant = " ".join(f"word{i}" for i in range(250))
    giant_chunks = sentence_aware_sliding_window([giant], chunk_size_words=100, overlap_words=20, min_chunk_words=10)
    assert len(giant_chunks) == 3
    assert len(giant_chunks[0].split()) == 100


def test_sliding_window_with_meta_preserves_page_numbers():
    items = [
        ("First page sentence one with enough words to count properly.", 1),
        ("First page sentence two with enough words to count properly.", 1),
        ("Second page sentence three with enough words to count properly.", 2),
        ("Second page sentence four with enough words to count properly.", 2),
    ]
    chunks = sentence_aware_sliding_window_with_meta(
        items, chunk_size_words=18, overlap_words=6, min_chunk_words=5
    )
    assert chunks[0][1] == 1
    assert any(page == 2 for _, page in chunks)
