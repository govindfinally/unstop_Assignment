"""Unit tests for src/indexing/boilerplate.py (Scenario 4 & edge cases)."""

from src.config import Settings
from src.indexing.boilerplate import flag_boilerplate_sentences
from src.models import ExtractedDocument, SentenceItem


def _make_doc(doc_id: str, sentences: list[str], status: str = "ok") -> ExtractedDocument:
    return ExtractedDocument(
        document_id=doc_id,
        file_path=f"./{doc_id}",
        modality="pdf",
        status=status,
        raw_text=" ".join(sentences),
        sentences=[SentenceItem(text=s, page_number=1) for s in sentences],
    )


def test_scenario4_shared_prompt_is_flagged_while_original_answers_survive():
    prompt = [
        "Analyze how global supply chain bottlenecks impacted Q3 operating margins across retail sectors.",
        "Provide three actionable inventory recommendations supported by quantitative trade-off metrics.",
    ]
    docs = [
        _make_doc(f"honest_{i}.pdf", prompt + [f"Unique analytical insight number {i} for warehouse automation."])
        for i in range(5)
    ]

    total = flag_boilerplate_sentences(docs, Settings())
    assert total == 10  # 2 prompt sentences * 5 docs
    for doc in docs:
        assert doc.boilerplate_sentences_ignored == 2
        assert doc.sentences[0].is_boilerplate is True
        assert doc.sentences[1].is_boilerplate is True
        assert doc.sentences[2].is_boilerplate is False


def test_two_person_cheating_pair_is_not_mistaken_for_boilerplate():
    copied_sent = "Vertical integration across regional cold storage hubs reduced spoilage by twenty percent."
    docs = [
        _make_doc("cheater_a.pdf", [copied_sent, "Extra unique sentence from student A."]),
        _make_doc("cheater_b.pdf", [copied_sent, "Extra unique sentence from student B."]),
    ] + [
        _make_doc(f"clean_{i}.pdf", [f"Completely independent original analysis from candidate {i} on fintech."])
        for i in range(8)
    ]

    total = flag_boilerplate_sentences(docs, Settings())
    assert total == 0
    assert docs[0].sentences[0].is_boilerplate is False
    assert docs[1].sentences[0].is_boilerplate is False


def test_sentence_repeated_inside_single_document_only_counts_once():
    repeated = "This exact sentence is repeated many times inside a single student document."
    docs = [
        _make_doc("spammer.pdf", [repeated] * 10),
        _make_doc("other_1.pdf", ["Completely unrelated financial analysis on quarterly cash flows."]),
        _make_doc("other_2.pdf", ["Another unrelated operational study on last mile delivery fleets."]),
        _make_doc("other_3.pdf", ["A third distinct report focusing on renewable energy procurement."]),
    ]

    total = flag_boilerplate_sentences(docs, Settings())
    assert total == 0
    assert all(not s.is_boilerplate for s in docs[0].sentences)


def test_short_sentences_and_small_batches_are_safely_ignored():
    # Short sentences (< 6 words) should not be counted
    docs = [_make_doc(f"d{i}.pdf", ["Thank you very much.", f"Original long sentence number {i} here."]) for i in range(4)]
    assert flag_boilerplate_sentences(docs, Settings()) == 0

    # Small batch (< 3 valid docs) should skip boilerplate filtering
    small_batch = [
        _make_doc("a.pdf", ["Identical long shared sentence across only two documents in the folder."]),
        _make_doc("b.pdf", ["Identical long shared sentence across only two documents in the folder."]),
    ]
    assert flag_boilerplate_sentences(small_batch, Settings()) == 0


def test_punctuation_and_case_variants_still_match_as_boilerplate():
    docs = [
        _make_doc("d1.pdf", ["Evaluate the Q3 Supply-Chain Delays & Margin Impact!"]),
        _make_doc("d2.pdf", ["evaluate the q3 supply chain delays   margin impact."]),
        _make_doc("d3.pdf", ["Evaluate the Q3 supply-chain delays, margin impact?"]),
        _make_doc("d4.pdf", ["Unique non-matching sentence about customer retention metrics."]),
    ]
    total = flag_boilerplate_sentences(docs, Settings())
    assert total == 3
    assert docs[0].sentences[0].is_boilerplate is True
    assert docs[1].sentences[0].is_boilerplate is True
    assert docs[2].sentences[0].is_boilerplate is True
    assert docs[3].sentences[0].is_boilerplate is False
