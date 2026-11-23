"""End-to-end integration tests for ReportBuilder and IntegrityPipeline."""

import json
from pathlib import Path

import docx
from fpdf import FPDF
import numpy as np

from src.config import Settings
from src.indexing.embedder import TextEmbedder
from src.models import DocumentReport
from src.pipeline import IntegrityPipeline
from src.reporting.report_builder import ReportBuilder


class _DeterministicTopicEmbedder:
    """Fast deterministic embedder for integration testing without downloading weights."""

    def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True):
        keywords = [
            "vertical integration",
            "cold storage",
            "solar microgrid",
            "fintech credit",
            "urban mobility",
            "healthcare diagnostics",
        ]
        rows = []
        for t in texts:
            low = t.lower()
            vec = [1.0 if kw in low else 0.02 for kw in keywords]
            # Add tiny hash-derived tie-breaker only if no keyword matched
            if all(v < 0.5 for v in vec):
                h = abs(hash(low)) % 100
                vec.append(1.0 + (h / 1000.0))
            else:
                vec.append(0.05)
            arr = np.array(vec, dtype=np.float32)
            arr /= np.linalg.norm(arr)
            rows.append(arr)
        return np.vstack(rows)


def _write_pdf(path: Path, text: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 8, text)
    pdf.output(str(path))


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(str(path))


def test_same_stem_files_do_not_overwrite_each_other(tmp_path: Path):
    builder = ReportBuilder()
    reports = [
        DocumentReport(document_id="case.pdf", status="ok", total_integrity_risk="LOW"),
        DocumentReport(document_id="case.docx", status="ok", total_integrity_risk="HIGH", peer_similarity_percentage=85.0),
    ]
    builder.write_all(reports, tmp_path)

    assert (tmp_path / "case.pdf.json").exists()
    assert (tmp_path / "case.docx.json").exists()
    assert (tmp_path / "case.pdf.md").exists()
    assert (tmp_path / "case.docx.md").exists()
    assert (tmp_path / "summary_matrix.md").exists()
    assert (tmp_path / "summary_matrix.json").exists()

    summary_md = (tmp_path / "summary_matrix.md").read_text(encoding="utf-8")
    # HIGH risk doc must appear before LOW risk doc in summary matrix
    assert summary_md.index("case.docx") < summary_md.index("case.pdf")


def test_empty_input_directory_exits_cleanly(tmp_path: Path):
    in_dir = tmp_path / "empty_in"
    out_dir = tmp_path / "empty_out"
    in_dir.mkdir()

    settings = Settings(external_kb_dir=tmp_path / "no_kb")
    pipeline = IntegrityPipeline(settings, embedder=TextEmbedder(settings, model=_DeterministicTopicEmbedder()))
    reports = pipeline.run(in_dir, out_dir)

    assert reports == []
    assert (out_dir / "summary_matrix.md").exists()


def test_end_to_end_mixed_batch_handles_copies_boilerplate_and_corrupted_files(tmp_path: Path):
    in_dir = tmp_path / "submissions"
    out_dir = tmp_path / "reports"
    kb_dir = tmp_path / "kb"
    in_dir.mkdir()
    kb_dir.mkdir()

    shared_prompt = (
        "Analyze the quarterly operational trade-offs and propose a measurable strategic roadmap for leadership. "
        "Support every recommendation with concrete financial and logistics metrics."
    )

    # 1. Exact copy pair (PDF + DOCX)
    copied_body = (
        "Our vertical integration strategy consolidates regional suppliers to protect operating margins. "
        "Vertical integration also stabilizes raw material procurement during port congestion."
    )
    _write_pdf(in_dir / "copy_a.pdf", f"{shared_prompt} {copied_body}")
    _write_docx(in_dir / "copy_b.docx", [shared_prompt, copied_body])

    # 2. Honest original docs sharing the exact same prompt (Scenario 4)
    _write_pdf(
        in_dir / "honest_solar.pdf",
        f"{shared_prompt} Deploying a solar microgrid reduces rural factory downtime by forty percent each summer. "
        "Battery storage further lowers peak diesel generator expenses.",
    )
    _write_docx(
        in_dir / "honest_fintech.docx",
        [
            shared_prompt,
            "Alternative fintech credit scoring helps small kirana merchants access affordable working capital. "
            "Repayment reminders over vernacular voice calls keep default rates under two percent.",
        ],
    )

    # 3. Corrupted & zero-byte files (Scenario 3) + ignored manifest JSON
    (in_dir / "empty_zero.pdf").write_bytes(b"")
    (in_dir / "corrupt_doc.pdf").write_bytes(b"corrupted non-pdf bytes")
    (in_dir / "dataset_manifest.json").write_text('{"ignored": true}', encoding="utf-8")

    # 4. External reference KB
    (kb_dir / "urban_ref.txt").write_text(
        "Urban mobility corridors with dedicated electric bus lanes reduce commuter travel times significantly.",
        encoding="utf-8",
    )

    settings = Settings(external_kb_dir=kb_dir, min_chunk_words=8)
    pipeline = IntegrityPipeline(
        settings,
        embedder=TextEmbedder(settings, model=_DeterministicTopicEmbedder()),
    )
    reports = pipeline.run(in_dir, out_dir)
    by_id = {r.document_id: r for r in reports}

    # Manifest JSON must be ignored; 6 supported files processed
    assert len(reports) == 6
    assert "dataset_manifest.json" not in by_id

    # Corrupted & empty files skipped safely without crashing
    assert by_id["empty_zero.pdf"].status == "skipped_empty"
    assert by_id["empty_zero.pdf"].total_integrity_risk == "SKIPPED"
    assert by_id["corrupt_doc.pdf"].status == "skipped_corrupted"

    # Shared prompt discounted across all 4 valid docs
    for valid_id in ("copy_a.pdf", "copy_b.docx", "honest_solar.pdf", "honest_fintech.docx"):
        assert by_id[valid_id].boilerplate_sentences_ignored == 2

    # Honest docs remain LOW peer similarity despite sharing the prompt!
    assert by_id["honest_solar.pdf"].peer_similarity_percentage == 0.0
    assert by_id["honest_fintech.docx"].peer_similarity_percentage == 0.0

    # Copied pair flagged HIGH with exact peer match
    assert by_id["copy_a.pdf"].peer_similarity_percentage >= 95.0
    assert by_id["copy_a.pdf"].total_integrity_risk == "HIGH"
    assert by_id["copy_a.pdf"].peer_matches[0].matched_file == "copy_b.docx"

    # Verify JSON schema compliance on disk
    disk_json = json.loads((out_dir / "copy_a.pdf.json").read_text(encoding="utf-8"))
    for key in ("document_id", "total_integrity_risk", "scores", "peer_matches", "multimodal_matches", "explanations"):
        assert key in disk_json


def test_pdf_chunks_never_cross_page_boundaries(tmp_path: Path):
    from src.models import ExtractedDocument, SentenceItem

    settings = Settings(chunk_size_words=120, chunk_overlap_words=25, min_chunk_words=5)
    pipeline = IntegrityPipeline(
        settings,
        embedder=TextEmbedder(settings, model=_DeterministicTopicEmbedder()),
    )
    doc = ExtractedDocument(
        document_id="multi.pdf",
        file_path=str(tmp_path / "multi.pdf"),
        modality="pdf",
        status="ok",
        sentences=[
            SentenceItem("Short page one executive summary text here.", page_number=1),
            SentenceItem("Exact page two copied operational margin paragraph here.", page_number=2),
            SentenceItem("Short page three financial payback summary text here.", page_number=3),
        ],
    )
    chunks = pipeline._build_chunks_for_document(doc)
    assert len(chunks) == 3
    assert [c.page_number for c in chunks] == [1, 2, 3]
    assert "page two" in chunks[1].text.lower()
    assert "page one" not in chunks[1].text.lower()
