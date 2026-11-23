"""Unit tests for unified PDF & DOCX extractor (src/extractors/document_extractor.py)."""

from pathlib import Path

import docx
from fpdf import FPDF

from src.config import Settings
from src.extractors.document_extractor import DocumentExtractor


def _create_sample_pdf(path: Path, pages_text: list[str]) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    for text in pages_text:
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 8, text)
    pdf.output(str(path))


def _create_sample_docx(path: Path, paragraphs: list[str], table_rows: list[list[str]] | None = None) -> None:
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    if table_rows:
        table = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
        for r_idx, row in enumerate(table_rows):
            for c_idx, val in enumerate(row):
                table.cell(r_idx, c_idx).text = val
    doc.save(str(path))


def test_zero_byte_pdf_and_docx_are_skipped_gracefully(tmp_path: Path):
    extractor = DocumentExtractor(Settings())
    empty_pdf = tmp_path / "empty.pdf"
    empty_docx = tmp_path / "empty.docx"
    empty_pdf.write_bytes(b"")
    empty_docx.write_bytes(b"")

    res_pdf = extractor.extract(empty_pdf)
    res_docx = extractor.extract(empty_docx)

    assert res_pdf.status == "skipped_empty"
    assert len(res_pdf.warnings) == 1
    assert res_docx.status == "skipped_empty"
    assert len(res_docx.warnings) == 1


def test_corrupted_pdf_and_docx_do_not_crash(tmp_path: Path):
    extractor = DocumentExtractor(Settings())
    bad_pdf = tmp_path / "corrupt.pdf"
    bad_docx = tmp_path / "corrupt.docx"
    bad_pdf.write_bytes(b"not a real pdf binary header 12345")
    bad_docx.write_bytes(b"not a real zip/docx archive 67890")

    res_pdf = extractor.extract(bad_pdf)
    res_docx = extractor.extract(bad_docx)

    assert res_pdf.status == "skipped_corrupted"
    assert res_pdf.sentences == []
    assert res_docx.status == "skipped_corrupted"
    assert res_docx.sentences == []


def test_blank_pdf_returns_skipped_empty(tmp_path: Path):
    extractor = DocumentExtractor(Settings())
    blank_pdf = tmp_path / "blank.pdf"
    _create_sample_pdf(blank_pdf, ["   "])

    res = extractor.extract(blank_pdf)
    assert res.status == "skipped_empty"
    assert res.sentences == []


def test_multipage_pdf_strips_headers_and_preserves_page_numbers(tmp_path: Path):
    extractor = DocumentExtractor(Settings())
    pdf_path = tmp_path / "case_report.pdf"
    _create_sample_pdf(
        pdf_path,
        [
            "Unstop Challenge 2026\nVertical integration reduced unit costs by 14 percent.\nPage 1 of 3",
            "Unstop Challenge 2026\nOperating margins contracted by 4.2 percent in Q3.\nPage 2 of 3",
            "Unstop Challenge 2026\nRegional warehouses lowered lead time to two days.\nPage 3 of 3",
        ],
    )

    res = extractor.extract(pdf_path)
    assert res.status == "ok"
    assert res.modality == "pdf"
    assert len(res.sentences) == 3
    assert [s.page_number for s in res.sentences] == [1, 2, 3]
    joined = " ".join(s.text for s in res.sentences)
    assert "Unstop Challenge 2026" not in joined
    assert "Page 1 of 3" not in joined


def test_docx_extracts_paragraphs_tables_and_defeats_enter_spam(tmp_path: Path):
    extractor = DocumentExtractor(Settings())
    docx_path = tmp_path / "summary.docx"
    _create_sample_docx(
        docx_path,
        paragraphs=["Operating", "margins", "dropped", "4.2 percent.", "Revenue grew strongly in Q4."],
        table_rows=[["Metric", "Value"], ["EBITDA Margin", "18.5 percent."]],
    )

    res = extractor.extract(docx_path)
    assert res.status == "ok"
    assert res.modality == "docx"
    assert len(res.sentences) >= 2
    assert res.sentences[0].text == "Operating margins dropped 4.2 percent."
    assert res.sentences[0].page_number == 1
    assert any("EBITDA Margin" in s.text for s in res.sentences)
