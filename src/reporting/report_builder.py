"""Explainable JSON/Markdown audit reports (<30s reviewer cards) and batch summary matrix."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.logger import get_logger
from src.models import DocumentReport, MatchEvidence

logger = get_logger("reporting.report_builder")

_RISK_ORDER: Dict[str, int] = {
    "HIGH": 0,
    "MEDIUM": 1,
    "LOW": 2,
    "SKIPPED": 3,
    "ERROR": 4,
}


def _escape_md_cell(text: str, max_len: int = 240) -> str:
    """Clean newlines/pipes and truncate long passages for crisp Markdown tables."""
    flat = " ".join(text.split())
    if len(flat) > max_len:
        flat = flat[: max_len - 3].rstrip() + "..."
    return flat.replace("|", "\\|")


def _format_location(loc: Dict[str, object]) -> str:
    """Render page/timestamp dict cleanly for human reviewers."""
    if not loc:
        return "N/A"
    parts: List[str] = []
    if "query_page" in loc:
        parts.append(f"Query p.{loc['query_page']}")
    if "query_timestamp" in loc:
        parts.append(f"Query {loc['query_timestamp']}")
    if "matched_page" in loc:
        parts.append(f"Match p.{loc['matched_page']}")
    if "matched_timestamp" in loc:
        parts.append(f"Match {loc['matched_timestamp']}")
    if "matched_source" in loc and "matched_page" not in loc and "matched_timestamp" not in loc:
        parts.append(f"Source: {loc['matched_source']}")
    return " | ".join(parts) if parts else "N/A"


def _render_evidence_section(title: str, matches: List[MatchEvidence]) -> List[str]:
    if not matches:
        return [f"## {title}", "", "_No significant overlaps detected._", ""]

    lines = [
        f"## {title} ({len(matches)} match(es))",
        "",
        "| Matched File | Score | Location | Query Passage | Matched Source Passage |",
        "| :--- | :---: | :--- | :--- | :--- |",
    ]
    for m in matches:
        lines.append(
            f"| `{m.matched_file}` | **{m.similarity_score:.2f}** | "
            f"{_format_location(m.location)} | "
            f"\"{_escape_md_cell(m.query_passage)}\" | "
            f"\"{_escape_md_cell(m.matched_passage)}\" |"
        )
    lines.append("")
    return lines


class ReportBuilder:
    """Writes schema-compliant JSON, <30s Markdown audit cards, and batch summary matrix.

    DSA: Risk-priority Hash Map + Timsort + linear serialization.
    TC: O(D log D + M) where D = documents, M = total matches.
    SC: O(M) for serialized report buffers.
    """

    def render_markdown(self, report: DocumentReport) -> str:
        """Render a human-verifiable (<30s) Markdown audit report for one document."""
        lines: List[str] = [
            f"# Integrity Audit Report: `{report.document_id}`",
            "",
            f"- **Overall Risk:** **{report.total_integrity_risk}**",
            f"- **Extraction Status:** `{report.status}`",
            f"- **Shared Boilerplate Sentences Ignored:** `{report.boilerplate_sentences_ignored}`",
            "",
            "## Integrity Score Breakdown",
            "",
            "| Signal | Score |",
            "| :--- | :---: |",
            f"| Internal Peer Similarity (incl. Cross-Modal) | **{report.peer_similarity_percentage:.2f}%** |",
            f"| External / Reference Similarity | **{report.external_similarity_percentage:.2f}%** |",
            f"| AI-Generated Content Likelihood | **{report.ai_generated_probability:.2f}%** |",
            "",
            "## Reviewer Summary",
            "",
            report.explanations or "No integrity flags triggered.",
            "",
        ]

        if report.warnings:
            lines.append("## Warnings")
            lines.append("")
            for w in report.warnings:
                lines.append(f"- {w}")
            lines.append("")

        lines.extend(_render_evidence_section("Peer-to-Peer Matches", report.peer_matches))
        lines.extend(_render_evidence_section("Cross-Modal Matches (Audio/Video <-> Text)", report.multimodal_matches))
        lines.extend(_render_evidence_section("External Reference Matches", report.external_matches))
        return "\n".join(lines).strip() + "\n"

    def _sort_reports(self, reports: List[DocumentReport]) -> List[DocumentReport]:
        return sorted(
            reports,
            key=lambda r: (
                _RISK_ORDER.get(r.total_integrity_risk, 99),
                -max(
                    r.peer_similarity_percentage,
                    r.external_similarity_percentage,
                    r.ai_generated_probability,
                ),
                r.document_id,
            ),
        )

    def render_summary_matrix_md(self, reports: List[DocumentReport]) -> str:
        """Render aggregate batch summary matrix sorted by risk descending (Scenario 5)."""
        ordered = self._sort_reports(reports)
        lines: List[str] = [
            "# Batch Submission Integrity Summary Matrix",
            "",
            f"**Total Submissions Evaluated:** {len(ordered)}",
            "",
            "| Document | Status | Risk | Peer % | External % | AI % | Boilerplate Ignored | Top Matched Source |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
        ]

        for r in ordered:
            all_matches = sorted(
                r.peer_matches + r.multimodal_matches + r.external_matches,
                key=lambda m: m.similarity_score,
                reverse=True,
            )
            top_match = (
                f"`{all_matches[0].matched_file}` ({all_matches[0].similarity_score:.2f})"
                if all_matches
                else "None"
            )
            lines.append(
                f"| `{r.document_id}` | `{r.status}` | **{r.total_integrity_risk}** | "
                f"{r.peer_similarity_percentage:.1f}% | {r.external_similarity_percentage:.1f}% | "
                f"{r.ai_generated_probability:.1f}% | {r.boilerplate_sentences_ignored} | {top_match} |"
            )

        lines.append("")
        return "\n".join(lines)

    def write_all(self, reports: List[DocumentReport], output_dir: Path) -> None:
        """Write per-document .json/.md reports plus summary_matrix.json/.md."""
        output_dir.mkdir(parents=True, exist_ok=True)
        ordered = self._sort_reports(reports)

        for report in ordered:
            # Keep full document_id (incl. extension) in report filename to avoid collisions
            # when e.g. 'case.pdf' and 'case.docx' coexist in the same folder.
            json_path = output_dir / f"{report.document_id}.json"
            md_path = output_dir / f"{report.document_id}.md"

            json_path.write_text(
                json.dumps(report.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            md_path.write_text(self.render_markdown(report), encoding="utf-8")

        matrix_md_path = output_dir / "summary_matrix.md"
        matrix_json_path = output_dir / "summary_matrix.json"

        matrix_md_path.write_text(self.render_summary_matrix_md(ordered), encoding="utf-8")
        matrix_json_path.write_text(
            json.dumps(
                {
                    "total_submissions": len(ordered),
                    "summary": [r.to_dict() for r in ordered],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        logger.info(
            "Wrote %d per-document JSON/MD reports + summary_matrix to %s",
            len(ordered),
            output_dir,
        )
