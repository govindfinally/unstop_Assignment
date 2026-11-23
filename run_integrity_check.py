"""CLI entrypoint for the Multimodal Submission Integrity & Plagiarism Detection Engine.

Usage:
    python run_integrity_check.py --input_dir ./sample_submissions --output_dir ./reports
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import Settings
from src.logger import get_logger, setup_logging
from src.pipeline import IntegrityPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal Submission Integrity & Plagiarism Detection Engine"
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing candidate submissions (.pdf, .docx, .mp4).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory where JSON/Markdown audit reports will be saved.",
    )
    parser.add_argument(
        "--external_kb_dir",
        type=Path,
        default=None,
        help="Optional directory of external reference texts.",
    )
    parser.add_argument(
        "--use_hf_ai_detector",
        action="store_true",
        help="Enable optional Hugging Face RoBERTa AI detector (blended with heuristics).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings(
        log_level=args.log_level,
        use_hf_ai_classifier=args.use_hf_ai_detector,
    )
    if args.external_kb_dir is not None:
        settings.external_kb_dir = args.external_kb_dir

    setup_logging(settings.log_dir, settings.log_file_name, settings.log_level)
    logger = get_logger("cli")
    logger.info("Starting Multimodal Integrity Check")

    pipeline = IntegrityPipeline(settings)
    pipeline.run(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
