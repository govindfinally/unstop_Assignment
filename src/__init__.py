"""Multimodal Submission Integrity & Plagiarism Detection Engine."""

from __future__ import annotations

import os

# Prevent OpenMP/MKL DLL load-order clash on Windows when both faiss and pymupdf are used.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import faiss as _faiss  # noqa: F401
except Exception:
    pass
