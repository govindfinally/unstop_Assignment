# Evaluation Report — Multimodal Submission Integrity & Plagiarism Detection Engine

This document evaluates the prototype against our 15-file synthetic ground-truth benchmark ([dataset_manifest.json](sample_submissions/dataset_manifest.json)) using real `faster-whisper` (`base.en`, `int8` CPU), `sentence-transformers/all-MiniLM-L6-v2`, and `FAISS` (`IndexFlatIP`).

---

## 1. Benchmark Dataset Summary

- **Total Submissions:** 15 (`6 PDF`, `6 DOCX`, `3 MP4`) + 3 local reference articles in `external_knowledge/`.
- **Data Compliance:** 100% synthetic original business case studies and open-style reference summaries (zero proprietary or private student data).
- **Ground-Truth Split:**
  - **7 Expected Flagged (Positive Class):** Exact copy pair (`sub_01`, `sub_02`), cross-modal MP4 reading `sub_01` p.2 (`sub_03`), synonym-paraphrased DOCX pair (`sub_04`, `sub_05`), external knowledge copy (`sub_06`), and LLM-styled prose (`sub_07`).
  - **6 Expected Clean (Negative Class):** 5 honest submissions sharing an identical 150-word competition prompt (`sub_08`–`sub_12`) + 1 original spoken MP4 pitch (`sub_13`).
  - **2 Degraded / Edge-Case Files:** 1 zero-byte PDF (`sub_14`) + 1 silent MP4 video (`sub_15`) — evaluated on graceful skip behavior without crashing.

---

## 2. Quantitative Metrics & Confusion Matrices

All 13 valid submissions were evaluated across two operating thresholds (`HIGH` risk only vs. `HIGH + MEDIUM` combined), plus 2/2 degraded files verified for graceful handling.

### Operating Point A — Actionable Flag (`HIGH` Risk Only)

| | Predicted Flagged (`HIGH`) | Predicted Clean (`MEDIUM` / `LOW`) |
| :--- | :---: | :---: |
| **Actual Positive (Plagiarized / AI)** | **7 (TP)** | **0 (FN)** |
| **Actual Negative (Honest / Clean)** | **0 (FP)** | **6 (TN)** |

- **Precision:** `7 / (7 + 0) = 100.0%`
- **Recall:** `7 / (7 + 0) = 100.0%`
- **F1-Score:** `100.0%`
- **Degraded File Survival Rate:** `2 / 2 (100.0%)` — `sub_14_corrupt_empty.pdf` (`skipped_empty`) and `sub_15_silent_video.mp4` (`skipped_no_audio`) logged clear warnings with zero crashes.

### Operating Point B — Strict Review Queue (`HIGH` or `MEDIUM` Risk)

| | Predicted Flagged (`HIGH` / `MEDIUM`) | Predicted Clean (`LOW`) |
| :--- | :---: | :---: |
| **Actual Positive (Plagiarized / AI)** | **7 (TP)** | **0 (FN)** |
| **Actual Negative (Honest / Clean)** | **3 (FP)** | **3 (TN)** |

- **Precision:** `7 / (7 + 3) = 70.0%`
- **Recall:** `7 / (7 + 0) = 100.0%`
- **F1-Score:** `82.4%`
- *Why the 3 FPs occurred:* `sub_10`, `sub_11`, and `sub_12` scored **0.0% peer similarity** and **0.0% external similarity**, but triggered `MEDIUM` risk (`56.8%–62.5%`) solely from the statistical AI heuristic on short (~75-word) formal prose. Analyzed in detail in Section 4 below.

---

## 3. Per-Document & Scenario Verification Results

| File | Scenario / Ground Truth | Status | Risk | Peer % | External % | AI % | Boilerplate Ignored | Top Matched Source |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `sub_01_vertical_orig.pdf` | Exact + Cross-Modal Source | `ok` | **HIGH** | 96.6% | 0.0% | 50.0% | 0 | `sub_02_vertical_copy.docx` (0.97) |
| `sub_02_vertical_copy.docx` | Exact Copy + Enter-Spam | `ok` | **HIGH** | 96.6% | 0.0% | 61.5% | 0 | `sub_01_vertical_orig.pdf` (0.97) |
| `sub_03_pitch_copied.mp4` | **Scenario 1: Cross-Modal** | `ok` | **HIGH** | 98.9% | 0.0% | 65.0% | 0 | `sub_01_vertical_orig.pdf` (0.99, p.2) |
| `sub_04_coldchain_a.docx` | **Scenario 2: Paraphrase** | `ok` | **HIGH** | 85.2% | 0.0% | 49.0% | 0 | `sub_05_coldchain_paraphrase.docx` (0.85) |
| `sub_05_coldchain_paraphrase.docx` | **Scenario 2: Paraphrase** | `ok` | **HIGH** | 85.2% | 0.0% | 46.9% | 0 | `sub_04_coldchain_a.docx` (0.85) |
| `sub_06_external_wiki.pdf` | External KB Plagiarism | `ok` | **HIGH** | 0.0% | 92.5% | 39.1% | 0 | `ref_maritime_supply_chains.txt` (0.93) |
| `sub_07_ai_generated.docx` | AI-Generated Prose | `ok` | **HIGH** | 0.0% | 0.0% | 100.0% | 0 | None |
| `sub_08_honest_prompt_1.pdf` | **Scenario 4: Shared Prompt** | `ok` | **LOW** | 0.0% | 0.0% | 13.5% | 6 | None |
| `sub_09_honest_prompt_2.pdf` | **Scenario 4: Shared Prompt** | `ok` | **LOW** | 0.0% | 0.0% | 45.7% | 6 | None |
| `sub_10_honest_prompt_3.pdf` | **Scenario 4: Shared Prompt** | `ok` | **MEDIUM** | 0.0% | 0.0% | 58.8% | 6 | None |
| `sub_11_honest_prompt_4.docx` | **Scenario 4: Shared Prompt** | `ok` | **MEDIUM** | 0.0% | 0.0% | 62.5% | 6 | None |
| `sub_12_honest_prompt_5.docx` | **Scenario 4: Shared Prompt** | `ok` | **MEDIUM** | 0.0% | 0.0% | 56.8% | 6 | None |
| `sub_13_pitch_original.mp4` | Clean Original MP4 | `ok` | **LOW** | 0.0% | 0.0% | 30.0% | 0 | None |
| `sub_14_corrupt_empty.pdf` | **Scenario 3: 0-Byte PDF** | `skipped_empty` | **SKIPPED** | 0.0% | 0.0% | 0.0% | 0 | None |
| `sub_15_silent_video.mp4` | **Scenario 3: Silent MP4** | `skipped_no_audio` | **SKIPPED** | 0.0% | 0.0% | 0.0% | 0 | None |

- **Scenario 1 (Cross-Modal):** `faster-whisper base.en` accurately transcribed `sub_03_pitch_copied.mp4`, stripped spoken disfluencies (`um`, `uh`, `ah`), and matched `sub_01_vertical_orig.pdf` **Page 2** at **`0.99` (`98.9%` weighted similarity)** with exact timestamp `00:00:00 - 00:00:36`.
- **Scenario 2 (Paraphrased Content):** Heavy synonym rewriting between `sub_04` and `sub_05` scored `0.85` cosine similarity (`85.2%` weighted peer similarity).
- **Scenario 4 (Boilerplate):** All 6 sentences of the shared 150-word competition prompt were stripped across all 5 honest submissions (`30` total occurrences discounted), yielding **0.0% peer similarity** on every honest submission.
- **Automated Test Suite:** `40 / 40` `pytest` unit & integration tests passing in `< 3s` with `0` warnings.

---

## 4. Failure & Ambiguity Analysis

### Failure Case 1: AI Heuristic Over-Sensitivity on Short Formal Business Prose
- **Observed Behavior:** Three genuinely human-written Scenario-4 answers (`sub_10_honest_prompt_3.pdf`, `sub_11_honest_prompt_4.docx`, `sub_12_honest_prompt_5.docx`) scored `56.8%–62.5%` AI likelihood (`MEDIUM` risk) even though they contained `0.00` LLM discourse markers and `0.0%` plagiarism.
- **Root Cause:** After stripping the 6 shared prompt sentences, each remaining answer was only ~70–80 words (3–4 formal sentences of similar length, `CV = 0.18–0.24`, and zero casual contractions). With only 3–4 sentences, sentence-length variance (burstiness) has high estimation noise, and formal business English naturally lacks contractions.
- **Resolution:**
  1. Raise the statistical AI detector's minimum sample-size guard from `30 words` to `>= 100 words` and `>= 6 sentences` before emitting a `MEDIUM`/`HIGH` AI flag.
  2. Use `--use_hf_ai_detector` (blending `roberta-base-openai-detector` or a perplexity-ratio detector like Binoculars) for production escalation, and never auto-penalize on AI heuristics alone without human review.

### Failure Case 2: Cross-Page Sliding-Window Page Attribution Offset (Caught & Fixed)
- **Observed Behavior (Initial Run):** In our initial end-to-end run, `sub_03_pitch_copied.mp4` correctly matched the paragraph from **Page 2** of `sub_01_vertical_orig.pdf` (`score = 0.85`), but the report cited **`Match p.1`** instead of **`Match p.2`**.
- **Root Cause:** Because Page 1 of `sub_01_vertical_orig.pdf` was short (~45 words), the global 120-word sliding window started on Page 1 and spanned into Page 2, tagging the resulting chunk with the window's starting page (`p.1`).
- **Resolution (Implemented):** Updated `IntegrityPipeline._build_chunks_for_document()` to run `sentence_aware_sliding_window_with_meta()` **per page** for PDFs (verified by `test_pdf_chunks_never_cross_page_boundaries`), ensuring PDF page citations are strictly pinpoint-accurate. For passages that genuinely span a page break in multi-page prose, a production enhancement would add a short cross-page overlap chunk explicitly tagged with `page_range: "1-2"`.

### Failure Case 3: Small-Batch Boilerplate vs. Multi-Student Collusion Ambiguity
- **Observed Behavior (Design Limitation):** Our sentence-level boilerplate filter flags verbatim sentences appearing in `>= max(3, ceil(0.25 * N))` valid documents. While this cleanly isolates shared competition prompts (Scenario 4) and protects 2-student cheating pairs, in a very small batch (e.g., `N = 6`), a 3-student verbatim collusion ring could cross the threshold and be misclassified as boilerplate.
- **Resolution:** In production, accept an optional `--template_file` containing the official competition prompt/rubric for deterministic exclusion, and restrict automatic corpus-frequency boilerplate detection to batches with `N >= 10` (or require appearance in `>= 30%` of submissions).

---

## 5. Architectural Trade-Off Analysis

| Component | Chosen Option | Alternative Considered | Why We Chose This Trade-Off |
| :--- | :--- | :--- | :--- |
| **ASR Model** | `faster-whisper` (`base.en`, `int8` CPU) | `whisper-tiny.en` / `whisper-medium` | Prioritized **accuracy over speed**: `tiny.en` drops domain terms & percentages (`4.2%`), risking false negatives or unfair accusations; `base.en` (~145 MB) runs smoothly on CPU while transcribing numbers and business jargon accurately. |
| **Embedding Model** | `all-MiniLM-L6-v2` (384-dim, ~90 MB) | `bge-large-en-v1.5` (1024-dim, ~1.3 GB) | 10x smaller memory footprint and fast CPU batch encoding while still achieving `0.85` cosine similarity on heavy synonym paraphrases. |
| **Vector Index** | `FAISS IndexFlatIP` (L2-normalized = exact Cosine) | `IndexHNSWFlat` / `ChromaDB` | Zero approximation error and negligible latency (`< 2 ms`) for 15–500 documents, with a 1-line upgrade path to `IndexHNSWFlat` at 10,000+ documents. |
| **Boilerplate Filter** | Sentence-level normalized `Counter` **before** chunking | Chunk-level semantic similarity filter | Strictly `O(N)` linear, prevents chunk boundary shifts, keeps prompts out of AI detection, and avoids accidentally excusing paraphrased cheating rings. |
| **AI Detection** | Hybrid (Statistical Heuristics default + optional `--use_hf_ai_detector`) | Paid API / Always-on heavy transformer | Instant `< 5 ms` offline execution by default with transparent metrics, plus real RoBERTa neural classification on demand with automatic offline fallback. |
