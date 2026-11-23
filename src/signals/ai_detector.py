"""Hybrid AI-generated text likelihood estimator (Statistical Heuristics + Optional HF RoBERTa)."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Callable, List, Optional

from src.config import Settings
from src.extractors.normalizer import split_sentences
from src.logger import get_logger

logger = get_logger("signals.ai_detector")

_WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")
_CONTRACTION_RE = re.compile(
    r"\b(?:don't|can't|won't|isn't|aren't|wasn't|weren't|haven't|hasn't|didn't|"
    r"we're|they're|it's|that's|there's|i'm|you're|wouldn't|couldn't|shouldn't)\b",
    re.IGNORECASE,
)

# High-frequency LLM discourse markers / transitions
_LLM_MARKERS_RE = re.compile(
    r"\b(?:"
    r"furthermore|moreover|additionally|consequently|nevertheless|nonetheless|"
    r"in conclusion|to summarize|in summary|ultimately|paramount|pivotal|"
    r"delve into|holistic approach|multifaceted|underscores|landscape|"
    r"it is crucial to note|it is important to note|plays a crucial role|"
    r"foster innovation|seamless integration|leverage synergies|robust framework|"
    r"navigating the complexities|comprehensive strategy|notably|thereby"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class AIDetectionResult:
    """Structured AI-generation estimation result with transparent signal breakdown."""

    ai_generated_probability: float
    method: str  # "heuristic", "hybrid_hf_roberta", "insufficient_text"
    burstiness_cv: float = 0.0
    marker_density_per_100w: float = 0.0
    contraction_rate_per_100w: float = 0.0


class AIDetector:
    """Estimates AI-generation probability on post-boilerplate text.

    Default mode uses fast, explainable statistical heuristics (burstiness CV,
    LLM discourse marker density, contraction rate, lexical uniformity).
    When settings.use_hf_ai_classifier=True, blends a Hugging Face RoBERTa
    detector (0.7 HF + 0.3 heuristic) with graceful fallback on error/offline.

    DSA: Precompiled Regex + Hash Set (unique words) + single-pass mean/std.
    TC: O(W) linear in word count.
    SC: O(V) where V is unique vocabulary size.
    """

    def __init__(
        self,
        settings: Settings,
        hf_pipeline_factory: Optional[Callable[[], Any]] = None,
    ) -> None:
        self.settings = settings
        self._hf_pipeline_factory = hf_pipeline_factory
        self._hf_pipeline: Optional[Any] = None
        self._hf_failed = False

    def _get_hf_pipeline(self) -> Optional[Any]:
        if self._hf_failed:
            return None
        if self._hf_pipeline is not None:
            return self._hf_pipeline

        try:
            if self._hf_pipeline_factory is not None:
                self._hf_pipeline = self._hf_pipeline_factory()
            else:
                from transformers import pipeline

                logger.info("Loading HF AI-detector classifier '%s'", self.settings.ai_classifier_model)
                self._hf_pipeline = pipeline(
                    "text-classification",
                    model=self.settings.ai_classifier_model,
                    truncation=True,
                    max_length=512,
                )
            return self._hf_pipeline
        except Exception as exc:
            self._hf_failed = True
            logger.warning(
                "HF AI classifier unavailable (%s: %s); falling back to statistical heuristics",
                type(exc).__name__,
                exc,
            )
            return None

    def _score_heuristic(self, text: str) -> AIDetectionResult:
        words = _WORD_RE.findall(text)
        word_count = len(words)
        if word_count == 0:
            return AIDetectionResult(ai_generated_probability=0.0, method="insufficient_text")

        if word_count < 30:
            logger.debug("Text too short (%d words < 30) for reliable AI detection", word_count)
            return AIDetectionResult(ai_generated_probability=15.0, method="insufficient_text")

        sentences = split_sentences(text)
        sent_lengths = [len(_WORD_RE.findall(s)) for s in sentences if _WORD_RE.findall(s)]

        # 1. Burstiness (Coefficient of Variation of sentence lengths)
        # Humans: high variance (CV >= 0.45); LLMs: smooth/uniform (CV <= 0.25)
        if len(sent_lengths) >= 3:
            mean_len = sum(sent_lengths) / len(sent_lengths)
            variance = sum((x - mean_len) ** 2 for x in sent_lengths) / len(sent_lengths)
            cv = math.sqrt(variance) / mean_len if mean_len > 0 else 0.0
        else:
            cv = 0.35

        # Map CV: 0.15 -> 1.0 (very AI-like), 0.55 -> 0.0 (very human-like)
        uniformity_score = max(0.0, min(1.0, (0.55 - cv) / 0.40))

        # 2. LLM discourse marker density per 100 words
        marker_hits = len(_LLM_MARKERS_RE.findall(text))
        marker_density = (marker_hits / word_count) * 100.0
        marker_score = max(0.0, min(1.0, marker_density / 2.2))

        # 3. Contraction rate per 100 words (casual human writing uses contractions)
        contraction_hits = len(_CONTRACTION_RE.findall(text))
        contraction_rate = (contraction_hits / word_count) * 100.0
        formality_score = max(0.0, min(1.0, 1.0 - (contraction_rate / 1.5)))

        # 4. Mean sentence length polish (LLMs tend to hover around 18-28 words/sentence)
        avg_sent_len = (sum(sent_lengths) / len(sent_lengths)) if sent_lengths else 15.0
        polish_score = 1.0 if 17.0 <= avg_sent_len <= 28.0 else max(0.0, 1.0 - abs(avg_sent_len - 22.0) / 18.0)

        # Weighted blend -> 0..100%
        raw_prob = (
            0.35 * uniformity_score
            + 0.35 * marker_score
            + 0.15 * formality_score
            + 0.15 * polish_score
        ) * 100.0
        prob = round(max(0.0, min(100.0, raw_prob)), 2)

        return AIDetectionResult(
            ai_generated_probability=prob,
            method="heuristic",
            burstiness_cv=round(cv, 3),
            marker_density_per_100w=round(marker_density, 3),
            contraction_rate_per_100w=round(contraction_rate, 3),
        )

    def _score_hf(self, text: str) -> Optional[float]:
        clf = self._get_hf_pipeline()
        if clf is None:
            return None
        try:
            outputs = clf(text[:2000])
            if not outputs:
                return None
            top = outputs[0]
            label = str(top.get("label", "")).lower()
            score = float(top.get("score", 0.5))
            # RoBERTa OpenAI detector uses "Fake" (AI) vs "Real" (Human)
            if "fake" in label or "ai" in label or "chatgpt" in label or label == "label_1":
                return score * 100.0
            return (1.0 - score) * 100.0
        except Exception as exc:
            logger.warning("HF AI classifier inference failed (%s: %s); using heuristic", type(exc).__name__, exc)
            return None

    def estimate(self, document_id: str, text: str) -> AIDetectionResult:
        """Estimate AI-generated probability (0-100%) on clean post-boilerplate text."""
        heuristic_res = self._score_heuristic(text)
        if heuristic_res.method == "insufficient_text":
            logger.info("[%s] AI probability = %.2f%% (method=%s)", document_id, heuristic_res.ai_generated_probability, heuristic_res.method)
            return heuristic_res

        if self.settings.use_hf_ai_classifier:
            hf_score = self._score_hf(text)
            if hf_score is not None:
                blended = round(0.70 * hf_score + 0.30 * heuristic_res.ai_generated_probability, 2)
                logger.info(
                    "[%s] AI probability = %.2f%% (hybrid: HF=%.2f%%, heuristic=%.2f%%)",
                    document_id,
                    blended,
                    hf_score,
                    heuristic_res.ai_generated_probability,
                )
                return AIDetectionResult(
                    ai_generated_probability=blended,
                    method="hybrid_hf_roberta",
                    burstiness_cv=heuristic_res.burstiness_cv,
                    marker_density_per_100w=heuristic_res.marker_density_per_100w,
                    contraction_rate_per_100w=heuristic_res.contraction_rate_per_100w,
                )

        logger.info(
            "[%s] AI probability = %.2f%% (heuristic: cv=%.2f, markers=%.2f/100w)",
            document_id,
            heuristic_res.ai_generated_probability,
            heuristic_res.burstiness_cv,
            heuristic_res.marker_density_per_100w,
        )
        return heuristic_res
