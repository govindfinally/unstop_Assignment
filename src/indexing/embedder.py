"""Dense vector embedding wrapper using Sentence-Transformers (all-MiniLM-L6-v2)."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from src.config import Settings
from src.logger import get_logger

logger = get_logger("indexing.embedder")

_DEFAULT_EMBEDDING_DIM = 384


class TextEmbedder:
    """Encodes text chunks into L2-normalized float32 embeddings.

    DSA: Contiguous 2D dense float32 matrix (np.ndarray).
    TC: O(N * L_tokens) across mini-batches.
    SC: O(M + N * D) where M is model weights (~90MB) and D=384.
    """

    def __init__(
        self,
        settings: Settings,
        model: Optional[Any] = None,
        batch_size: int = 32,
    ) -> None:
        self.settings = settings
        self._model = model
        self.batch_size = batch_size
        self._dim = _DEFAULT_EMBEDDING_DIM

    def _get_model(self) -> Any:
        """Lazy-load SentenceTransformer once and cache on the instance."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model '%s'", self.settings.embedding_model_name)
            self._model = SentenceTransformer(self.settings.embedding_model_name)
            dim = self._model.get_sentence_embedding_dimension()
            if dim is not None:
                self._dim = int(dim)
        return self._model

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into L2-normalized float32 embeddings of shape (N, D)."""
        if not texts:
            logger.debug("Empty text list passed to TextEmbedder.encode(); returning (0, %d)", self._dim)
            return np.empty((0, self._dim), dtype=np.float32)

        model = self._get_model()
        logger.info("Encoding %d text chunk(s) with batch_size=%d", len(texts), self.batch_size)
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        arr = np.ascontiguousarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self._dim = arr.shape[1]
        return arr
