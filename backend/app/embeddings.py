"""Lazy, reusable sentence embedding provider."""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: str | list[str]) -> np.ndarray:
        values = [texts] if isinstance(texts, str) else texts
        embeddings = self.model.encode(values, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(embeddings, dtype=np.float32)


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(get_settings().model_name)
