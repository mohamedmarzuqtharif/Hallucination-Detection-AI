"""FAISS-backed semantic retrieval and index management."""
import json
from pathlib import Path
from threading import RLock
from typing import Any

import faiss
import numpy as np

from app.config import Settings, get_settings
from app.embeddings import EmbeddingService, get_embedding_service


class RetrievalService:
    def __init__(self, settings: Settings, embedder: EmbeddingService) -> None:
        self.settings, self.embedder = settings, embedder
        self.index_path = settings.data_dir / "faiss" / "knowledge.index"
        self.docs_path = settings.data_dir / "faiss" / "documents.json"
        self._lock = RLock()
        self.index: Any = None
        self.documents: list[dict[str, Any]] = []

    def load(self) -> None:
        with self._lock:
            if not self.index_path.exists() or not self.docs_path.exists():
                source = self.settings.data_dir / "documents" / "knowledge_base.json"
                if not source.exists():
                    raise FileNotFoundError("No knowledge base or FAISS index found")
                self.build(json.loads(source.read_text(encoding="utf-8")))
            self.index = faiss.read_index(str(self.index_path))
            self.documents = json.loads(self.docs_path.read_text(encoding="utf-8"))
            if getattr(self.index, "metric_type", faiss.METRIC_L2) != faiss.METRIC_INNER_PRODUCT:
                self.build(self.documents)

    def build(self, documents: list[dict[str, Any]]) -> int:
        clean = [dict(doc, text=str(doc["text"]).strip()) for doc in documents if str(doc.get("text", "")).strip()]
        if not clean:
            raise ValueError("At least one non-empty document is required")
        embeddings = self.embedder.encode([doc["text"] for doc in clean])
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(np.ascontiguousarray(embeddings))
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        self.docs_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
        with self._lock:
            self.index, self.documents = index, clean
        return len(clean)

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if self.index is None:
            self.load()
        k = min(top_k or self.settings.top_k, len(self.documents))
        scores, indices = self.index.search(self.embedder.encode(query), k)
        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
            if idx < 0:
                continue
            similarity = float(np.clip(score, -1, 1))
            results.append({"document": self.documents[idx], "similarity": round(similarity, 4),
                            "distance": round(1 - similarity, 4), "rank": rank})
        return results


_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    global _service
    if _service is None:
        _service = RetrievalService(get_settings(), get_embedding_service())
    return _service


def retrieve_evidence(question: str, top_k: int = 1) -> str | list[dict[str, Any]]:
    """Compatibility helper; structured results are returned when top_k > 1."""
    results = get_retrieval_service().retrieve(question, top_k)
    return results[0]["document"]["text"] if top_k == 1 and results else results
