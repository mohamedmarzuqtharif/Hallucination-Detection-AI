"""Evidence-grounding hallucination detector."""
from typing import Any

import numpy as np

from app.config import Settings, get_settings
from app.embeddings import EmbeddingService, get_embedding_service
from app.retrieval import RetrievalService, get_retrieval_service


class HallucinationDetector:
    def __init__(self, settings: Settings, embedder: EmbeddingService, retriever: RetrievalService) -> None:
        self.settings, self.embedder, self.retriever = settings, embedder, retriever

    def detect(self, question: str, answer: str, top_k: int | None = None) -> dict[str, Any]:
        evidence = self.retriever.retrieve(question, top_k)
        evidence_texts = [item["document"]["text"] for item in evidence]
        answer_vector = self.embedder.encode(answer)[0]
        evidence_vectors = self.embedder.encode(evidence_texts)
        similarities = evidence_vectors @ answer_vector
        best = max(float(np.max(similarities)), 0.0) if len(similarities) else 0.0
        score = float(np.clip(1 - best, 0, 1))
        hallucinated = score > self.settings.hallucination_threshold
        confidence = score if hallucinated else best
        reason = ("The answer has weak semantic support in the retrieved knowledge base."
                  if hallucinated else "The answer is semantically supported by the retrieved evidence.")
        return {"question": question, "answer": answer, "prediction": "Hallucinated" if hallucinated else "Grounded",
                "hallucination_score": round(score, 4), "confidence": round(confidence, 4),
                "similarity": round(best, 4), "retrieved_evidence": evidence, "reason": reason}


def get_detector() -> HallucinationDetector:
    return HallucinationDetector(get_settings(), get_embedding_service(), get_retrieval_service())


def detect_hallucination(question: str, answer: str) -> dict[str, Any]:
    return get_detector().detect(question, answer)
