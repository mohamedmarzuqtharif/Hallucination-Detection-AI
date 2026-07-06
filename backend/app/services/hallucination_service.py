"""Application use cases independent of HTTP."""
from typing import Any
from app.database import HistoryRepository
from app.detector import HallucinationDetector
from app.retrieval import RetrievalService

class HallucinationService:
    def __init__(self, detector: HallucinationDetector, history: HistoryRepository, retriever: RetrievalService) -> None:
        self.detector, self.history, self.retriever = detector, history, retriever
    def detect(self, question: str, answer: str, top_k: int) -> dict[str, Any]:
        result = self.detector.detect(question, answer, top_k)
        result["id"] = self.history.add(result)
        return result
    def generate(self, question: str, top_k: int) -> dict[str, Any]:
        evidence = self.retriever.retrieve(question, top_k)
        answer = " ".join(item["document"]["text"] for item in evidence[:2]) if evidence else "I do not have enough evidence to answer."
        return {"question": question, "answer": answer, "evidence": evidence, "note": "Extractive local generation"}
