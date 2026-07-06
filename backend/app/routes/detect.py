"""Hallucination workflow endpoints."""
from fastapi import APIRouter, Depends, Query
from app.database import HistoryRepository, get_history_repository
from app.detector import get_detector
from app.retrieval import get_retrieval_service
from app.schemas.request import DetectRequest, GenerateRequest, ImportRequest
from app.services.hallucination_service import HallucinationService

router = APIRouter(tags=["hallucination detection"])
def get_service(history: HistoryRepository = Depends(get_history_repository)) -> HallucinationService:
    return HallucinationService(get_detector(), history, get_retrieval_service())
@router.post("/detect")
def detect(request: DetectRequest, service: HallucinationService = Depends(get_service)):
    return service.detect(request.question, request.answer, request.top_k)
@router.post("/generate")
def generate(request: GenerateRequest, service: HallucinationService = Depends(get_service)):
    return service.generate(request.question.strip(), request.top_k)
@router.get("/history")
def history(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), repository: HistoryRepository = Depends(get_history_repository)):
    return {"items": repository.list(limit, offset), "limit": limit, "offset": offset}
@router.post("/knowledge-base/import")
def import_documents(request: ImportRequest):
    return {"indexed_documents": get_retrieval_service().build(request.documents)}
