"""FastAPI application entry point."""
import logging
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.config import get_settings
from app.routes.detect import router

settings = get_settings()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)
app = FastAPI(title=settings.app_name, version="1.0.0", description="Evidence-grounded LLM response verification API")
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
@app.middleware("http")
async def request_logging(request: Request, call_next):
    started = time.perf_counter(); response = await call_next(request)
    logger.info("%s %s %s %.1fms", request.method, request.url.path, response.status_code, (time.perf_counter() - started) * 1000)
    return response
@app.get("/", tags=["system"])
def home(): return {"name": settings.app_name, "status": "running", "docs": "/docs"}
@app.get("/health", tags=["system"])
def health(): return {"status": "healthy", "environment": settings.environment}
