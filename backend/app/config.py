"""Application configuration loaded from environment variables."""
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("HD_APP_NAME", "Hallucination Detection AI")
    environment: str = os.getenv("HD_ENVIRONMENT", "development")
    log_level: str = os.getenv("HD_LOG_LEVEL", "INFO")
    model_name: str = os.getenv("HD_MODEL_NAME", "all-MiniLM-L6-v2")
    top_k: int = int(os.getenv("HD_TOP_K", "5"))
    hallucination_threshold: float = float(os.getenv("HD_HALLUCINATION_THRESHOLD", "0.35"))
    data_dir: Path = Path(os.getenv("HD_DATA_DIR", str(BACKEND_DIR / "data")))
    database_url: str = os.getenv("HD_DATABASE_URL", f"sqlite:///{(BACKEND_DIR / 'data' / 'history.db').as_posix()}")
    cors_origins: str = os.getenv("HD_CORS_ORIGINS", "http://localhost:5173")
    @property
    def allowed_origins(self) -> list[str]:
        return [value.strip() for value in self.cors_origins.split(",") if value.strip()]

@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if not 1 <= settings.top_k <= 20: raise ValueError("HD_TOP_K must be between 1 and 20")
    if not 0 <= settings.hallucination_threshold <= 1: raise ValueError("HD_HALLUCINATION_THRESHOLD must be between 0 and 1")
    return settings
