"""API request contracts."""
from typing import Any
from pydantic import BaseModel, Field, field_validator

class DetectRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    answer: str = Field(min_length=1, max_length=10000)
    top_k: int = Field(default=5, ge=1, le=20)
    @field_validator("question", "answer")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value: raise ValueError("must not be blank")
        return value

class GenerateRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)

class ImportRequest(BaseModel):
    documents: list[dict[str, Any]] = Field(min_length=1)
