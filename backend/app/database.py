"""SQLite persistence for detection history."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from app.config import get_settings

class HistoryRepository:
    def __init__(self, database_url: str) -> None:
        self.path = Path(database_url.removeprefix("sqlite:///"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS detections (
              id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL, answer TEXT NOT NULL,
              prediction TEXT NOT NULL, hallucination_score REAL NOT NULL, confidence REAL NOT NULL,
              payload TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)""")

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path); db.row_factory = sqlite3.Row
        try:
            yield db; db.commit()
        finally:
            db.close()

    def add(self, result: dict[str, Any]) -> int:
        with self.connection() as db:
            cursor = db.execute("INSERT INTO detections(question,answer,prediction,hallucination_score,confidence,payload) VALUES(?,?,?,?,?,?)",
                (result["question"], result["answer"], result["prediction"], result["hallucination_score"], result["confidence"], json.dumps(result)))
            return int(cursor.lastrowid or 0)

    def list(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self.connection() as db:
            rows = db.execute("SELECT id,payload,created_at FROM detections ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        return [dict(json.loads(row["payload"]), id=row["id"], created_at=row["created_at"]) for row in rows]

_repository: HistoryRepository | None = None
def get_history_repository() -> HistoryRepository:
    global _repository
    if _repository is None: _repository = HistoryRepository(get_settings().database_url)
    return _repository
