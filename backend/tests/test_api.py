"""API tests with lightweight dependency overrides."""
from fastapi.testclient import TestClient
from app.main import app
from app.routes.detect import get_service

class FakeService:
    def detect(self, question, answer, top_k):
        return {"question": question, "answer": answer, "prediction": "Grounded", "hallucination_score": .1,
                "confidence": .9, "similarity": .9, "retrieved_evidence": [], "reason": "supported", "id": 1}
    def generate(self, question, top_k):
        return {"question": question, "answer": "Evidence", "evidence": []}

app.dependency_overrides[get_service] = lambda: FakeService()
client = TestClient(app)

def test_health():
    assert client.get("/health").json()["status"] == "healthy"
def test_detect_validation():
    assert client.post("/detect", json={"question": "", "answer": "x"}).status_code == 422
def test_detect():
    response = client.post("/detect", json={"question": "What is Earth?", "answer": "A planet."})
    assert response.status_code == 200
    assert response.json()["prediction"] == "Grounded"
