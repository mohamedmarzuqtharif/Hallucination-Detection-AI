# Verity AI — Hallucination Detection

A full-stack retrieval-augmented system that checks whether an LLM answer is supported by a local knowledge base. It retrieves evidence with normalized Sentence Transformer embeddings and FAISS, scores answer-to-evidence similarity, explains the decision, and persists each analysis.

## Architecture

```text
React/Vite UI ──► FastAPI routes ──► HallucinationService
                                          ├── SentenceTransformer
                                          ├── FAISS top-K retrieval
                                          ├── Evidence detector
                                          └── SQLite history
```

The API is intentionally provider-neutral. `/generate` supplies a safe extractive baseline; a hosted or local LLM can later be added behind the service without changing routes or the UI.

## Features

- `POST /detect`: prediction, hallucination score, confidence, reason, and ranked evidence
- `POST /generate`: locally grounded extractive answer
- `GET /history`, `GET /health`, and interactive Swagger at `/docs`
- `POST /knowledge-base/import`: replace documents and rebuild the FAISS index
- Validated inputs, environment settings, CORS, request logging, and SQLite persistence
- Responsive React UI with dark mode, charts, history, loading, and error states
- Docker images, API tests, and classical-ML training/evaluation entry points

## Local setup

Requires Python 3.11+ and Node 20+.

```bash
python -m venv venv
# Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
cd backend
python scripts/build_index.py
uvicorn app.main:app --reload
```

The first model use downloads `sentence-transformers/all-MiniLM-L6-v2`. In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`; the API runs at `http://localhost:8000`.

Copy `backend/.env.example` to `backend/.env` and `frontend/.env.example` to `frontend/.env` when overriding defaults.

## API example

```bash
curl -X POST http://localhost:8000/detect -H "Content-Type: application/json" \
  -d '{"question":"Who created Python?","answer":"Guido van Rossum created Python in 1991.","top_k":5}'
```

Documents must contain a non-empty `text` property; arbitrary metadata is preserved:

```json
{"documents":[{"id":"python-1","text":"Python was created by Guido van Rossum.","source":"internal"}]}
```

## Tests and production

```bash
cd backend && pytest
cd frontend && npm run build
docker compose up --build
```

For Render, deploy `backend` with start command `uvicorn app.main:app --host 0.0.0.0 --port $PORT`. For Vercel, deploy `frontend` and set `VITE_API_URL` to the backend URL. Use a persistent disk for SQLite/FAISS, or switch the history repository to PostgreSQL.

## Repository map

- `backend/app`: configuration, routes, services, retrieval, detection, persistence
- `backend/data`: source documents and generated FAISS artifacts
- `backend/scripts`: dataset and index utilities
- `backend/tests`: API tests with model-independent dependency overrides
- `frontend/src`: pages, visualizations, navigation, API layer
- `training`: Random Forest/XGBoost training, RoBERTa fine-tuning, evaluation, and shared metrics

## Detection caveats

Semantic similarity is a useful grounding signal, not a universal fact checker. Negation, numerical precision, multi-claim answers, and incomplete knowledge bases require stronger NLI or claim-level models. Natural extensions include cross-encoder reranking, calibrated thresholds, XGBoost, RoBERTa NLI fine-tuning, PostgreSQL, authentication, monitoring, and dataset-version tracking.
