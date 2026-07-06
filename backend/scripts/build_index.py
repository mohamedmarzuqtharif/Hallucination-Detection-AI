"""Build the FAISS index from a JSON document list."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.retrieval import get_retrieval_service

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path(__file__).parents[1] / "data/documents/knowledge_base.json")
    args = parser.parse_args()
    documents = json.loads(args.input.read_text(encoding="utf-8"))
    print(f"Indexed {get_retrieval_service().build(documents)} documents")

if __name__ == "__main__":
    main()
