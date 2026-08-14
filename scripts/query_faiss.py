#!/usr/bin/env python3
"""Query the local FAISS policy index from the command line."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


def load_index(index_dir: Path):
    index = faiss.read_index(str(index_dir / "policy.index"))
    records = json.loads((index_dir / "records.json").read_text(encoding="utf-8"))
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    return index, records, manifest


def search(question: str, index_dir: Path, top_k: int) -> list[dict]:
    index, records, manifest = load_index(index_dir)

    if manifest.get("embedder") == "sentence-transformer":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(manifest["model"])
        query = model.encode([question], convert_to_numpy=True).astype("float32")
    else:
        vectorizer = HashingVectorizer(
            n_features=int(manifest.get("dimensions", 1024)),
            alternate_sign=False,
            norm=None,
            ngram_range=(1, 2),
            stop_words="english",
        )
        query = vectorizer.transform([question]).toarray().astype("float32")

    faiss.normalize_L2(query)

    scores, ids = index.search(query, top_k)
    results = []
    for score, record_index in zip(scores[0], ids[0]):
        if record_index < 0:
            continue
        record = records[int(record_index)]
        results.append({"score": float(score), **record})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Search local policy docs with FAISS.")
    parser.add_argument("question", help="Question to search for")
    parser.add_argument("--index", default="vector_index", help="FAISS index folder")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    args = parser.parse_args()

    results = search(args.question, Path(args.index), args.top_k)
    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]
        print(f"\n[{rank}] score={result['score']:.3f}")
        print(f"{metadata['title']} - {metadata['section']}")
        print(result["text"][:700].strip())


if __name__ == "__main__":
    main()
