#!/usr/bin/env python3
"""Build a local FAISS index from Markdown policy documents."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import faiss
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDER = "hashing"
HASHING_DIMENSIONS = 1024


def parse_frontmatter(markdown: str) -> tuple[dict[str, str], str]:
    if not markdown.startswith("---\n"):
        return {}, markdown

    end = markdown.find("\n---\n", 4)
    if end == -1:
        return {}, markdown

    raw_meta = markdown[4:end].strip()
    body = markdown[end + len("\n---\n") :]
    metadata: dict[str, str] = {}

    for line in raw_meta.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')

    return metadata, body


def split_sections(body: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"^##\s+(.+)$", body, flags=re.MULTILINE))
    if not matches:
        return [("Full Document", body.strip())]

    sections: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        section_title = match.group(1).strip()
        section_text = body[start:end].strip()
        sections.append((section_title, section_text))

    return sections


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "section"


def load_records(source_dir: Path) -> list[dict]:
    records: list[dict] = []

    for path in sorted(source_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue

        metadata, body = parse_frontmatter(path.read_text(encoding="utf-8"))
        document_id = metadata.get("document_id", path.stem)
        title = metadata.get("title", path.stem.replace("-", " ").title())

        for section_title, section_text in split_sections(body):
            record_id = f"{document_id}::{slugify(section_title)}"
            records.append(
                {
                    "id": record_id,
                    "text": section_text,
                    "metadata": {
                        **metadata,
                        "document_id": document_id,
                        "title": title,
                        "section": section_title,
                        "source_file": path.name,
                    },
                }
            )

    return records


def build_embeddings(records: list[dict], embedder: str, model_name: str) -> np.ndarray:
    texts = [record["text"] for record in records]

    if embedder == "sentence-transformer":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    elif embedder == "hashing":
        vectorizer = HashingVectorizer(
            n_features=HASHING_DIMENSIONS,
            alternate_sign=False,
            norm=None,
            ngram_range=(1, 2),
            stop_words="english",
        )
        embeddings = vectorizer.transform(texts).toarray()
    else:
        raise ValueError(f"Unknown embedder: {embedder}")

    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings


def build_index(records: list[dict], embedder: str, model_name: str) -> faiss.Index:
    embeddings = build_embeddings(records, embedder, model_name)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local FAISS policy index.")
    parser.add_argument("--source", default="data/policies", help="Markdown policy folder")
    parser.add_argument("--out", default="vector_index", help="Output folder")
    parser.add_argument(
        "--embedder",
        default=DEFAULT_EMBEDDER,
        choices=["hashing", "sentence-transformer"],
        help="Embedding backend. Use hashing for a no-download local MVP.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Sentence Transformer model")
    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(source_dir)
    if not records:
        raise SystemExit(f"No Markdown policy records found in {source_dir}")

    index = build_index(records, args.embedder, args.model)

    faiss.write_index(index, str(output_dir / "policy.index"))
    (output_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "embedder": args.embedder,
                "dimensions": index.d,
                "record_count": len(records),
                "source_dir": str(source_dir),
                "index_type": "IndexFlatIP with L2-normalized embeddings",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Indexed {len(records)} policy sections into {output_dir}")


if __name__ == "__main__":
    main()
