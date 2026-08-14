# GenAI Business Use Case Prototype on Google Cloud

This project is a low-cost portfolio prototype for an HR Policy Assistant Builder. An HR user can provide policy documents, build a local vector index, and test policy Q&A retrieval before moving to a hosted Google Cloud deployment.

The MVP uses a local FAISS index so the first version can run on a laptop without managed vector database costs.

## What The Demo Does

- Reads synthetic Markdown policy documents from `data/policies/`
- Splits each policy into citation-friendly sections
- Creates local vectors with a no-download hashing embedder by default
- Stores vectors in a local FAISS index
- Lets you search policy sections from the command line or Streamlit
- Shows the path toward a Gemini-powered answer generator

## Setup With Conda

FAISS officially recommends conda for prebuilt installs. From this folder:

```bash
conda env create -f environment.yml
conda activate genai-policy-builder
```

## Setup With Pip

Pip may work on many laptops, but conda is the safer FAISS path.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Build The Local Vector Index

```bash
python scripts/ingest_faiss.py
```

The default embedder is a lightweight local hashing embedder. It is less semantic than a modern embedding model, but it is free, private, fast, and good enough for the first builder demo.

To try Sentence Transformers later:

```bash
python scripts/ingest_faiss.py --embedder sentence-transformer
```

This creates:

```text
vector_index/
├── manifest.json
├── policy.index
└── records.json
```

## Test Retrieval From The Command Line

```bash
python scripts/query_faiss.py "Can I get reimbursed for a home office monitor?"
```

## Start The Streamlit Demo

```bash
streamlit run app.py
```

## Upgrade Path

Tier 0: Static GitHub demo with screenshots and sample answers.

Tier 1: Local builder using FAISS on a laptop.

Tier 2: Hosted Cloud Run demo using Gemini for answer generation.

Tier 3: Production pattern with authentication, audit logs, document versioning, Cloud Storage, Firestore, and managed vector search.

## Responsible AI Controls

- Synthetic demo data only
- Local retrieval before generation
- Citation metadata on every chunk
- Insufficient-context handling
- Risk labels for HR, Finance, Legal, Security, and Privacy topics
- Human review expected before taking business action
