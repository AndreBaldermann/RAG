# RAG

Simple personal-website RAG using **ChromaDB** + **Ollama (Llama 3.1 8B)** + optional **FastAPI UI**.

## Scope (your preferences)
- Public website crawling only (no auth/cookies)
- Single domain only
- English answers only
- Target website: `https://passionfordata.de`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Also run Ollama locally:

```bash
ollama pull llama3.1:8b
ollama serve
```

## CLI

### 1) Ingest
```bash
python -m src.rag ingest --url https://passionfordata.de --max-pages 30
```

### 2) Retrieval only
```bash
python -m src.rag retrieve --question "What topics does the website cover?" --k 4
```

### 3) Final answer with Ollama
```bash
python -m src.rag ask --question "What services are offered?" --k 4 --model llama3.1:8b
```

## Web UI (FastAPI)

```bash
uvicorn src.web:app --reload
```

Open `http://127.0.0.1:8000`.

## Does multilingual cause problems?
Not necessarily, but it can reduce retrieval precision with small local models and mixed-language content.
For your setup, **English-only answers** are a good default.
