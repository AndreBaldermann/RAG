# RAG Playground

A versatile local RAG playground with:
- multiple chunking concepts (fixed/sentence/paragraph/graph-like)
- multiple generator concepts (decoder-only, seq2seq-style mode, extractive)
- ChromaDB retrieval + Ollama generation
- FastAPI Web UI with ingest timing and CPU/GPU usage snapshots

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b
ollama serve
```

## Run Web UI

```bash
uvicorn src.web:app --reload
```

Open: `http://127.0.0.1:8000`

## Features

### Chunking concepts
- `fixed`: fixed-length char chunks with overlap
- `sentence`: sentence-window chunking
- `paragraph`: paragraph-window chunking
- `graph`: graph-inspired neighborhood chunks over nearby lines

### Generator concepts
- `ollama_llama3_8b`: decoder-only local generation
- `ollama_mistral`: decoder-only alternative
- `seq2seq_flan_t5`: seq2seq-style prompting mode
- `extractive_only`: no generation, just retrieved passages

## Notes
- Single-domain public crawl works well for `https://passionfordata.de`.
- English-only responses are default to reduce multilingual instability in smaller local models.
- CPU/GPU metrics are best-effort snapshots (GPU requires NVML-compatible setup).
