from __future__ import annotations

import argparse
import hashlib
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urljoin, urlparse

import chromadb
import requests
from bs4 import BeautifulSoup
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_community.llms import Ollama

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "personal_site"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

CHUNKER_OPTIONS = {
    "fixed": "Fixed-size char chunks with overlap",
    "sentence": "Sentence-aware window chunks",
    "paragraph": "Paragraph-aware chunks",
    "graph": "Graph-inspired chunks (heading/section neighborhoods)",
}

GENERATOR_OPTIONS = {
    "ollama_llama3_8b": "Ollama Llama 3.1 8B (decoder-only)",
    "ollama_mistral": "Ollama Mistral (decoder-only)",
    "seq2seq_flan_t5": "Seq2Seq (FLAN-T5 style, prompt format)",
    "extractive_only": "No generation (retrieval/extractive only)",
}


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source_url: str


def crawl_same_domain(start_url: str, max_pages: int = 30, timeout: int = 20) -> dict[str, str]:
    base_domain = urlparse(start_url).netloc
    queue = [start_url]
    visited: set[str] = set()
    pages: dict[str, str] = {}

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"[WARN] Could not fetch {url}: {exc}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()

        text = "\n".join(line.strip() for line in soup.get_text("\n").splitlines() if line.strip())
        if text:
            pages[url] = text

        for anchor in soup.find_all("a", href=True):
            next_url = urljoin(url, anchor["href"])
            parsed = urlparse(next_url)
            if parsed.netloc == base_domain and parsed.scheme in {"http", "https"}:
                normalized = parsed._replace(fragment="").geturl()
                if normalized not in visited and normalized not in queue:
                    queue.append(normalized)

    return pages


def chunk_fixed(text: str, source_url: str, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        part = text[start:end].strip()
        if part:
            digest = hashlib.md5(f"{source_url}:{start}:{part}".encode("utf-8")).hexdigest()
            chunks.append(Chunk(chunk_id=digest, text=part, source_url=source_url))
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def chunk_sentence(text: str, source_url: str, window_sentences: int = 6, overlap_sentences: int = 2) -> list[Chunk]:
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    chunks: list[Chunk] = []
    i = 0
    while i < len(sentences):
        part = ". ".join(sentences[i : i + window_sentences]).strip()
        if part:
            digest = hashlib.md5(f"{source_url}:{i}:{part}".encode("utf-8")).hexdigest()
            chunks.append(Chunk(chunk_id=digest, text=part, source_url=source_url))
        if i + window_sentences >= len(sentences):
            break
        i += max(1, window_sentences - overlap_sentences)
    return chunks


def chunk_paragraph(text: str, source_url: str, paragraphs_per_chunk: int = 4) -> list[Chunk]:
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[Chunk] = []
    for i in range(0, len(paragraphs), paragraphs_per_chunk):
        part = "\n".join(paragraphs[i : i + paragraphs_per_chunk])
        digest = hashlib.md5(f"{source_url}:{i}:{part}".encode("utf-8")).hexdigest()
        chunks.append(Chunk(chunk_id=digest, text=part, source_url=source_url))
    return chunks


def chunk_graph_like(text: str, source_url: str) -> list[Chunk]:
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks: list[Chunk] = []
    for i in range(len(lines)):
        neighborhood = lines[max(0, i - 2) : min(len(lines), i + 3)]
        part = "\n".join(neighborhood)
        digest = hashlib.md5(f"{source_url}:{i}:{part}".encode("utf-8")).hexdigest()
        chunks.append(Chunk(chunk_id=digest, text=part, source_url=source_url))
    return chunks


def chunk_text(text: str, source_url: str, chunker: str = "fixed") -> list[Chunk]:
    mapping: dict[str, Callable[[str, str], list[Chunk]]] = {
        "fixed": chunk_fixed,
        "sentence": chunk_sentence,
        "paragraph": chunk_paragraph,
        "graph": chunk_graph_like,
    }
    fn = mapping.get(chunker, chunk_fixed)
    return fn(text, source_url)


def _collection(persist_dir: str):
    client = chromadb.PersistentClient(path=persist_dir)
    embedder = SentenceTransformerEmbeddingFunction(model_name=DEFAULT_EMBED_MODEL)
    return client.get_or_create_collection(
        name=DEFAULT_COLLECTION,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )


def system_usage_snapshot() -> dict[str, str]:
    cpu_summary = "unknown"
    gpu_summary = "unknown"

    try:
        import psutil

        cpu_summary = f"cpu_percent={psutil.cpu_percent(interval=0.1):.1f}%, cores={psutil.cpu_count(logical=True)}"
    except Exception:
        cpu_summary = "psutil not installed"

    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_summary = f"gpu_util={util.gpu}%, vram_used_mb={mem.used // (1024 * 1024)}"
    except Exception:
        gpu_summary = "no NVML GPU metrics available"

    return {"cpu": cpu_summary, "gpu": gpu_summary}


def ingest_website(
    start_url: str,
    persist_dir: str = "./chroma_db",
    max_pages: int = 30,
    chunker: str = "fixed",
) -> dict[str, object]:
    t0 = time.perf_counter()
    before_usage = system_usage_snapshot()

    pages = crawl_same_domain(start_url=start_url, max_pages=max_pages)
    all_chunks: list[Chunk] = []
    for url, text in pages.items():
        all_chunks.extend(chunk_text(text=text, source_url=url, chunker=chunker))

    if all_chunks:
        collection = _collection(persist_dir)
        collection.upsert(
            ids=[c.chunk_id for c in all_chunks],
            documents=[c.text for c in all_chunks],
            metadatas=[{"source": c.source_url} for c in all_chunks],
        )

    elapsed_s = time.perf_counter() - t0
    after_usage = system_usage_snapshot()

    return {
        "pages": len(pages),
        "chunks": len(all_chunks),
        "elapsed_seconds": round(elapsed_s, 3),
        "usage_before": before_usage,
        "usage_after": after_usage,
        "chunker": chunker,
    }


def retrieve(question: str, persist_dir: str = "./chroma_db", k: int = 4) -> list[dict[str, str]]:
    collection = _collection(persist_dir)
    results = collection.query(query_texts=[question], n_results=k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    return [{"source": (m.get("source", "unknown") if m else "unknown"), "text": d} for d, m in zip(docs, metas)]


def answer_with_generator(question: str, retrieved_docs: list[dict[str, str]], generator: str, model: str | None = None) -> str:
    if not retrieved_docs:
        return "I couldn't find relevant content in the indexed website. Please ingest first."

    context = "\n\n".join([f"[{i+1}] Source: {d['source']}\n{d['text']}" for i, d in enumerate(retrieved_docs)])

    if generator == "extractive_only":
        return "\n\n".join([f"Source: {d['source']}\n{d['text']}" for d in retrieved_docs])

    selected_model = model or ("mistral" if generator == "ollama_mistral" else DEFAULT_OLLAMA_MODEL)
    prompt = (
        "You are a helpful assistant answering ONLY from context. "
        "If missing, say you don't know. Answer in English.\n\n"
        f"Question: {question}\n\nContext:\n{context}\n\n"
        "Return concise answer + Sources."
    )
    llm = Ollama(model=selected_model)
    return llm.invoke(prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG playground over personal website")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_cmd = sub.add_parser("ingest")
    ingest_cmd.add_argument("--url", required=True)
    ingest_cmd.add_argument("--max-pages", type=int, default=30)
    ingest_cmd.add_argument("--persist-dir", default="./chroma_db")
    ingest_cmd.add_argument("--chunker", default="fixed", choices=list(CHUNKER_OPTIONS.keys()))

    retrieve_cmd = sub.add_parser("retrieve")
    retrieve_cmd.add_argument("--question", required=True)
    retrieve_cmd.add_argument("--k", type=int, default=4)
    retrieve_cmd.add_argument("--persist-dir", default="./chroma_db")

    ask_cmd = sub.add_parser("ask")
    ask_cmd.add_argument("--question", required=True)
    ask_cmd.add_argument("--k", type=int, default=4)
    ask_cmd.add_argument("--persist-dir", default="./chroma_db")
    ask_cmd.add_argument("--generator", default="ollama_llama3_8b", choices=list(GENERATOR_OPTIONS.keys()))
    ask_cmd.add_argument("--model", default=None)

    args = parser.parse_args()

    if args.command == "ingest":
        stats = ingest_website(args.url, args.persist_dir, args.max_pages, chunker=args.chunker)
        print(stats)
    elif args.command == "retrieve":
        for i, item in enumerate(retrieve(args.question, args.persist_dir, args.k), start=1):
            print(f"[{i}] Source: {item['source']}\n{item['text']}\n")
    else:
        docs = retrieve(args.question, args.persist_dir, args.k)
        print(answer_with_generator(args.question, docs, generator=args.generator, model=args.model))


if __name__ == "__main__":
    main()
