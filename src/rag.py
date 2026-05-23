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


def chunk_text(text: str, source_url: str, chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
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


def _collection(persist_dir: str):
    client = chromadb.PersistentClient(path=persist_dir)
    embedder = SentenceTransformerEmbeddingFunction(model_name=DEFAULT_EMBED_MODEL)
    return client.get_or_create_collection(
        name=DEFAULT_COLLECTION,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_website(start_url: str, persist_dir: str = "./chroma_db", max_pages: int = 30) -> tuple[int, int]:
    pages = crawl_same_domain(start_url=start_url, max_pages=max_pages)
    all_chunks: list[Chunk] = []
    for url, text in pages.items():
        all_chunks.extend(chunk_text(text=text, source_url=url))

    if not all_chunks:
        return 0, 0

    collection = _collection(persist_dir)
    collection.upsert(
        ids=[c.chunk_id for c in all_chunks],
        documents=[c.text for c in all_chunks],
        metadatas=[{"source": c.source_url} for c in all_chunks],
    )
    return len(pages), len(all_chunks)


def retrieve(question: str, persist_dir: str = "./chroma_db", k: int = 4) -> list[dict[str, str]]:
    collection = _collection(persist_dir)
    results = collection.query(query_texts=[question], n_results=k)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    output = []
    for doc, meta in zip(docs, metas):
        source = meta.get("source", "unknown") if meta else "unknown"
        output.append({"source": source, "text": doc})
    return output


def answer_with_ollama(
    question: str,
    retrieved_docs: list[dict[str, str]],
    model: str = DEFAULT_OLLAMA_MODEL,
) -> str:
    if not retrieved_docs:
        return "I couldn't find relevant content in the indexed website. Please ingest first."

    context_blocks = []
    for i, item in enumerate(retrieved_docs, start=1):
        context_blocks.append(f"[{i}] Source: {item['source']}\n{item['text']}")
    context = "\n\n".join(context_blocks)

    prompt = f"""You are a helpful assistant answering questions ONLY from the provided website context.
If the answer is not in the context, say you don't know.
Answer in English.

Question: {question}

Context:
{context}

Return:
1) A concise answer.
2) A 'Sources:' section listing source URLs used.
"""

    llm = Ollama(model=model)
    return llm.invoke(prompt)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG over personal website with ChromaDB + optional Ollama answers")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_cmd = sub.add_parser("ingest", help="Crawl website and index into ChromaDB")
    ingest_cmd.add_argument("--url", required=True, help="Start URL, e.g. https://passionfordata.de")
    ingest_cmd.add_argument("--max-pages", type=int, default=30)
    ingest_cmd.add_argument("--persist-dir", default="./chroma_db")

    retrieve_cmd = sub.add_parser("retrieve", help="Retrieve top-k chunks")
    retrieve_cmd.add_argument("--question", required=True)
    retrieve_cmd.add_argument("--k", type=int, default=4)
    retrieve_cmd.add_argument("--persist-dir", default="./chroma_db")

    ask_cmd = sub.add_parser("ask", help="Generate answer with Ollama from retrieved context")
    ask_cmd.add_argument("--question", required=True)
    ask_cmd.add_argument("--k", type=int, default=4)
    ask_cmd.add_argument("--persist-dir", default="./chroma_db")
    ask_cmd.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)

    args = parser.parse_args()

    if args.command == "ingest":
        pages, chunks = ingest_website(args.url, args.persist_dir, args.max_pages)
        print(f"Ingest complete: {pages} pages, {chunks} chunks.")

    elif args.command == "retrieve":
        docs = retrieve(args.question, args.persist_dir, args.k)
        if not docs:
            print("No hits found. Run ingest first.")
            return
        for i, item in enumerate(docs, start=1):
            print(f"[{i}] Source: {item['source']}\n{item['text']}\n")

    elif args.command == "ask":
        docs = retrieve(args.question, args.persist_dir, args.k)
        print(answer_with_ollama(args.question, docs, model=args.model))


if __name__ == "__main__":
    main()
