from __future__ import annotations

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.rag import (
    CHUNKER_OPTIONS,
    DEFAULT_OLLAMA_MODEL,
    GENERATOR_OPTIONS,
    answer_with_generator,
    ingest_website,
    retrieve,
)

app = FastAPI(title="Personal Website RAG Playground")
templates = Jinja2Templates(directory="src/templates")


def render_home(request: Request, **kwargs):
    base = {
        "request": request,
        "answer": None,
        "retrieved": [],
        "status": None,
        "metrics": None,
        "default_url": "https://passionfordata.de",
        "default_model": DEFAULT_OLLAMA_MODEL,
        "chunkers": CHUNKER_OPTIONS,
        "generators": GENERATOR_OPTIONS,
        "selected_chunker": "fixed",
        "selected_generator": "ollama_llama3_8b",
    }
    base.update(kwargs)
    return templates.TemplateResponse("index.html", base)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return render_home(request)


@app.post("/ingest", response_class=HTMLResponse)
def ingest(
    request: Request,
    url: str = Form(...),
    max_pages: int = Form(30),
    persist_dir: str = Form("./chroma_db"),
    chunker: str = Form("fixed"),
):
    stats = ingest_website(start_url=url, max_pages=max_pages, persist_dir=persist_dir, chunker=chunker)
    status = (
        f"Ingest complete: {stats['pages']} pages, {stats['chunks']} chunks, "
        f"time={stats['elapsed_seconds']}s, chunker={stats['chunker']}"
    )
    return render_home(
        request,
        status=status,
        metrics={"before": stats["usage_before"], "after": stats["usage_after"]},
        default_url=url,
        selected_chunker=chunker,
    )


@app.post("/ask", response_class=HTMLResponse)
def ask(
    request: Request,
    question: str = Form(...),
    k: int = Form(4),
    generator: str = Form("ollama_llama3_8b"),
    model: str = Form(DEFAULT_OLLAMA_MODEL),
    persist_dir: str = Form("./chroma_db"),
):
    retrieved = retrieve(question=question, persist_dir=persist_dir, k=k)
    answer = answer_with_generator(question=question, retrieved_docs=retrieved, generator=generator, model=model)
    return render_home(
        request,
        answer=answer,
        retrieved=retrieved,
        default_model=model,
        selected_generator=generator,
    )
