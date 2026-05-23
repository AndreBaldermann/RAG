from __future__ import annotations

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from src.rag import DEFAULT_OLLAMA_MODEL, answer_with_ollama, ingest_website, retrieve

app = FastAPI(title="Personal Website RAG")
templates = Jinja2Templates(directory="src/templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer": None,
            "retrieved": [],
            "default_url": "https://passionfordata.de",
            "default_model": DEFAULT_OLLAMA_MODEL,
        },
    )


@app.post("/ingest", response_class=HTMLResponse)
def ingest(request: Request, url: str = Form(...), max_pages: int = Form(30), persist_dir: str = Form("./chroma_db")):
    pages, chunks = ingest_website(start_url=url, max_pages=max_pages, persist_dir=persist_dir)
    status = f"Ingest complete: {pages} pages, {chunks} chunks."
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "status": status,
            "answer": None,
            "retrieved": [],
            "default_url": url,
            "default_model": DEFAULT_OLLAMA_MODEL,
        },
    )


@app.post("/ask", response_class=HTMLResponse)
def ask(
    request: Request,
    question: str = Form(...),
    k: int = Form(4),
    model: str = Form(DEFAULT_OLLAMA_MODEL),
    persist_dir: str = Form("./chroma_db"),
):
    retrieved = retrieve(question=question, persist_dir=persist_dir, k=k)
    answer = answer_with_ollama(question=question, retrieved_docs=retrieved, model=model)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "answer": answer,
            "retrieved": retrieved,
            "default_url": "https://passionfordata.de",
            "default_model": model,
        },
    )
