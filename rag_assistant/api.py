"""API REST (FastAPI).

  uvicorn rag_assistant.api:app --reload
  → documentation interactive : http://localhost:8000/docs
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config
from .core import Assistant, Indexer, build_llm, build_vectorstore


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["Mon assurance couvre-t-elle un dégât d'eau ?"])


@lru_cache
def get_vectorstore():
    return build_vectorstore()


@lru_cache
def get_assistant() -> Assistant:
    return Assistant(get_vectorstore(), build_llm())


app = FastAPI(title="Mes papiers — assistant privé pour vos documents", version="1.0")


@app.get("/health")
def health():
    return {"status": "ok", "llm": config.LLM_MODEL, "embeddings": config.EMBED_MODEL}


@app.post("/ingest")
def ingest(vs=Depends(get_vectorstore)):
    report = Indexer(vs).ingest_dir(config.DOCS_DIR)
    return report.__dict__


@app.post("/ask")
def ask(req: AskRequest, assistant: Assistant = Depends(get_assistant)):
    try:
        return assistant.ask(req.question).to_dict()
    except ConnectionError as exc:  # Ollama arrêté
        raise HTTPException(status_code=503, detail=f"LLM indisponible : {exc}") from exc
