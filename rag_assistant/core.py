"""Cœur du RAG : indexation (ingestion) et réponse aux questions."""

from __future__ import annotations

import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from . import config
from .loaders import iter_files, load_file, split
from .prompts import NO_ANSWER, PROMPT_VERSION, SYSTEM_PROMPT, USER_TEMPLATE


# --- Fabriques ----------------------------------------------------------------

class PrefixedEmbeddings(Embeddings):
    """Ajoute un préfixe de tâche avant chaque texte, différent pour les morceaux indexés et pour les
    questions (convention de nomic-embed-text). Le texte enregistré dans Chroma n'est pas modifié."""

    def __init__(self, inner: Embeddings, query_prefix: str = "", doc_prefix: str = ""):
        self.inner = inner
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    @staticmethod
    def _with(prefix: str, text: str) -> str:
        return f"{prefix} {text}" if prefix else text

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents([self._with(self.doc_prefix, t) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(self._with(self.query_prefix, text))


def build_embeddings() -> Embeddings:
    from langchain_ollama import OllamaEmbeddings

    base = OllamaEmbeddings(model=config.EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
    if not (config.EMBED_QUERY_PREFIX or config.EMBED_DOC_PREFIX):
        return base
    return PrefixedEmbeddings(base, config.EMBED_QUERY_PREFIX, config.EMBED_DOC_PREFIX)


def build_llm() -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(model=config.LLM_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0,
                      num_ctx=config.LLM_NUM_CTX)


def build_vectorstore(embeddings: Embeddings | None = None, persist_dir: Path | None = None):
    from langchain_chroma import Chroma

    return Chroma(
        collection_name=config.COLLECTION,
        embedding_function=embeddings or build_embeddings(),
        persist_directory=str(persist_dir or config.CHROMA_DIR),
        collection_configuration={"hnsw": {"space": "cosine"}},  # score de pertinence = 1 − distance cosinus
    )


# --- Indexation -----------------------------------------------------------------

@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    chunks_written: int = 0


class Indexer:
    """Ingestion idempotente : un fichier inchangé n'est pas réindexé ; un fichier modifié
    remplace ses anciens morceaux ; un fichier supprimé du dossier est retiré de l'index."""

    def __init__(self, vectorstore):
        self.vs = vectorstore

    def _indexed(self) -> dict[str, tuple[str, list[str]]]:
        got = self.vs.get(include=["metadatas"])
        out: dict[str, tuple[str, list[str]]] = {}
        for id_, meta in zip(got["ids"], got["metadatas"]):
            src = meta.get("source", "?")
            out.setdefault(src, (meta.get("file_hash", ""), []))[1].append(id_)
        return out

    def ingest_dir(self, directory: Path) -> IngestReport:
        report = IngestReport()
        indexed = self._indexed()
        files = iter_files(directory)
        for path in files:
            docs = load_file(path)
            new_hash = docs[0].metadata["file_hash"] if docs else ""
            old = indexed.get(path.name)
            if old and old[0] == new_hash:
                report.unchanged.append(path.name)
                continue
            if old:
                self.vs.delete(ids=old[1])
                report.updated.append(path.name)
            else:
                report.added.append(path.name)
            chunks = split(docs)
            if chunks:
                ids = [f"{path.name}:{new_hash}:{c.metadata['chunk']}" for c in chunks]
                self.vs.add_documents(chunks, ids=ids)
                report.chunks_written += len(chunks)
        present = {p.name for p in files}
        for src, (_, ids) in indexed.items():
            if src not in present:
                self.vs.delete(ids=ids)
                report.removed.append(src)
        return report


# --- Réponse aux questions -----------------------------------------------------

@dataclass
class Source:
    ref: int
    source: str
    section: str
    score: float
    excerpt: str


@dataclass
class Answer:
    question: str
    answer: str
    answered: bool
    sources: list[Source]
    llm_called: bool
    latency_ms: int
    model: str
    prompt_version: str = PROMPT_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


class Assistant:
    def __init__(self, vectorstore, llm: BaseChatModel, top_k: int | None = None, min_relevance: float | None = None):
        self.vs = vectorstore
        self.llm = llm
        self.top_k = top_k or config.TOP_K
        self.min_relevance = config.MIN_RELEVANCE if min_relevance is None else min_relevance

    def retrieve(self, question: str) -> list[Source]:
        hits = self.vs.similarity_search_with_relevance_scores(question, k=self.top_k)
        kept = [(d, s) for d, s in hits if s >= self.min_relevance]
        return [
            Source(ref=i, source=d.metadata.get("source", "?"), section=d.metadata.get("section", ""),
                   score=round(float(s), 3), excerpt=d.page_content)
            for i, (d, s) in enumerate(kept, start=1)
        ]

    def ask(self, question: str) -> Answer:
        t0 = time.perf_counter()
        model = getattr(self.llm, "model", type(self.llm).__name__)
        sources = self.retrieve(question)
        if not sources:
            # Garde-fou : rien de pertinent → on ne laisse pas le LLM improviser.
            return Answer(question, NO_ANSWER, False, [], False, int((time.perf_counter() - t0) * 1000), model)

        context = "\n\n".join(
            f"[{s.ref}] (source : {s.source}{' › ' + s.section if s.section else ''})\n{s.excerpt}" for s in sources
        )
        messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(USER_TEMPLATE.format(context=context, question=question))]
        text = str(self.llm.invoke(messages).content).strip()

        answered = NO_ANSWER.lower()[:40] not in text.lower()
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        # On ne renvoie que les sources effectivement citées (ou toutes si le modèle n'a rien cité).
        shown = ([s for s in sources if s.ref in cited] or sources) if answered else []
        return Answer(question, text, answered, shown, True, int((time.perf_counter() - t0) * 1000), model)
