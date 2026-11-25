from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_assistant import api, config
from rag_assistant.core import Assistant, Indexer, PrefixedEmbeddings, build_vectorstore
from rag_assistant.loaders import load_file, split
from rag_assistant.prompts import NO_ANSWER

from .conftest import ExplodingLLM, HashEmbeddings


# --- Chargement et découpage ---------------------------------------------------

def test_markdown_is_split_by_section_with_titles(docs_dir):
    docs = load_file(docs_dir / "assurance_habitation.md")
    sections = [d.metadata["section"] for d in docs]
    assert any("Dégâts d'eau" in s for s in sections)
    assert all(d.metadata["source"] == "assurance_habitation.md" for d in docs)


def test_pdf_is_loaded_page_by_page(docs_dir):
    docs = load_file(docs_dir / "abonnement_gym.pdf")
    assert docs and docs[0].metadata["page"] == 1 and "39,99 $" in docs[0].page_content


def test_chunks_respect_size():
    docs = load_file(config.DOCS_DIR / "assurance_habitation.md")
    assert all(len(c.page_content) <= 300 for c in split(docs, chunk_size=300, chunk_overlap=50))


# --- Ingestion idempotente -------------------------------------------------------

def test_ingest_is_idempotent_and_tracks_changes(docs_dir, vectorstore):
    idx = Indexer(vectorstore)
    first = idx.ingest_dir(docs_dir)
    assert len(first.added) == 6 and first.chunks_written > 6
    total = len(vectorstore.get()["ids"])

    second = idx.ingest_dir(docs_dir)
    assert second.added == [] and len(second.unchanged) == 6
    assert len(vectorstore.get()["ids"]) == total  # aucun doublon

    (docs_dir / "forfait_mobile.md").write_text("# Forfait\n\nNouveau forfait à 50 $ par mois.", encoding="utf-8")
    (docs_dir / "garantie_lave_vaisselle.md").unlink()
    third = idx.ingest_dir(docs_dir)
    assert third.updated == ["forfait_mobile.md"]
    assert third.removed == ["garantie_lave_vaisselle.md"]
    sources = {m["source"] for m in vectorstore.get(include=["metadatas"])["metadatas"]}
    assert "garantie_lave_vaisselle.md" not in sources


# --- Préfixes de tâche (nomic-embed-text) ------------------------------------------

def test_prefixes_go_to_the_model_but_not_into_the_index(docs_dir, tmp_path):
    seen = []

    class Recorder(HashEmbeddings):
        def _vec(self, text):
            seen.append(text)
            return super()._vec(text)

    emb =PrefixedEmbeddings(Recorder(), query_prefix="search_query:", doc_prefix="search_document:")
    vs = build_vectorstore(embeddings=emb, persist_dir=tmp_path / "chroma_prefix")
    Indexer(vs).ingest_dir(docs_dir)
    assert seen and all(t.startswith("search_document: ") for t in seen)

    seen.clear()
    vs.similarity_search("montant du loyer", k=1)
    assert seen == ["search_query: montant du loyer"]
    stored = vs.get(include=["documents"])["documents"]
    assert not any(d.startswith("search_document:") for d in stored)  # les citations restent propres


# --- Réponses --------------------------------------------------------------------

def make_assistant(vectorstore, docs_dir, llm, **kw):
    Indexer(vectorstore).ingest_dir(docs_dir)
    return Assistant(vectorstore, llm, **kw)


def test_retrieval_finds_the_right_document(docs_dir, vectorstore):
    a = make_assistant(vectorstore, docs_dir, ExplodingLLM(responses=[]), min_relevance=0.1)
    srcs = [s.source for s in a.retrieve("vol de vélo verrouillé à l'extérieur du domicile")]
    assert srcs[0] == "assurance_habitation.md"


def test_answer_keeps_only_cited_sources(docs_dir, vectorstore):
    llm = FakeListChatModel(responses=["Le vol de vélo est couvert jusqu'à 1 500 $ [2]."])
    a = make_assistant(vectorstore, docs_dir, llm, min_relevance=0.0)
    ans = a.ask("vol de vélo verrouillé à l'extérieur")
    assert ans.answered and ans.llm_called
    assert [s.ref for s in ans.sources] == [2]


def test_irrelevant_question_is_refused_without_calling_llm(docs_dir, vectorstore):
    a = make_assistant(vectorstore, docs_dir, ExplodingLLM(responses=[]), min_relevance=0.3)
    ans = a.ask("xylophone zébré quantique")
    assert ans.answer == NO_ANSWER and not ans.answered and not ans.llm_called and ans.sources == []


def test_llm_refusal_is_detected(docs_dir, vectorstore):
    a = make_assistant(vectorstore, docs_dir, FakeListChatModel(responses=[NO_ANSWER]), min_relevance=0.0)
    ans = a.ask("Combien coûte mon assurance auto ?")
    assert not ans.answered and ans.sources == []


# --- API ---------------------------------------------------------------------------

def test_api_ask_and_ingest(docs_dir, vectorstore, monkeypatch):
    monkeypatch.setattr(api.config, "DOCS_DIR", docs_dir)
    llm = FakeListChatModel(responses=["Le loyer est de 1 450 $ par mois [1]."])
    api.app.dependency_overrides[api.get_vectorstore] = lambda: vectorstore
    api.app.dependency_overrides[api.get_assistant] = lambda: Assistant(vectorstore, llm, min_relevance=0.0)
    try:
        client = TestClient(api.app)
        assert client.get("/health").json()["status"] == "ok"
        assert len(client.post("/ingest").json()["added"]) == 6
        r = client.post("/ask", json={"question": "Quel est le montant de mon loyer ?"})
        assert r.status_code == 200 and r.json()["answered"] is True and r.json()["sources"]
        assert client.post("/ask", json={"question": "?"}).status_code == 422  # validation de l'entrée
    finally:
        api.app.dependency_overrides.clear()
