"""Faux modèles pour tester sans Ollama.

HashEmbeddings : plongements « sac de mots » hachés. Ils ne comprennent rien au sens,
mais deux textes qui partagent des mots sont proches : suffisant pour tester la
recherche, les seuils et l'ingestion de façon déterministe.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
import unicodedata

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from rag_assistant import config
from rag_assistant.core import build_vectorstore

STOP = {"le", "la", "les", "de", "des", "du", "un", "une", "et", "a", "en", "est", "pour", "que", "qui", "quel",
        "quelle", "sur", "au", "aux", "il", "se", "par", "ou", "ne", "pas", "faut", "doit", "on", "d", "l"}


class HashEmbeddings(Embeddings):
    dim = 512

    def _vec(self, text: str) -> list[float]:
        t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z0-9]+", t):
            if w in STOP or len(w) < 2:
                continue
            w = w[:6]  # racine grossière : « kilometrage » ≈ « kilometre »
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class ExplodingLLM(FakeListChatModel):
    """Échoue si on l'appelle : sert à vérifier que le LLM n'est PAS appelé."""

    def _call(self, *args, **kwargs):
        raise AssertionError("Le LLM n'aurait pas dû être appelé")


@pytest.fixture
def docs_dir(tmp_path):
    d = tmp_path / "docs"
    shutil.copytree(config.DOCS_DIR, d)
    return d


@pytest.fixture
def vectorstore(tmp_path):
    return build_vectorstore(embeddings=HashEmbeddings(), persist_dir=tmp_path / "chroma")
