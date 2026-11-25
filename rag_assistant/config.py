"""Configuration (variables d'environnement, avec des valeurs par défaut raisonnables)."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = Path(os.getenv("DOCS_DIR", ROOT / "data" / "docs"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ROOT / "chroma_db"))
COLLECTION = os.getenv("COLLECTION", "mes_papiers")

# --- Ollama (LLM open source exécuté en local) ------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2")             # ~3 milliards de paramètres, tourne sur un portable
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")  # modèle de plongements (embeddings)
LLM_NUM_CTX = int(os.getenv("LLM_NUM_CTX", "8192"))
# Préfixes de tâche : nomic-embed-text a été entraîné à distinguer une question (« search_query: »)
# d'un passage à retrouver (« search_document: »). Ils servent seulement au calcul des vecteurs : le texte
# stocké et cité reste intact. Changer de modèle ou de préfixes impose de réindexer (supprimer chroma_db).
_NOMIC = EMBED_MODEL.startswith("nomic-embed-text")
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "search_query:" if _NOMIC else "")
EMBED_DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", "search_document:" if _NOMIC else "")

# --- Découpage et recherche -------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))        # caractères par morceau
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))  # chevauchement pour ne pas couper une idée en deux
TOP_K = int(os.getenv("TOP_K", "4"))                    # nombre de morceaux envoyés au LLM
# Score de pertinence minimal (similarité cosinus, 0 à 1). Sous ce seuil, le morceau est ignoré ;
# si aucun morceau ne passe, l'assistant répond qu'il ne sait pas, sans appeler le LLM.
# Mesuré avec nomic-embed-text (README, « Réglages à calibrer ») : scores tassés entre 0,59 et 0,77,
# et aucun seuil ne sépare les questions couvertes (0,634 au plus bas) des questions hors documents
# (jusqu'à 0,723). 0,60 est donc un plancher : il n'écarte aucune question couverte mesurée et
# n'arrête que l'hors-sujet total. Le vrai garde-fou est la consigne du prompt.
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.60"))
