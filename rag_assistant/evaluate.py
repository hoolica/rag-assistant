"""Évaluation du RAG sur un jeu de questions de référence (eval/questions.json).

Trois familles de mesures, parce qu'un RAG peut échouer à trois endroits :
- **Recherche** : le bon document est-il parmi les morceaux retrouvés ? (taux de succès@k)
- **Réponse** : la réponse contient-elle les éléments attendus ? (rappel des mots-clés)
- **Refus** : l'assistant refuse-t-il les questions hors documents, sans refuser les autres ?
"""

from __future__ import annotations

import json
import statistics
import unicodedata
from datetime import datetime
from pathlib import Path

from . import config
from .core import Assistant
from .prompts import PROMPT_VERSION

QUESTIONS = config.ROOT / "eval" / "questions.json"
RESULTS_DIR = config.ROOT / "eval" / "results"


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold().replace(" ", " ").replace("\xa0", " ")
    return " ".join(s.split())


def run_eval(assistant: Assistant, questions_path: Path = QUESTIONS) -> dict:
    items = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    rows = []
    for it in items:
        answerable = it.get("answerable", True)
        retrieved = [s.source for s in assistant.retrieve(it["question"])]
        ans = assistant.ask(it["question"])
        row = {"question": it["question"], "answerable": answerable, "answered": ans.answered,
               "latency_ms": ans.latency_ms, "retrieved": retrieved, "answer": ans.answer}
        if answerable:
            row["retrieval_hit"] = any(src in retrieved for src in it["expected_sources"])
            kws = it.get("expected_keywords", [])
            found = [k for k in kws if _norm(k) in _norm(ans.answer)]
            row["keyword_recall"] = len(found) / len(kws) if kws else 1.0
            row["missing_keywords"] = [k for k in kws if k not in found]
        rows.append(row)

    ans_rows = [r for r in rows if r["answerable"]]
    neg_rows = [r for r in rows if not r["answerable"]]
    pct = lambda xs: round(sum(xs) / len(xs), 3) if xs else None  # noqa: E731
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "llm": config.LLM_MODEL, "embeddings": config.EMBED_MODEL, "prompt_version": PROMPT_VERSION,
        "top_k": assistant.top_k, "min_relevance": assistant.min_relevance,
        "chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP,
        "metrics": {
            "retrieval_hit_rate": pct([r["retrieval_hit"] for r in ans_rows]),
            "keyword_recall": pct([r["keyword_recall"] for r in ans_rows]),
            "answered_when_answerable": pct([r["answered"] for r in ans_rows]),
            "refused_when_unanswerable": pct([not r["answered"] for r in neg_rows]),
            "median_latency_ms": int(statistics.median(r["latency_ms"] for r in rows)) if rows else 0,
        },
        "questions": rows,
    }


def save(report: dict) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}_{report['llm'].replace(':', '-')}_{report['prompt_version']}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
