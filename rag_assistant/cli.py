"""Ligne de commande.

  python -m rag_assistant ingest                 # indexe data/docs (idempotent)
  python -m rag_assistant ask "Mon assurance couvre-t-elle un dégât d'eau ?"
  python -m rag_assistant eval                   # mesure la qualité sur eval/questions.json
"""

from __future__ import annotations

import argparse
import sys

from . import config
from .core import Assistant, Indexer, build_llm, build_vectorstore
from .evaluate import run_eval, save


def cmd_ingest(args) -> int:
    r = Indexer(build_vectorstore()).ingest_dir(config.DOCS_DIR)
    print(f"Ajoutés : {r.added or '-'}\nMis à jour : {r.updated or '-'}\nInchangés : {r.unchanged or '-'}\n"
          f"Retirés : {r.removed or '-'}\nMorceaux écrits : {r.chunks_written}")
    return 0


def cmd_ask(args) -> int:
    a = Assistant(build_vectorstore(), build_llm()).ask(args.question)
    print(f"\n{a.answer}\n")
    for s in a.sources:
        where = s.source + (f" › {s.section}" if s.section else "")
        print(f"  [{s.ref}] {where}  (pertinence {s.score:.2f})")
    print(f"\n  {a.latency_ms} ms · {a.model} · {a.prompt_version}" + ("" if a.llm_called else " · LLM non appelé"))
    return 0


def cmd_eval(args) -> int:
    report = run_eval(Assistant(build_vectorstore(), build_llm()))
    m = report["metrics"]
    print(f"Modèle {report['llm']} · plongements {report['embeddings']} · prompt {report['prompt_version']}\n")
    print(f"  Recherche : bon document retrouvé     {m['retrieval_hit_rate']:.0%}")
    print(f"  Réponse : mots-clés attendus présents {m['keyword_recall']:.0%}")
    print(f"  A répondu aux questions couvertes     {m['answered_when_answerable']:.0%}")
    print(f"  A refusé les questions hors documents {m['refused_when_unanswerable']:.0%}")
    print(f"  Latence médiane                       {m['median_latency_ms']} ms\n")
    for q in report["questions"]:
        bad = (q["answerable"] and (not q["retrieval_hit"] or q["keyword_recall"] < 1 or not q["answered"])) or \
              (not q["answerable"] and q["answered"])
        if bad:
            print(f"  ✗ {q['question']}\n      → {q['answer'][:160]}")
    print(f"\nRapport : {save(report)}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="rag_assistant", description="Mes papiers : assistant RAG privé pour vos documents personnels")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest").set_defaults(func=cmd_ingest)
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.set_defaults(func=cmd_ask)
    sub.add_parser("eval").set_defaults(func=cmd_eval)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
