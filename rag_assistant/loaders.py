"""Chargement et découpage des documents (Markdown, texte, PDF).

Les fichiers Markdown sont d'abord découpés par section (titres #, ##...) : chaque
morceau garde le titre de sa section dans ses métadonnées, ce qui améliore la
recherche et permet de citer « fichier › section ». Les sections longues sont
ensuite redécoupées avec chevauchement.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config

SUPPORTED = {".md", ".txt", ".pdf"}
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.M)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _markdown_sections(text: str) -> list[tuple[str, str]]:
    """Retourne [(titre de section, texte)] en gardant le titre du document en préfixe."""
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]
    doc_title = matches[0].group(2) if matches[0].group(1) == "#" else ""
    sections = []
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = m.group(2)
        if doc_title and title != doc_title:
            title = f"{doc_title} › {title}"
        sections.append((title, text[m.start():end]))
    return [(t, s) for t, s in sections if s.strip()]


def load_file(path: Path) -> list[Document]:
    path = Path(path)
    meta = {"source": path.name, "file_hash": file_hash(path)}
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [
            Document(page_content=page.extract_text() or "", metadata={**meta, "page": i, "section": f"page {i}"})
            for i, page in enumerate(reader.pages, start=1)
            if (page.extract_text() or "").strip()
        ]
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".md":
        return [Document(page_content=s, metadata={**meta, "section": t}) for t, s in _markdown_sections(text)]
    return [Document(page_content=text, metadata={**meta, "section": ""})]


def split(docs: list[Document], chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk"] = i
    return chunks


def iter_files(directory: Path) -> list[Path]:
    return sorted(p for p in Path(directory).rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
