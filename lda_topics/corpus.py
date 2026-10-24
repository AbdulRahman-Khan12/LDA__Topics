"""The frozen document collection.

``build_corpus`` samples the Reuters-21578 articles once (it needs NLTK and
its Reuters corpus); the result is committed as JSONL so that everybody who
clones the repository works on byte-identical documents. ``load_corpus`` is
all the rest of the project ever calls.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from .config import CATEGORIES, CORPUS_FILE, DOCS_PER_CATEGORY, MIN_WORDS, SEED


@dataclass(frozen=True)
class Document:
    doc_id: str
    category: str      # gold label - never shown to the model
    title: str
    text: str


def _clean(raw: str) -> tuple[str, str]:
    raw = raw.replace("&lt;", "<").replace("&amp;", "&").replace("&gt;", ">")
    lines = [ln.strip() for ln in raw.strip().splitlines()]
    title = lines[0].title() if lines else ""
    body = " ".join(ln for ln in lines[1:] if ln)
    return title, re.sub(r"\s+", " ", body).strip()


def build_corpus() -> list[Document]:
    from nltk.corpus import reuters   # only needed to (re)build the frozen file

    rng = random.Random(SEED)
    docs: list[Document] = []
    for category in CATEGORIES:
        ids = [f for f in reuters.fileids(category)
               if reuters.categories(f) == [category] and len(reuters.words(f)) >= MIN_WORDS]
        for fid in sorted(rng.sample(sorted(ids), min(DOCS_PER_CATEGORY, len(ids)))):
            title, text = _clean(reuters.raw(fid))
            docs.append(Document(fid.replace("/", "-"), category, title, text))
    return docs


def save_corpus(docs: list[Document]) -> None:
    CORPUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_FILE.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")


def load_corpus() -> list[Document]:
    with CORPUS_FILE.open(encoding="utf-8") as fh:
        return [Document(**json.loads(line)) for line in fh if line.strip()]
