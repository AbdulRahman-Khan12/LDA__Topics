"""Text preprocessing - from raw newswire text to a bag of lemmas.

Steps, in order, for every document:

1. spaCy ``en_core_web_sm`` tokenises, tags and lemmatises the text
   (the parser and NER are disabled - they are slow and unused);
2. keep only content words: nouns, proper nouns, adjectives and verbs;
3. drop stopwords, numbers, punctuation, ticker symbols and tokens shorter
   than three characters;
4. lowercase the lemma and drop newswire boilerplate (``mln``, ``dlrs``,
   ``say`` ...) that appears in every category and so can only blur topics.

Lemmatising matters more for LDA than for most tasks: without it *price*,
*prices* and *priced* are three unrelated words, each with a third of the
counts, and none of them makes a topic's top-10 list.

If spaCy is not installed, :func:`fallback_tokens` gives a regex tokeniser
with the same filters minus lemmatisation, so the app still runs.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache

from .config import DOMAIN_STOPWORDS, KEEP_POS, MIN_TOKEN_LEN, TOKENS_FILE
from .corpus import Document

TICKER = re.compile(r"^<.*>$")
ALPHA = re.compile(r"^[a-z][a-z\-]*[a-z]$")


@lru_cache(maxsize=1)
def load_nlp():
    import spacy

    return spacy.load("en_core_web_sm", disable=["parser", "ner"])


def _keep(lemma: str) -> bool:
    return (len(lemma) >= MIN_TOKEN_LEN and ALPHA.match(lemma) is not None
            and lemma not in DOMAIN_STOPWORDS)


def spacy_tokens(doc) -> list[str]:
    out = []
    for tok in doc:
        if tok.pos_ not in KEEP_POS or tok.is_stop or tok.like_num or TICKER.match(tok.text):
            continue
        lemma = tok.lemma_.lower().strip()
        if _keep(lemma):
            out.append(lemma)
    return out


def fallback_tokens(text: str) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    words = re.findall(r"[a-zA-Z][a-zA-Z\-]+", re.sub(r"<[^>]*>", " ", text))
    return [w.lower() for w in words
            if w.lower() not in ENGLISH_STOP_WORDS and _keep(w.lower())]


def tokenize(text: str) -> list[str]:
    try:
        return spacy_tokens(load_nlp()(text))
    except Exception:          # spaCy or the model missing - degrade, don't crash
        return fallback_tokens(text)


def preprocess_corpus(docs: list[Document]) -> list[dict]:
    nlp = load_nlp()
    texts = [f"{d.title}. {d.text}" for d in docs]
    rows = []
    for d, parsed in zip(docs, nlp.pipe(texts, batch_size=64)):
        rows.append({"doc_id": d.doc_id, "category": d.category,
                     "n_raw_tokens": len(parsed), "tokens": spacy_tokens(parsed)})
    return rows


def save_tokens(rows: list[dict]) -> None:
    TOKENS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with TOKENS_FILE.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_tokens() -> list[dict]:
    with TOKENS_FILE.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
