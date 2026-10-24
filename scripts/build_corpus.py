"""(Re)build the frozen corpus from NLTK's copy of Reuters-21578.

    python scripts/build_corpus.py

You do NOT need to run this - data/corpus/reuters_subset.jsonl is committed.
It is here so the sampling is reproducible and inspectable.
Requires: pip install nltk  and  python -m nltk.downloader reuters
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lda_topics.corpus import build_corpus, save_corpus  # noqa: E402

docs = build_corpus()
save_corpus(docs)
print(f"{len(docs)} documents written")
for cat, n in sorted(Counter(d.category for d in docs).items()):
    print(f"  {cat:<10} {n}")
