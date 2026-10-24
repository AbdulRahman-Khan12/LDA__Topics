"""Document-term representation.

LDA is a bag-of-words model, so a document is reduced to word counts. The
vocabulary is pruned on document frequency:

* ``MIN_DF``: a word seen in fewer than 5 documents cannot characterise a
  topic, and rare words make the topic-word distributions noisy;
* ``MAX_DF``: a word in more than 40% of documents (``price`` in a commodities
  corpus) separates nothing.

The matrix is built with scikit-learn's ``CountVectorizer`` on the already
tokenised lemmas (``analyzer=list``), which returns a sparse
documents x vocabulary count matrix and the vocabulary in column order.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from .config import MAX_DF, MIN_DF


@dataclass
class DTM:
    matrix: sparse.csr_matrix      # documents x vocabulary counts
    vocab: list[str]
    doc_ids: list[str]
    categories: list[str]

    @property
    def n_tokens(self) -> int:
        return int(self.matrix.sum())

    def doc_word_ids(self) -> list[list[int]]:
        """Each document as a list of word ids, one entry per token (for Gibbs)."""
        out = []
        for row in self.matrix:
            ids = np.repeat(row.indices, row.data)
            out.append(ids.tolist())
        return out


def build_dtm(rows: list[dict], min_df: int = MIN_DF, max_df: float = MAX_DF) -> DTM:
    vectorizer = CountVectorizer(analyzer=list, min_df=min_df, max_df=max_df)
    matrix = vectorizer.fit_transform([r["tokens"] for r in rows]).tocsr()
    matrix.sort_indices()
    return DTM(matrix, vectorizer.get_feature_names_out().tolist(),
               [r["doc_id"] for r in rows], [r["category"] for r in rows])
