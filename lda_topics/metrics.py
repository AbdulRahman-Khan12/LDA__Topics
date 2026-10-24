"""How good are the topics? Five complementary measures.

* **NPMI coherence** (Bouma 2009; Röder et al. 2015) - do a topic's top words
  actually co-occur in documents? Range -1..1, higher is better. It is the
  intrinsic measure that agrees best with human judgements of topic quality.
* **UMass coherence** (Mimno et al. 2011) - the same idea with conditional
  log probabilities; always <= 0, closer to 0 is better.
* **Held-out perplexity** - how surprised the model is by unseen text
  (see :func:`lda_topics.models.heldout_perplexity`). Lower is better, but it
  is known to correlate poorly with human interpretability, which is why it
  is never used alone.
* **Topic diversity** (Dieng et al. 2020) - share of unique words among all
  topics' top words. 1.0 means no two topics share a top word.
* **Alignment with the gold categories** - NMI and purity between each
  document's dominant topic and its Reuters category. The categories are
  never shown to the model, so this measures whether the unsupervised
  structure matches the human one.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
from scipy import sparse
from scipy.spatial.distance import jensenshannon
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score

from .config import TOP_N


def top_word_ids(phi: np.ndarray, n: int = TOP_N) -> np.ndarray:
    return np.argsort(-phi, axis=1)[:, :n]


def coherence(phi: np.ndarray, X: sparse.csr_matrix, n: int = TOP_N) -> dict:
    """Per-topic NPMI and UMass from document co-occurrence in ``X``."""
    B = (X > 0).astype(np.int32).tocsc()
    D = B.shape[0]
    df = np.asarray(B.sum(axis=0)).ravel()
    npmi, umass = [], []
    for top in top_word_ids(phi, n):
        cols = B[:, top]
        co = (cols.T @ cols).toarray()            # n x n co-document counts
        n_scores, u_scores = [], []
        for i, j in itertools.combinations(range(len(top)), 2):
            p_i, p_j, p_ij = df[top[i]] / D, df[top[j]] / D, co[i, j] / D
            if p_ij == 0:
                n_scores.append(-1.0)
            else:
                pmi = math.log(p_ij / (p_i * p_j))
                n_scores.append(pmi / -math.log(p_ij))
            # UMass: ordered pairs, more frequent (higher-ranked) word conditions
            u_scores.append(math.log((co[i, j] + 1) / df[top[i]]))
        npmi.append(float(np.mean(n_scores)))
        umass.append(float(np.mean(u_scores)))
    return {"npmi": npmi, "umass": umass,
            "npmi_mean": float(np.mean(npmi)), "umass_mean": float(np.mean(umass))}


def diversity(phi: np.ndarray, n: int = TOP_N) -> float:
    top = top_word_ids(phi, n)
    return len(set(top.ravel().tolist())) / top.size


def alignment(theta: np.ndarray, categories: list[str]) -> dict:
    dominant = theta.argmax(axis=1)
    cats = sorted(set(categories))
    table = np.zeros((theta.shape[1], len(cats)), dtype=int)
    for k, c in zip(dominant, categories):
        table[k, cats.index(c)] += 1
    purity = table.max(axis=1).sum() / len(categories)
    return {"nmi": float(normalized_mutual_info_score(categories, dominant)),
            "purity": float(purity), "table": table, "categories": cats,
            "dominant": dominant}


def js_matrix(phi_a: np.ndarray, phi_b: np.ndarray) -> np.ndarray:
    """Jensen-Shannon distance between every topic of A and every topic of B (base 2, 0..1)."""
    return np.array([[jensenshannon(a, b, base=2) for b in phi_b] for a in phi_a])


def match_topics(phi_a: np.ndarray, phi_b: np.ndarray) -> list[tuple[int, int, float]]:
    """One-to-one matching of A's topics to B's (Hungarian algorithm on JS distance)."""
    dist = js_matrix(phi_a, phi_b)
    rows, cols = linear_sum_assignment(dist)
    return [(int(r), int(c), float(dist[r, c])) for r, c in zip(rows, cols)]
