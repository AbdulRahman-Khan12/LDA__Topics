"""Turning φ and θ into something a person can read.

* :func:`relevance` re-ranks a topic's words (Sievert & Shirley 2014):

      relevance(w | k) = λ · log φ_kw + (1 - λ) · log(φ_kw / p(w))

  λ = 1 is plain probability, which favours words that are frequent
  everywhere; lowering λ promotes words that are *specific* to the topic.
* :func:`intertopic_map` places topics in 2-D by classical multidimensional
  scaling of their Jensen-Shannon distances - the same projection LDAvis uses.
* :func:`representative_docs` lists the documents with the highest share of a
  topic: reading three of them is the fastest way to name a topic.
"""

from __future__ import annotations

import numpy as np

from .metrics import js_matrix


def relevance(phi: np.ndarray, p_w: np.ndarray, lam: float) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return lam * np.log(phi) + (1 - lam) * np.log(phi / p_w[None, :])


def top_words(phi: np.ndarray, vocab: list[str], k: int, n: int = 10,
              p_w: np.ndarray | None = None, lam: float = 1.0) -> list[tuple[str, float]]:
    score = phi[k] if p_w is None or lam >= 1.0 else relevance(phi, p_w, lam)[k]
    order = np.argsort(-score)[:n]
    return [(vocab[i], float(phi[k, i])) for i in order]


def intertopic_map(phi: np.ndarray) -> np.ndarray:
    """K x 2 coordinates from classical MDS (PCoA) on JS distances."""
    d = js_matrix(phi, phi)
    k = d.shape[0]
    j = np.eye(k) - np.ones((k, k)) / k
    b = -0.5 * j @ (d ** 2) @ j
    vals, vecs = np.linalg.eigh(b)
    order = np.argsort(vals)[::-1][:2]
    return vecs[:, order] * np.sqrt(np.maximum(vals[order], 0))


def representative_docs(theta: np.ndarray, k: int, n: int = 3) -> list[int]:
    return np.argsort(-theta[:, k])[:n].tolist()
