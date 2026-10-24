"""The two inference methods behind one interface.

* :func:`fit_vb` - **variational Bayes** (scikit-learn's batch
  ``LatentDirichletAllocation``, Blei, Ng & Jordan 2003; Hoffman et al. 2010).
  It replaces the true posterior with a factorised approximation and optimises
  it deterministically. Fast, and the same seed gives the same answer.
* :func:`fit_gibbs` - **collapsed Gibbs sampling** (:mod:`lda_topics.gibbs`).
  It draws samples from the true posterior; asymptotically exact, but
  stochastic and slower.

Both return a :class:`TopicModel` holding φ (topic-word, K x V) and θ
(document-topic, D x K), so everything downstream - metrics, interpretation,
the app - is written once and works on either.

:func:`fold_in` estimates θ for documents the model never saw, with φ held
fixed, by expectation-maximisation. It is used for the held-out perplexity of
*both* models, so their perplexities are computed by the same estimator and
are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from sklearn.decomposition import LatentDirichletAllocation

from .config import ALPHA, ETA, GIBBS_BURN_IN, GIBBS_ITERS, GIBBS_THIN, SEED, VB_MAX_ITER
from .gibbs import gibbs_lda


@dataclass
class TopicModel:
    name: str
    phi: np.ndarray                 # K x V, rows sum to 1
    theta: np.ndarray               # D_train x K, rows sum to 1
    extra: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return self.phi.shape[0]


def fit_vb(X: sparse.csr_matrix, k: int, alpha: float = ALPHA, eta: float = ETA,
           max_iter: int = VB_MAX_ITER, seed: int = SEED) -> TopicModel:
    lda = LatentDirichletAllocation(
        n_components=k, doc_topic_prior=alpha, topic_word_prior=eta,
        learning_method="batch", max_iter=max_iter, random_state=seed)
    theta = lda.fit_transform(X)
    phi = lda.components_ / lda.components_.sum(axis=1, keepdims=True)
    return TopicModel("variational", phi, theta, {"n_iter": lda.n_iter_,
                                                  "bound": float(lda.bound_)})


def fit_gibbs(X: sparse.csr_matrix, k: int, alpha: float = ALPHA, eta: float = ETA,
              iters: int = GIBBS_ITERS, burn_in: int = GIBBS_BURN_IN,
              thin: int = GIBBS_THIN, seed: int = SEED, progress=None) -> TopicModel:
    # Token order changes the chain. Some SciPy operations sort a CSR matrix's
    # indices in place, so sort explicitly to make the run reproducible no
    # matter what touched X before.
    X = X.copy()
    X.sort_indices()
    docs = [np.repeat(row.indices, row.data).tolist() for row in X]
    res = gibbs_lda(docs, X.shape[1], k, alpha, eta, iters, burn_in, thin, seed,
                    progress=progress)
    return TopicModel("gibbs", res.phi, res.theta, {"trace": res.trace, "z": res.z,
                                                    "docs": docs})


def fold_in(phi: np.ndarray, X: sparse.csr_matrix, alpha: float = ALPHA,
            iters: int = 60) -> np.ndarray:
    """θ for unseen documents with φ fixed (EM over the document's tokens)."""
    K = phi.shape[0]
    theta = np.full((X.shape[0], K), 1.0 / K)
    for d in range(X.shape[0]):
        row = X.getrow(d)
        if row.nnz == 0:
            continue
        p = phi[:, row.indices]                       # K x n_unique
        counts = row.data.astype(float)
        t = theta[d]
        for _ in range(iters):
            resp = t[:, None] * p
            resp /= resp.sum(axis=0, keepdims=True)
            t = alpha + (resp * counts).sum(axis=1)
            t /= t.sum()
        theta[d] = t
    return theta


def heldout_perplexity(phi: np.ndarray, X: sparse.csr_matrix, seed: int = SEED) -> float:
    """Document-completion perplexity (Wallach et al., 2009).

    Each held-out document's tokens are split in half at random: θ is
    estimated from the first half, and the second half is scored. Scoring the
    same tokens used to estimate θ would flatter the model.
    """
    rng = np.random.default_rng(seed)
    first, second = [], []
    for row in X:
        tokens = np.repeat(row.indices, row.data)
        rng.shuffle(tokens)
        half = len(tokens) // 2
        first.append(np.bincount(tokens[:half], minlength=X.shape[1]))
        second.append(np.bincount(tokens[half:], minlength=X.shape[1]))
    A = sparse.csr_matrix(np.vstack(first))
    B = sparse.csr_matrix(np.vstack(second))
    theta = fold_in(phi, A)
    log_lik, n = 0.0, 0
    for d in range(B.shape[0]):
        row = B.getrow(d)
        if row.nnz == 0:
            continue
        probs = theta[d] @ phi[:, row.indices]
        log_lik += float((row.data * np.log(probs)).sum())
        n += int(row.data.sum())
    return float(np.exp(-log_lik / n))
