"""Collapsed Gibbs sampling for LDA (Griffiths & Steyvers, 2004).

The sampling-based counterpart of the variational model. Every token carries
a topic assignment z; one sweep resamples each z from its full conditional
with the document-topic and topic-word distributions integrated out:

    p(z_i = k | z_-i, w)  ∝  (n_dk + α) · (n_kw + η) / (n_k + Vη)

where the counts exclude token i itself. After ``burn_in`` sweeps the chain is
treated as sampling from the posterior, and the point estimates

    φ_kw = (n_kw + η) / (n_k + Vη)        θ_dk = (n_dk + α) / (N_d + Kα)

are averaged over every ``thin``-th sweep. Averaging over samples, rather than
reading the last one, is what makes the estimate a posterior mean.

The inner loop is plain Python over Python lists: with K around 8, list
arithmetic is faster than NumPy, whose per-call overhead dominates on
eight-element arrays. One sweep over ~45k tokens takes well under a second.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GibbsResult:
    phi: np.ndarray                    # K x V topic-word
    theta: np.ndarray                  # D x K document-topic
    z: list[list[int]]                 # final topic of every token
    trace: list[tuple[int, float]] = field(default_factory=list)  # (sweep, log-likelihood)


def log_likelihood(nkw, nk, ndk, nd, alpha, eta, V, K) -> float:
    """log p(w, z) up to a constant - rises and flattens as the chain mixes."""
    lg = math.lgamma
    ll = 0.0
    for k in range(K):
        ll += lg(V * eta) - V * lg(eta) - lg(nk[k] + V * eta)
        ll += sum(lg(c + eta) for c in nkw[k] if c)  + (V - sum(1 for c in nkw[k] if c)) * lg(eta)
    for d, counts in enumerate(ndk):
        ll += lg(K * alpha) - K * lg(alpha) - lg(nd[d] + K * alpha)
        ll += sum(lg(c + alpha) for c in counts)
    return ll


def gibbs_lda(docs: list[list[int]], V: int, K: int, alpha: float, eta: float,
              iters: int, burn_in: int, thin: int, seed: int = 42,
              trace_every: int = 10, progress=None) -> GibbsResult:
    rng = random.Random(seed)
    D = len(docs)
    nkw = [[0] * V for _ in range(K)]
    nk = [0] * K
    ndk = [[0] * K for _ in range(D)]
    nd = [len(doc) for doc in docs]
    z = []
    for d, doc in enumerate(docs):                    # random initialisation
        zd = []
        for w in doc:
            k = rng.randrange(K)
            zd.append(k)
            nkw[k][w] += 1
            nk[k] += 1
            ndk[d][k] += 1
        z.append(zd)

    v_eta = V * eta
    phi_sum = np.zeros((K, V))
    theta_sum = np.zeros((D, K))
    samples = 0
    trace = [(0, log_likelihood(nkw, nk, ndk, nd, alpha, eta, V, K))]
    krange = range(K)

    for it in range(1, iters + 1):
        for d, doc in enumerate(docs):
            zd, ndk_d = z[d], ndk[d]
            for i, w in enumerate(doc):
                k = zd[i]
                nkw[k][w] -= 1
                nk[k] -= 1
                ndk_d[k] -= 1
                # unnormalised full conditional, accumulated as a CDF
                total = 0.0
                cdf = []
                for t in krange:
                    total += (ndk_d[t] + alpha) * (nkw[t][w] + eta) / (nk[t] + v_eta)
                    cdf.append(total)
                u = rng.random() * total
                k = 0
                while cdf[k] < u:
                    k += 1
                zd[i] = k
                nkw[k][w] += 1
                nk[k] += 1
                ndk_d[k] += 1
        if it % trace_every == 0:
            trace.append((it, log_likelihood(nkw, nk, ndk, nd, alpha, eta, V, K)))
        if it > burn_in and (it - burn_in) % thin == 0:
            nkw_a = np.asarray(nkw, dtype=float)
            phi_sum += (nkw_a + eta) / (nkw_a.sum(axis=1, keepdims=True) + v_eta)
            ndk_a = np.asarray(ndk, dtype=float)
            theta_sum += (ndk_a + alpha) / (ndk_a.sum(axis=1, keepdims=True) + K * alpha)
            samples += 1
        if progress:
            progress(it, iters)

    return GibbsResult(phi_sum / samples, theta_sum / samples, z, trace)
