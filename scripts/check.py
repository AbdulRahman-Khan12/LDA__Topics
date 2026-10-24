"""Self-check for LDA__Topics.

    python scripts/check.py

Verifies the committed artefacts and the core maths on a tiny synthetic
corpus whose answer is known, so a passing run means the samplers and metrics
are right, not merely that they run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lda_topics.config import ARTIFACT_DIR, K  # noqa: E402
from lda_topics.corpus import load_corpus  # noqa: E402
from lda_topics.metrics import alignment, coherence, diversity, match_topics  # noqa: E402
from lda_topics.models import fit_gibbs, fit_vb, fold_in  # noqa: E402

RULE = "-" * 64
failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


print(RULE + "\n1. CORPUS AND ARTEFACTS\n" + RULE)
docs = load_corpus()
check("849 documents", len(docs) == 849, str(len(docs)))
check("8 categories", len({d.category for d in docs}) == 8)
for name in ("topics.csv", "doc_topics.csv", "metrics.json", "k_sweep.csv", "gibbs_trace.csv",
             "topic_matching.csv", "phi_variational.npy", "phi_gibbs.npy", "vocab.txt", "p_w.npy",
             "gibbs_assignments.jsonl"):
    check(f"artefact {name}", (ARTIFACT_DIR / name).exists())
for name in ("variational", "gibbs"):
    phi = np.load(ARTIFACT_DIR / f"phi_{name}.npy")
    check(f"{name}: phi is K x V and rows sum to 1",
          phi.shape[0] == K and np.allclose(phi.sum(axis=1), 1, atol=1e-3))
metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
check("both models beat chance alignment (NMI > 0.3)",
      all(metrics["models"][m]["nmi"] > 0.3 for m in ("variational", "gibbs")))

print(RULE + "\n2. SYNTHETIC CORPUS WITH KNOWN TOPICS\n" + RULE)
# Three disjoint 5-word topics; every document draws from exactly one.
rng = np.random.default_rng(0)
V, true_k, n_docs = 15, 3, 90
rows = []
for d in range(n_docs):
    k = d % true_k
    words = rng.integers(k * 5, k * 5 + 5, size=40)
    rows.append(np.bincount(words, minlength=V))
X = sparse.csr_matrix(np.vstack(rows))
labels = [str(d % true_k) for d in range(n_docs)]
true_phi = np.zeros((true_k, V))
for k in range(true_k):
    true_phi[k, k * 5:k * 5 + 5] = 0.2

vb = fit_vb(X, true_k, max_iter=50)
gb = fit_gibbs(X, true_k, iters=60, burn_in=30, thin=5)
for name, m in (("variational", vb), ("gibbs", gb)):
    js = np.mean([d for _, _, d in match_topics(m.phi, true_phi)])
    check(f"{name} recovers the true topics", js < 0.2, f"mean JS {js:.3f}")
    check(f"{name} recovers the document labels", alignment(m.theta, labels)["nmi"] > 0.95)
check("fold-in puts a pure document on one topic",
      fold_in(true_phi, sparse.csr_matrix(rows[0])).max() > 0.95)
check("perfect topics have diversity 1.0", diversity(true_phi, n=5) == 1.0)
check("true topics are coherent (NPMI > 0)", coherence(true_phi, X, n=5)["npmi_mean"] > 0)
check("Gibbs log-likelihood rises", gb.extra["trace"][-1][1] > gb.extra["trace"][0][1])

print(RULE)
print("ALL CHECKS PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
sys.exit(1 if failures else 0)
