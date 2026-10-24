"""Run the whole experiment and write every artefact the app and report use.

    python scripts/run_pipeline.py            # everything (~3 minutes on a laptop)
    python scripts/run_pipeline.py --skip-sweep   # reuse the committed K sweep

Stages
  1. preprocess   data/corpus/*.jsonl  -> data/processed/tokens.jsonl
  2. DTM          vocabulary pruning, train / held-out split
  3. K sweep      variational LDA for K in K_SWEEP x 3 seeds -> k_sweep.csv
  4. models       variational LDA and collapsed Gibbs at the chosen K
  5. evaluate     coherence, perplexity, diversity, gold alignment, VB<->Gibbs matching
  6. export       topic tables, document mixtures, Gibbs trace and token assignments

All artefacts are committed, so the app starts instantly on a fresh clone.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lda_topics import config as C  # noqa: E402
from lda_topics.corpus import load_corpus  # noqa: E402
from lda_topics.dtm import build_dtm  # noqa: E402
from lda_topics.interpret import intertopic_map, representative_docs, top_words  # noqa: E402
from lda_topics.metrics import (alignment, coherence, diversity, match_topics,  # noqa: E402
                                top_word_ids)
from lda_topics.models import fit_gibbs, fit_vb, fold_in, heldout_perplexity  # noqa: E402
from lda_topics.preprocess import load_tokens, preprocess_corpus, save_tokens  # noqa: E402

A = C.ARTIFACT_DIR


def write_csv(name: str, rows: list[dict]) -> None:
    with (A / name).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def step(msg: str) -> float:
    print(f"\n== {msg}")
    return time.perf_counter()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sweep", action="store_true")
    ap.add_argument("--skip-preprocess", action="store_true")
    args = ap.parse_args()
    A.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()

    # 1 ------------------------------------------------------------------------
    t = step("1. preprocessing")
    docs = load_corpus()
    if args.skip_preprocess and C.TOKENS_FILE.exists():
        rows = load_tokens()
    else:
        rows = preprocess_corpus(docs)
        save_tokens(rows)
    raw_tokens = sum(r["n_raw_tokens"] for r in rows)
    kept_tokens = sum(len(r["tokens"]) for r in rows)
    print(f"   {len(rows)} documents, {raw_tokens:,} spaCy tokens -> {kept_tokens:,} content lemmas "
          f"({time.perf_counter() - t:.1f}s)")

    # 2 ------------------------------------------------------------------------
    step("2. document-term matrix")
    dtm = build_dtm(rows)
    X = dtm.matrix
    rng = np.random.default_rng(C.SEED)
    order = rng.permutation(X.shape[0])
    n_ho = int(round(C.HOLDOUT_FRACTION * X.shape[0]))
    ho_idx, tr_idx = np.sort(order[:n_ho]), np.sort(order[n_ho:])
    X_tr, X_ho = X[tr_idx], X[ho_idx]
    cats_tr = [dtm.categories[i] for i in tr_idx]
    lengths = np.asarray(X.sum(axis=1)).ravel()
    print(f"   vocabulary {len(dtm.vocab):,} words (min_df={C.MIN_DF}, max_df={C.MAX_DF}); "
          f"{dtm.n_tokens:,} tokens; train {len(tr_idx)} / held-out {len(ho_idx)} documents")

    # 3 ------------------------------------------------------------------------
    sweep_file = A / "k_sweep.csv"
    if args.skip_sweep and sweep_file.exists():
        step("3. K sweep (reusing k_sweep.csv)")
        with sweep_file.open() as fh:
            sweep = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    else:
        t = step("3. K sweep - variational LDA, 3 seeds per K")
        sweep = []
        for k in C.K_SWEEP:
            runs = []
            for seed in (1, 2, 3):
                m = fit_vb(X_tr, k, seed=seed, max_iter=50)
                coh = coherence(m.phi, X_tr)
                runs.append([coh["npmi_mean"], coh["umass_mean"],
                             heldout_perplexity(m.phi, X_ho), diversity(m.phi),
                             alignment(m.theta, cats_tr)["nmi"]])
            r = np.array(runs)
            sweep.append({"k": k, "npmi": r[:, 0].mean(), "npmi_sd": r[:, 0].std(),
                          "umass": r[:, 1].mean(), "perplexity": r[:, 2].mean(),
                          "diversity": r[:, 3].mean(), "nmi": r[:, 4].mean()})
            print(f"   K={k:<3} NPMI {sweep[-1]['npmi']:.3f} ± {sweep[-1]['npmi_sd']:.3f}   "
                  f"perplexity {sweep[-1]['perplexity']:.0f}   NMI {sweep[-1]['nmi']:.3f}")
        write_csv("k_sweep.csv", [{k: round(v, 4) for k, v in s.items()} for s in sweep])
        print(f"   ({time.perf_counter() - t:.0f}s)")
    # one-standard-error rule: the smallest K whose coherence is within one SD of the best
    best = max(sweep, key=lambda s: s["npmi"])
    chosen = min(int(s["k"]) for s in sweep if s["npmi"] >= best["npmi"] - best["npmi_sd"])
    print(f"   best mean NPMI at K={int(best['k'])}; smallest K within one SD of it: K={chosen}")
    print(f"   coherence is flat across that range, so it cannot pick K on its own; the main")
    print(f"   models use K={C.K}, the number of sampled categories, so recovery can be tested")
    K = C.K

    # 4 ------------------------------------------------------------------------
    t = step(f"4. fitting K={K}: variational Bayes")
    vb = fit_vb(X_tr, K)
    print(f"   {vb.extra['n_iter']} iterations ({time.perf_counter() - t:.1f}s)")
    t = step(f"   collapsed Gibbs: {C.GIBBS_ITERS} sweeps, burn-in {C.GIBBS_BURN_IN}, thin {C.GIBBS_THIN}")
    gibbs = fit_gibbs(X_tr, K)
    print(f"   done ({time.perf_counter() - t:.1f}s)")

    # 5 ------------------------------------------------------------------------
    step("5. evaluation")
    p_w = np.asarray(X_tr.sum(axis=0)).ravel() / X_tr.sum()
    metrics, models = {}, {"variational": vb, "gibbs": gibbs}
    labels = json.loads(C.LABELS_FILE.read_text(encoding="utf-8")) if C.LABELS_FILE.exists() else {}
    topic_rows, doc_rows = [], []
    for name, m in models.items():
        coh = coherence(m.phi, X_tr)
        theta_all = np.zeros((X.shape[0], K))
        theta_all[tr_idx] = m.theta
        theta_all[ho_idx] = fold_in(m.phi, X_ho)
        align = alignment(theta_all, dtm.categories)
        metrics[name] = {"npmi": round(coh["npmi_mean"], 4), "umass": round(coh["umass_mean"], 4),
                         "perplexity": round(heldout_perplexity(m.phi, X_ho), 1),
                         "diversity": round(diversity(m.phi), 3), "nmi": round(align["nmi"], 4),
                         "purity": round(align["purity"], 4)}
        print(f"   {name:<12} " + "  ".join(f"{k} {v}" for k, v in metrics[name].items()))
        np.save(A / f"phi_{name}.npy", m.phi.astype(np.float32))
        coords = intertopic_map(m.phi)
        prevalence = theta_all.mean(axis=0)
        for k in range(K):
            col = align["table"][k]
            dominant_cat = align["categories"][int(col.argmax())] if col.sum() else ""
            topic_rows.append({
                "model": name, "topic": k,
                "label": labels.get(name, {}).get(str(k), ""),
                "top_words": " ".join(w for w, _ in top_words(m.phi, dtm.vocab, k, C.TOP_N)),
                "relevant_words": " ".join(w for w, _ in top_words(
                    m.phi, dtm.vocab, k, C.TOP_N, p_w, C.RELEVANCE_LAMBDA)),
                "prevalence": round(float(prevalence[k]), 4),
                "npmi": round(coh["npmi"][k], 4), "umass": round(coh["umass"][k], 4),
                "n_docs_dominant": int(col.sum()),
                "dominant_category": dominant_cat,
                "category_share": round(float(col.max() / col.sum()), 3) if col.sum() else 0.0,
                "x": round(float(coords[k, 0]), 4), "y": round(float(coords[k, 1]), 4),
                "representative_docs": " ".join(dtm.doc_ids[i] for i in representative_docs(theta_all, k)),
            })
            metrics[name].setdefault("contingency", align["table"].tolist())
        metrics[name]["categories"] = align["categories"]
        for i, doc_id in enumerate(dtm.doc_ids):
            doc_rows.append({"model": name, "doc_id": doc_id, "category": dtm.categories[i],
                             "split": "heldout" if i in set(ho_idx.tolist()) else "train",
                             "dominant": int(theta_all[i].argmax()),
                             **{f"t{k}": round(float(theta_all[i, k]), 4) for k in range(K)}})

    matches = match_topics(vb.phi, gibbs.phi)
    top_vb, top_gb = top_word_ids(vb.phi), top_word_ids(gibbs.phi)
    match_rows = []
    for a, b, dist in matches:
        sa, sb = set(top_vb[a].tolist()), set(top_gb[b].tolist())
        match_rows.append({"vb_topic": a, "gibbs_topic": b, "js_distance": round(dist, 4),
                           "top10_overlap": len(sa & sb),
                           "shared_words": " ".join(dtm.vocab[i] for i in top_vb[a] if i in sb)})
    metrics["agreement"] = {"mean_js": round(float(np.mean([m[2] for m in matches])), 4),
                            "mean_top10_overlap": round(float(np.mean([r["top10_overlap"] for r in match_rows])), 2)}
    print(f"   VB vs Gibbs: mean JS distance {metrics['agreement']['mean_js']}, "
          f"mean top-10 overlap {metrics['agreement']['mean_top10_overlap']}/10")

    # 6 ------------------------------------------------------------------------
    step("6. export")
    (A / "vocab.txt").write_text("\n".join(dtm.vocab), encoding="utf-8")
    np.save(A / "p_w.npy", p_w.astype(np.float32))
    write_csv("topics.csv", topic_rows)
    write_csv("doc_topics.csv", doc_rows)
    write_csv("topic_matching.csv", match_rows)
    write_csv("gibbs_trace.csv", [{"sweep": s, "log_likelihood": round(ll, 1)}
                                  for s, ll in gibbs.extra["trace"]])
    with (A / "gibbs_assignments.jsonl").open("w", encoding="utf-8") as fh:
        for j, i in enumerate(tr_idx):
            fh.write(json.dumps({"doc_id": dtm.doc_ids[i],
                                 "words": [dtm.vocab[w] for w in gibbs.extra["docs"][j]],
                                 "z": gibbs.extra["z"][j]}) + "\n")
    summary = {
        "corpus": {"documents": len(rows), "categories": len(set(dtm.categories)),
                   "raw_tokens": raw_tokens, "content_lemmas": kept_tokens,
                   "vocabulary": len(dtm.vocab), "dtm_tokens": dtm.n_tokens,
                   "mean_doc_length": round(float(lengths.mean()), 1),
                   "train_docs": len(tr_idx), "heldout_docs": len(ho_idx),
                   "sparsity": round(1 - X.nnz / (X.shape[0] * X.shape[1]), 4)},
        "settings": {"K": K, "alpha": C.ALPHA, "eta": C.ETA, "min_df": C.MIN_DF, "max_df": C.MAX_DF,
                     "vb_max_iter": C.VB_MAX_ITER, "gibbs_iters": C.GIBBS_ITERS,
                     "gibbs_burn_in": C.GIBBS_BURN_IN, "gibbs_thin": C.GIBBS_THIN,
                     "k_selected_by_rule": chosen, "k_best_npmi": int(best["k"])},
        "agreement": metrics.pop("agreement"),
        "models": metrics,
    }
    (A / "metrics.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"   artefacts written to {A.relative_to(C.PROJECT_ROOT)}/")
    print(f"\ntotal {time.perf_counter() - t_all:.0f}s")


if __name__ == "__main__":
    main()
