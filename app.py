"""LDA__Topics - Streamlit interface.

Run with:  streamlit run app.py

The app reads only the committed artefacts in data/artifacts/, so it starts
instantly and never retrains a model. Re-run ``python scripts/run_pipeline.py``
to regenerate them.

A topic model is only as useful as its interpretation, so the interface is
built around the three questions a reader asks of a topic: *what words define
it* (top words and relevance), *where does it sit relative to the others*
(the intertopic map), and *which documents is it about* (representative
documents and per-token assignments).
"""

from __future__ import annotations

import html
import json

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from lda_topics import __version__
from lda_topics.config import ARTIFACT_DIR, CORPUS_FILE, RELEVANCE_LAMBDA, TOKENS_FILE
from lda_topics.interpret import relevance

st.set_page_config(page_title="LDA__Topics", page_icon="◔", layout="wide")

PALETTE = ["#2E5EAA", "#C2410C", "#15803D", "#7E22CE", "#B91C1C", "#0E7490", "#A16207",
           "#BE185D", "#4D7C0F", "#334155", "#9333EA", "#0369A1", "#B45309", "#047857"]

st.markdown(
    """
    <style>
      .tok { padding: 0.05rem 0.25rem; border-radius: 3px; margin: 0.1rem; display: inline-block;
             font-size: 0.86rem; color: white; }
      .doc { font-size: 0.92rem; line-height: 1.6; background: #F6F7F9; padding: 0.7rem 0.9rem;
             border-left: 3px solid #2E5EAA; }
      .lbl { font-weight: 600; font-size: 1.02rem; }
      .meta { font-family: ui-monospace, Menlo, monospace; font-size: 0.76rem; color: #5B6573; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- data ----------------------------------------------------------------------
@st.cache_data
def load():
    a = ARTIFACT_DIR
    topics = pd.read_csv(a / "topics.csv")
    topics["label"] = topics["label"].fillna("")
    docs = pd.read_csv(a / "doc_topics.csv")
    corpus = pd.read_json(CORPUS_FILE, lines=True).set_index("doc_id")
    tokens = pd.read_json(TOKENS_FILE, lines=True).set_index("doc_id")
    metrics = json.loads((a / "metrics.json").read_text(encoding="utf-8"))
    vocab = (a / "vocab.txt").read_text(encoding="utf-8").split("\n")
    phi = {m: np.load(a / f"phi_{m}.npy").astype(float) for m in ("variational", "gibbs")}
    p_w = np.load(a / "p_w.npy").astype(float)
    assign = {}
    with (a / "gibbs_assignments.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            assign[row["doc_id"]] = row
    return dict(topics=topics, docs=docs, corpus=corpus, tokens=tokens, metrics=metrics,
                vocab=vocab, phi=phi, p_w=p_w, assign=assign,
                sweep=pd.read_csv(a / "k_sweep.csv"), trace=pd.read_csv(a / "gibbs_trace.csv"),
                matching=pd.read_csv(a / "topic_matching.csv"))


D = load()
K = D["metrics"]["settings"]["K"]
MODEL_NAMES = {"Variational Bayes": "variational", "Collapsed Gibbs sampling": "gibbs"}


def topic_name(model: str, k: int) -> str:
    row = D["topics"][(D["topics"].model == model) & (D["topics"].topic == k)].iloc[0]
    return f"T{k} · {row.label}" if row.label else f"T{k}"


# --- sidebar -------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### LDA__Topics")
    st.caption("Latent Dirichlet Allocation on 849 Reuters news articles, fitted two ways")
    model_label = st.radio("Model", list(MODEL_NAMES))
    model = MODEL_NAMES[model_label]
    lam = st.slider("Relevance λ", 0.0, 1.0, RELEVANCE_LAMBDA, 0.1,
                    help="1.0 ranks words by probability within the topic. Lower values favour words "
                         "that are specific to the topic rather than frequent everywhere "
                         "(Sievert & Shirley, 2014).")
    s = D["metrics"]["settings"]
    st.markdown(f"<div class='meta'>K={s['K']} · α={s['alpha']} · η={s['eta']}<br>"
                f"min_df={s['min_df']} · max_df={s['max_df']}<br>"
                f"Gibbs: {s['gibbs_iters']} sweeps, burn-in {s['gibbs_burn_in']}, thin {s['gibbs_thin']}"
                f"</div>", unsafe_allow_html=True)
    st.caption(f"v{__version__}")

phi = D["phi"][model]
topics_m = D["topics"][D["topics"].model == model].sort_values("topic")
docs_m = D["docs"][D["docs"].model == model].set_index("doc_id")

tabs = st.tabs(["Overview", "Topics", "Topic map", "Documents", "Choosing K",
                "Variational vs Gibbs", "Try your text"])

# --- overview --------------------------------------------------------------------
with tabs[0]:
    c = D["metrics"]["corpus"]
    cols = st.columns(5)
    cols[0].metric("Documents", c["documents"])
    cols[1].metric("Gold categories", c["categories"])
    cols[2].metric("Vocabulary", f"{c['vocabulary']:,}")
    cols[3].metric("Tokens in the DTM", f"{c['dtm_tokens']:,}")
    cols[4].metric("Topics (K)", K)
    st.markdown("#### Model quality")
    rows = []
    for name, label in (("variational", "Variational Bayes"), ("gibbs", "Collapsed Gibbs")):
        m = D["metrics"]["models"][name]
        rows.append({"model": label, "NPMI coherence ↑": m["npmi"], "UMass coherence ↑": m["umass"],
                     "held-out perplexity ↓": m["perplexity"], "topic diversity ↑": m["diversity"],
                     "NMI vs categories ↑": m["nmi"], "purity ↑": m["purity"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Coherence and diversity describe the topics themselves; perplexity is measured on "
               f"{c['heldout_docs']} held-out documents by document completion; NMI and purity compare each "
               "document's dominant topic with its Reuters category, which the model never saw.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Preprocessing, one document")
        doc_id = D["corpus"].index[0]
        doc = D["corpus"].loc[doc_id]
        st.markdown(f"<div class='doc'><b>{html.escape(doc.title)}</b><br>{html.escape(doc.text[:600])}"
                    f"{'…' if len(doc.text) > 600 else ''}</div>", unsafe_allow_html=True)
        st.markdown("**After spaCy lemmatisation, POS filtering and stopword removal**")
        st.code(" ".join(D["tokens"].loc[doc_id, "tokens"]), language="text", wrap_lines=True)
        st.caption(f"Corpus-wide: {c['raw_tokens']:,} spaCy tokens → {c['content_lemmas']:,} content lemmas "
                   f"→ {c['dtm_tokens']:,} after vocabulary pruning.")
    with right:
        st.markdown("#### Documents per gold category")
        counts = D["corpus"]["category"].value_counts().rename_axis("category").reset_index(name="documents")
        st.altair_chart(alt.Chart(counts).mark_bar(color="#2E5EAA").encode(
            x="documents:Q", y=alt.Y("category:N", sort="-x")), width="stretch")


# --- topics -------------------------------------------------------------------------
with tabs[1]:
    rel = relevance(phi, D["p_w"], lam)
    st.caption(f"{model_label}. Bars show P(word | topic); words are ranked by relevance at λ = {lam:.1f}.")
    for start in range(0, K, 2):
        cols = st.columns(2)
        for col, k in zip(cols, range(start, min(start + 2, K))):
            row = topics_m[topics_m.topic == k].iloc[0]
            order = np.argsort(-rel[k])[:12]
            frame = pd.DataFrame({"word": [D["vocab"][i] for i in order], "p": phi[k, order],
                                  "rank": range(len(order))})
            with col:
                st.markdown(f"<span class='lbl' style='color:{PALETTE[k]}'>{html.escape(topic_name(model, k))}</span>",
                            unsafe_allow_html=True)
                st.markdown(f"<div class='meta'>prevalence {row.prevalence:.1%} · NPMI {row.npmi:.3f} · "
                            f"dominant in {row.n_docs_dominant} docs, {row.category_share:.0%} of them "
                            f"'{row.dominant_category}'</div>", unsafe_allow_html=True)
                st.altair_chart(alt.Chart(frame).mark_bar(color=PALETTE[k]).encode(
                    x=alt.X("p:Q", title="P(word | topic)"),
                    y=alt.Y("word:N", sort=alt.SortField("rank"), title=None,
                            axis=alt.Axis(labelOverlap=False, labelLimit=140)),
                    tooltip=["word", alt.Tooltip("p:Q", format=".4f")]).properties(height=290),
                    width="stretch")
                reps = row.representative_docs.split()
                st.markdown("Representative: " + " · ".join(
                    f"*{html.escape(D['corpus'].loc[d, 'title'][:60])}*" for d in reps))


# --- map ----------------------------------------------------------------------------
with tabs[2]:
    st.markdown("#### Intertopic distance map")
    st.caption("Topics placed by classical multidimensional scaling of the Jensen-Shannon distances "
               "between their word distributions (the projection used by LDAvis). Circle area = prevalence. "
               "Topics close together share vocabulary.")
    frame = topics_m.assign(name=[topic_name(model, k) for k in topics_m.topic],
                            colour=[PALETTE[k] for k in topics_m.topic])
    base = alt.Chart(frame).encode(x=alt.X("x:Q", title="PC1", axis=alt.Axis(labels=False)),
                                   y=alt.Y("y:Q", title="PC2", axis=alt.Axis(labels=False)))
    circles = base.mark_circle(opacity=0.55, stroke="white").encode(
        size=alt.Size("prevalence:Q", scale=alt.Scale(range=[800, 6000]), legend=None),
        color=alt.Color("colour:N", scale=None),
        tooltip=["name", "top_words", alt.Tooltip("prevalence:Q", format=".1%")])
    text = base.mark_text(fontSize=12, fontWeight="bold").encode(text=alt.Text("topic:N"))
    st.altair_chart((circles + text).properties(height=480), width="stretch")
    st.dataframe(frame[["topic", "label", "top_words", "prevalence"]], hide_index=True, width="stretch")


# --- documents ----------------------------------------------------------------------
with tabs[3]:
    left, right = st.columns([2, 3])
    with left:
        cat = st.selectbox("Gold category", ["(all)"] + sorted(D["corpus"]["category"].unique()))
        pool = D["corpus"] if cat == "(all)" else D["corpus"][D["corpus"].category == cat]
        doc_id = st.selectbox("Document", pool.index.tolist(),
                              format_func=lambda d: f"{d} — {D['corpus'].loc[d, 'title'][:55]}")
        info = docs_m.loc[doc_id]
        mix = pd.DataFrame({"topic": [topic_name(model, k) for k in range(K)],
                            "share": [info[f"t{k}"] for k in range(K)], "k": range(K)})
        st.markdown(f"<div class='meta'>gold category: {info.category} · split: {info.split} · "
                    f"dominant topic: T{info.dominant}</div>", unsafe_allow_html=True)
        st.altair_chart(alt.Chart(mix).mark_bar().encode(
            x=alt.X("share:Q", scale=alt.Scale(domain=[0, 1]), title="θ (topic share)"),
            y=alt.Y("topic:N", sort=alt.SortField("k"), title=None),
            color=alt.Color("k:N", scale=alt.Scale(range=PALETTE[:K]), legend=None)).properties(height=260),
            width="stretch")
    with right:
        doc = D["corpus"].loc[doc_id]
        st.markdown(f"<div class='doc'><b>{html.escape(doc.title)}</b><br>{html.escape(doc.text)}</div>",
                    unsafe_allow_html=True)
        if model != "gibbs":
            st.caption("Switch the model to *Collapsed Gibbs sampling* in the sidebar to see the topic the "
                       "sampler assigned to every word of this document.")
        elif doc_id in D["assign"]:
            a = D["assign"][doc_id]
            st.markdown("**Gibbs sampler's topic for every word** (final sweep)")
            st.markdown(" ".join(f"<span class='tok' style='background:{PALETTE[z]}' title='T{z}'>"
                                 f"{html.escape(w)}</span>" for w, z in zip(a["words"], a["z"])),
                        unsafe_allow_html=True)
        else:
            st.caption("Held-out document: the sampler never saw it, so its mixture was estimated by "
                       "folding it in with the topics held fixed.")


# --- K -------------------------------------------------------------------------------
with tabs[4]:
    sw = D["sweep"]
    st.markdown("#### Choosing the number of topics")
    st.caption("Variational LDA at each K, three random seeds per K; points are means, bars ± one SD.")
    c1, c2, c3 = st.columns(3)
    band = alt.Chart(sw).mark_errorbar().encode(x="k:O", y=alt.Y("lo:Q", title="NPMI coherence"), y2="hi:Q"
                                                ).transform_calculate(lo="datum.npmi - datum.npmi_sd",
                                                                      hi="datum.npmi + datum.npmi_sd")
    c1.altair_chart(band + alt.Chart(sw).mark_line(point=True, color="#2E5EAA").encode(
        x=alt.X("k:O", title="K"), y="npmi:Q"), width="stretch")
    c2.altair_chart(alt.Chart(sw).mark_line(point=True, color="#C2410C").encode(
        x=alt.X("k:O", title="K"), y=alt.Y("perplexity:Q", scale=alt.Scale(zero=False),
                                           title="held-out perplexity")), width="stretch")
    c3.altair_chart(alt.Chart(sw).mark_line(point=True, color="#15803D").encode(
        x=alt.X("k:O", title="K"), y=alt.Y("nmi:Q", scale=alt.Scale(zero=False), title="NMI vs categories")),
        width="stretch")
    s = D["metrics"]["settings"]
    st.markdown(
        f"Coherence rises steeply up to K = {s['k_selected_by_rule']} and is flat after it: every K from "
        f"{s['k_selected_by_rule']} upward lies within one standard deviation of the best (K = {s['k_best_npmi']}). "
        "Perplexity keeps falling as K grows, as it almost always does, which is why it is not used to choose K. "
        f"The main models use **K = {K}**, inside the plateau and equal to the number of sampled categories, so "
        "the experiment can test whether LDA recovers them.")
    st.dataframe(sw.round(3), hide_index=True, width="stretch")


# --- VB vs Gibbs -----------------------------------------------------------------------
with tabs[5]:
    st.markdown("#### Do the two inference methods find the same topics?")
    st.caption("Topics matched one-to-one by the Hungarian algorithm on Jensen-Shannon distance.")
    match = D["matching"].copy()
    match["variational"] = [topic_name("variational", k) for k in match.vb_topic]
    match["gibbs"] = [topic_name("gibbs", k) for k in match.gibbs_topic]
    st.dataframe(match[["variational", "gibbs", "js_distance", "top10_overlap", "shared_words"]],
                 hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        st.markdown("**Gibbs sampler convergence**")
        line = alt.Chart(D["trace"]).mark_line(color="#7E22CE").encode(
            x=alt.X("sweep:Q"), y=alt.Y("log_likelihood:Q", scale=alt.Scale(zero=False), title="log p(w, z)"))
        rule = alt.Chart(pd.DataFrame({"sweep": [s["gibbs_burn_in"]]})).mark_rule(
            strokeDash=[4, 4], color="#64748B").encode(x="sweep:Q")
        st.altair_chart((line + rule).properties(height=280), width="stretch")
        st.caption(f"Samples are averaged from sweep {s['gibbs_burn_in']} (dashed line) onwards, "
                   f"every {s['gibbs_thin']}th sweep. The first point is the random initialisation.")
    with right:
        st.markdown(f"**Dominant topic vs gold category** — {model_label}")
        mm = D["metrics"]["models"][model]
        heat = pd.DataFrame(mm["contingency"], columns=mm["categories"])
        heat["topic"] = [topic_name(model, k) for k in range(K)]
        long = heat.melt(id_vars="topic", var_name="category", value_name="docs")
        st.altair_chart(alt.Chart(long).mark_rect().encode(
            x=alt.X("category:N"), y=alt.Y("topic:N", title=None),
            color=alt.Color("docs:Q", scale=alt.Scale(scheme="blues")), tooltip=["topic", "category", "docs"]
        ).properties(height=280) + alt.Chart(long).mark_text(fontSize=10).encode(
            x="category:N", y="topic:N", text="docs:Q"), width="stretch")


# --- try it --------------------------------------------------------------------------------
with tabs[6]:
    st.markdown("#### Infer the topics of new text")
    st.caption("The text is preprocessed exactly like the corpus, then its topic mixture is estimated with "
               "the selected model's topics held fixed (folding-in).")
    sample = ("The central bank cut its discount rate by half a point to 5.5 pct, and money market dealers "
              "expect commercial banks to lower their prime lending rates this week.")
    text = st.text_area("Text", sample, height=120)
    if st.button("Infer topics", type="primary"):
        from scipy import sparse

        from lda_topics.models import fold_in
        from lda_topics.preprocess import tokenize

        with st.spinner("Preprocessing ..."):
            toks = tokenize(text)
        index = {w: i for i, w in enumerate(D["vocab"])}
        ids = [index[t] for t in toks if t in index]
        if not ids:
            st.warning("None of the words in this text are in the model's vocabulary.")
        else:
            counts = np.bincount(ids, minlength=len(D["vocab"]))
            theta = fold_in(phi, sparse.csr_matrix(counts))[0]
            st.markdown("Words used: " + " ".join(f"`{t}`" for t in toks if t in index))
            mix = pd.DataFrame({"topic": [topic_name(model, k) for k in range(K)], "share": theta, "k": range(K)})
            st.altair_chart(alt.Chart(mix).mark_bar().encode(
                x=alt.X("share:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("topic:N", sort=alt.SortField("k")),
                color=alt.Color("k:N", scale=alt.Scale(range=PALETTE[:K]), legend=None)), width="stretch")
